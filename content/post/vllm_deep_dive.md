+++
title = 'vLLM 톺아보기'
date = 2026-09-14T17:00:00+09:00
draft = false
toc = true
tocBorder = true
tags = ['vLLM', 'Inference', 'LLM Serving', 'CUDA Graph', 'Speculative Decoding', 'MLOps', 'pinned']
+++

> **TL;DR**
> vLLM V1 엔진은 프론트엔드 · 엔진 코어 · 워커의 세 프로세스로 쪼개져 있고, 요청 하나가 토큰으로
> 나오기까지 그 경계를 여러 번 넘나든다. 이 글은 그 경로를 개요 한 장과 상세 여섯 장의 다이어그램으로
> 따라가면서, 스케줄러가 한 스텝에서 무엇을 결정하는지 · CUDA Graph가 언제 붙고 언제 떨어지는지 ·
> 어느 모델 러너가 선택되는지를 정리한다.
>
> 본 글은 vLLM 커밋 `dc36fcc` 기준이다. 다이어그램은 각각 단독 페이지로도 열 수 있으며, 전체 목록은
> [vLLM 엔진 다이어그램](/vllm/)에 모아두었다.

## 0. 왜 vLLM 내부를 봐야 하는가

서빙 프레임워크는 대개 블랙박스로 두고 처리량 숫자만 본다. 그런데 실제로 튜닝을 하다 보면
`max_num_batched_tokens`를 올렸는데 처리량이 안 오르거나, `enforce_eager=False`인데 CUDA Graph가
안 잡히거나, speculative decoding을 켰는데 오히려 느려지는 상황을 만난다.

이런 건 설정값 표를 아무리 봐도 답이 안 나온다. **어떤 판정이 어느 시점에 일어나는지**를 알아야
한다. 기동 시 한 번 정해지는 것, 매 스텝 다시 정해지는 것, 환경변수로 못 바꾸는 것이 각각 다르기
때문이다. 아래 다이어그램들은 그 경계를 드러내는 걸 목표로 그렸다.

## 1. 큰 그림: V1 엔진 실행 흐름

요청이 들어와 토큰이 스트리밍되어 나가기까지, 세 프로세스 경계를 넘나드는 왕복 경로.

{{< diagram src="/vllm/engine-flow.html" title="V1 엔진 실행 흐름" w="1360" h="700" >}}

프로세스는 셋으로 나뉜다.

- **프론트엔드 (P0)** — `AsyncLLM` / `LLMEngine`이 요청을 받고, `InputProcessor`가 토큰화와 멀티모달
  전처리를 한다. 나갈 때는 `OutputProcessor`가 detokenize와 정지 조건을 처리한다.
- **엔진 코어 (P1)** — `EngineCoreProc`의 `run_busy_loop()`이 계속 `step()`을 돌린다. 그 안에
  `Scheduler`와 `KVCacheManager`가 있다.
- **워커** — TP × PP 랭크마다 하나씩. `GPUModelRunner`가 입력 준비 → forward → 샘플링을 맡고,
  paged KV cache가 GPU HBM에 올라가 있다.

경계를 넘는 수단이 서로 다르다는 점이 중요하다. P0 ↔ P1은 **ZMQ 소켓 + msgpack**(`EngineCoreRequest`
가 들어가고 `EngineCoreOutputs`가 나온다)이고, P1 → 워커는 `rpc_broadcast_mq`로 `execute_model`을
전 랭크에 브로드캐스트한 뒤 `worker_response_mq`로 `ModelRunnerOutput`을 받는다.

결국 한 스텝은 `schedule()` → `execute_model()` → `update_from_output()`의 반복이다. 아래 절들은
이 세 구간을 하나씩 확대한 그림이다.

## 2. 엔진 기동 시퀀스

서빙이 시작되기 전까지. 워커 기동부터 KV 캐시 블록 수가 정해지기까지의 호출 순서.

{{< diagram src="/vllm/engine-startup.html" title="엔진 기동 시퀀스" w="1060" h="580" >}}

순서에 제약이 하나 있다. **가중치를 먼저 올려야 KV 캐시에 쓸 메모리를 잴 수 있다.**
`WorkerProc` 기동 → `init_device()` → `load_model()`까지 끝난 뒤에야 `profile_run()`이 더미 배치를
한 번 흘려서 피크 사용량을 재고, 남는 메모리에서 블록 수를 역산한다.

블록 수가 확정된 다음에 워커가 실제 KV 텐서를 잡고, 컴파일과 그래프 캡처가 이어진다. 기동이 느리게
느껴지는 구간은 대부분 여기다 — 그리고 이건 한 번만 일어난다.

## 3. 스케줄러 한 스텝

1절 개요의 Scheduler 구간을 확대한 그림. `schedule()`이 한 스텝에서 토큰 예산을 어떻게 나누는지.

{{< diagram src="/vllm/scheduler-step.html" title="스케줄러 한 스텝" w="1210" h="528" >}}

한 스텝에는 토큰 예산(스텝당 토큰 상한)이 있고, 스케줄러는 이걸 두 번에 걸쳐 나눠 쓴다.

1. **RUNNING 큐 먼저** — 진행 중인 디코드와 프리필에 배분한다. 각 요청마다 `num_new_tokens`를
   산정하고 `allocate_slots()`로 블록을 잡는다.
2. **남은 예산으로 WAITING 큐** — 새 요청의 프리필. `get_computed_blocks()`로 프리픽스 캐시
   히트분을 빼고, 남은 `token_budget`까지만 잘라서(청크 프리필) 넣는다. `max_num_seqs`와
   `max_loras` 상한도 여기서 걸린다.

예외 경로가 하나 있다. `allocate_slots()`가 실패하면 — 블록이 모자라면 — 최저 우선순위 요청을
**선점**해서 블록을 해제하고 재시도한다. 그리고 **선점이 한 번이라도 일어났으면 그 스텝의 WAITING
순회는 통째로 건너뛴다.** 이미 블록이 모자란 판에 새 요청을 더 받지 않겠다는 것이다.

메모리 압박이 있을 때 새 요청의 TTFT가 갑자기 튀는 현상은 대개 이 분기 때문이다.

## 4. Speculative Decoding 순환

draft가 한 스텝에 만들어져 다음 스텝에 검증되는 순환. 제안기 종류와 무관한 공통 골격.

{{< diagram src="/vllm/spec-decode.html" title="Speculative Decoding 순환" w="1000" h="500" >}}

헷갈리기 쉬운 지점은 **draft를 만드는 스텝과 검증하는 스텝이 같지 않다**는 것이다. 이번 스텝의 입력에
이미 지난 스텝이 만들어둔 draft가 섞여 들어가 있고, forward 한 번으로 그 draft를 검증하면서 동시에
다음 draft의 재료를 얻는다.

스텝 경계에서 두 가지가 정리된다. 거부된 토큰 수만큼 `num_computed_tokens`를 되감고, 새로 만든
draft를 요청에 붙인다. 블록 할당 쪽에서는 `num_lookahead_tokens`로 draft가 들어갈 KV 자리를 미리
잡아둔다 — 검증 후에 자리가 없으면 곤란하기 때문이다.

이 골격은 ngram이든 EAGLE이든 MTP든 동일하고, 제안기마다 다른 건 "다음 draft 제안" 상자의 내부다.

## 5. 그래프 모드가 갈리는 지점

엔진 전반에서 FULL · PIECEWISE · NONE이 어디서 갈라지는지. 설정 → 백엔드 능력 → 배치별 조회의 세 단계.

{{< diagram src="/vllm/graph-mode-split.html" title="그래프 모드가 갈리는 지점" w="1193" h="652" >}}

판정이 한 곳에서 끝나지 않고 세 단계로 나뉜다.

1. **설정** — `torch.compile`을 쓰지 않으면 `splitting_ops`가 비어서 PIECEWISE 자체가 성립하지 않는다.
   그래프를 쪼갤 지점이 없기 때문이다.
2. **백엔드 능력** — 어텐션 그룹 중 **가장 약한** 백엔드의 `min_cg_support`가 상한을 정한다. 백엔드
   하나가 못 따라오면 전체가 거기에 맞춰 내려간다.
3. **배치마다** — 같은 설정이어도 배치 모양에 따라 매 스텝 다시 갈린다.

"설정은 FULL로 줬는데 왜 안 걸리지"의 답은 보통 2번 아니면 3번이다. 1·2번은 기동 시 한 번 정해지고,
3번은 매 스텝 다시 판정된다는 차이를 기억해두면 로그를 읽기 쉬워진다.

`torch.compile` 쪽 배경은 이전 글 [torch.compile 탐구생활](/post/torch_compile_deep_dive/)에 정리해뒀다.

## 6. CUDA Graph 디스패치

1절 흐름의 `GPUModelRunner` 구간을 확대한 그림. 그래프를 언제 캡처하고 언제 eager로 떨어지는지.

{{< diagram src="/vllm/cudagraph-dispatch.html" title="CUDA Graph 디스패치" w="1235" h="652" >}}

앞 절의 판정이 실제로 실행되는 자리다. 기동 시 한 번 컴파일 · 커널 워밍업 · 그래프 캡처를 해두고,
런타임에는 매 스텝 배치 서술자를 만들어 캡처해둔 키와 대조한다. 맞는 키가 있으면 그 그래프를 재생하고,
없으면 FULL → PIECEWISE → NONE 순으로 내려간다. NONE까지 가면 eager 실행이다.

드래프터(speculative decoding의 제안기)는 자체 그래프를 따로 잡는데, 그 키는 target 모드에서
파생된다. V1과 V2 러너에서 파생 방식이 달라서, spec decoding을 쓰면서 러너가 바뀌면 그래프가
잡히는 모양도 같이 바뀐다.

## 7. 모델 러너 V1/V2 선택

내 설정이면 어느 러너를 쓰는지. `use_v2_model_runner`의 판정 순서.

{{< diagram src="/vllm/model-runner-gate.html" title="모델 러너 V1/V2 선택" w="1191" h="528" >}}

판정 순서가 위에서부터 내려온다.

- **강제 경로** — HiSparse와 워터마킹은 V2 고정이다. 환경변수로 끌 수 없다.
- **명시 지정** — 환경변수를 주면 아래의 자동 판정을 건너뛴다.
- **자동 판정** — 플랫폼 요건과 미지원 기능을 차례로 확인한다. 기본은 V2이고, V1은 폴백 경로다.

### V1 / V2 지원 범위

걸리는 조건이 길어서 [인덱스 페이지](/vllm/)에 목록으로 정리해두었다. 여기서 중요한 건 **걸렸을 때의
동작이 방향에 따라 다르다**는 점이다.

- V2가 못 하는 기능이면 → 조용히 **V1로 폴백**한다.
- V1이 못 하는 기능이면 → **`ValueError`로 죽는다.**

`vllm/config/vllm.py`의 서로 다른 두 함수라서 그렇다. 폴백 쪽은 에러가 안 나니까 성능이 예상과 다를 때
러너가 바뀌어 있는 걸 나중에야 알게 되는 경우가 있다.

## 8. 정리

처음 볼 때 순서를 추천하자면:

1. **1절 개요**로 프로세스 경계를 잡는다. 어디서 ZMQ를 타고 어디서 브로드캐스트인지.
2. 처리량이 문제면 **3절 스케줄러**. 토큰 예산 배분과 선점 분기가 대부분의 답을 갖고 있다.
3. 지연이 문제면 **5·6절 그래프 모드**. 기동 시 정해지는 것과 매 스텝 정해지는 것을 구분한다.
4. 기동이 느리면 **2절**. `profile_run()`과 그래프 캡처 구간이다.

각 절의 다이어그램은 전체 화면으로 열면 검색 · 포커스 · 내보내기가 된다.

---

**근거 코드**: `vllm/v1/engine/`, `vllm/v1/core/sched/scheduler.py`,
`vllm/v1/executor/multiproc_executor.py`, `vllm/v1/worker/gpu_model_runner.py`,
`vllm/v1/cudagraph_dispatcher.py`, `vllm/v1/spec_decode/`,
`vllm/config/vllm.py`, `vllm/config/compilation.py`, `vllm/compilation/backends.py`
