+++
title = 'vLLM 톺아보기'
date = 2026-09-14T17:00:00+09:00
draft = true
toc = true
tocBorder = true
tags = ['vLLM', 'Inference', 'LLM Serving', 'CUDA Graph', 'Speculative Decoding', 'MLOps']
+++

> **TL;DR**
> <!-- TODO: 한 문단 요약 -->
>
> 본 글은 vLLM 커밋 `dc36fcc` 기준이다. 각 절의 다이어그램은 단독 페이지로도 열 수 있으며,
> 전체 목록은 [vLLM 엔진 다이어그램](/vllm/)에 모아두었다.

## 0. 왜 vLLM 내부를 봐야 하는가

<!-- TODO: 문제의식. 왜 블랙박스로 두면 곤란한지 -->

## 1. 큰 그림: V1 엔진 실행 흐름

요청이 들어와 토큰이 스트리밍되어 나가기까지, 세 프로세스 경계를 넘나드는 왕복 경로.

{{< diagram src="/vllm/engine-flow.html" title="V1 엔진 실행 흐름" w="1360" h="700" >}}

<!-- TODO:
 - 프론트엔드(P0) · 엔진 코어(P1) · 워커 프로세스의 역할 분담
 - ZMQ 입출력 소켓과 rpc_broadcast_mq 구간
 - schedule → execute_model → update_from_output 순환
 - 근거: vllm/v1/engine/, vllm/v1/executor/multiproc_executor.py
-->

## 2. 엔진 기동 시퀀스

서빙이 시작되기 전까지. 워커 기동부터 KV 캐시 블록 수가 정해지기까지의 호출 순서.

{{< diagram src="/vllm/engine-startup.html" title="엔진 기동 시퀀스" w="1060" h="580" >}}

<!-- TODO:
 - WorkerProc 기동 → init_device() → load_model()
 - profile_run()으로 KV에 쓸 여유 메모리를 재는 방식
 - 블록 텐서 할당 · 컴파일 · 그래프 캡처
-->

## 3. 스케줄러 한 스텝

1절 개요의 Scheduler 구간을 확대한 그림. `schedule()`이 한 스텝에서 토큰 예산을 어떻게 나누는지.

{{< diagram src="/vllm/scheduler-step.html" title="스케줄러 한 스텝" w="1210" h="528" >}}

<!-- TODO:
 - RUNNING 큐를 먼저, 남은 예산으로 WAITING 큐
 - 블록이 모자랄 때의 선점과 재시도 루프
 - 프리픽스 캐시 히트와 청크 프리필 상한
 - 근거: vllm/v1/core/sched/scheduler.py
-->

## 4. Speculative Decoding 순환

draft가 한 스텝에 만들어져 다음 스텝에 검증되는 순환. 제안기 종류와 무관한 공통 골격.

{{< diagram src="/vllm/spec-decode.html" title="Speculative Decoding 순환" w="1000" h="500" >}}

<!-- TODO:
 - 지난 draft 검증 → 수락 → 다음 draft 제안
 - 거부분만큼 num_computed_tokens 롤백
 - num_lookahead_tokens로 KV 자리를 선예약하는 이유
 - 근거: vllm/v1/spec_decode/
-->

## 5. 그래프 모드가 갈리는 지점

엔진 전반에서 FULL · PIECEWISE · NONE이 어디서 갈라지는지. 설정 → 백엔드 능력 → 배치별 조회의 세 단계.

{{< diagram src="/vllm/graph-mode-split.html" title="그래프 모드가 갈리는 지점" w="1193" h="652" >}}

<!-- TODO:
 - VLLM_COMPILE이 아니면 PIECEWISE 자체가 불가
 - 어텐션 백엔드의 min_cg_support가 상한을 결정
 - 같은 설정이어도 배치 모양에 따라 매 스텝 재판정
 - 근거: vllm/config/compilation.py, vllm/compilation/backends.py
-->

## 6. CUDA Graph 디스패치

1절 흐름의 GPUModelRunner 구간을 확대한 그림. 그래프를 언제 캡처하고 언제 eager로 떨어지는지.

{{< diagram src="/vllm/cudagraph-dispatch.html" title="CUDA Graph 디스패치" w="1235" h="652" >}}

<!-- TODO:
 - 기동 1회: 컴파일 · 커널 워밍업 · 그래프 캡처
 - 스텝마다: FULL → PIECEWISE → NONE 키 조회
 - 드래프터 그래프는 target 모드에서 파생 (V1/V2 다름)
 - 근거: vllm/v1/cudagraph_dispatcher.py
 - torch.compile 쪽 배경은 이전 글([torch.compile 탐구생활](/post/torch_compile_deep_dive/)) 참고
-->

## 7. 모델 러너 V1/V2 선택

내 설정이면 어느 러너를 쓰는지. `use_v2_model_runner`의 판정 순서.

{{< diagram src="/vllm/model-runner-gate.html" title="모델 러너 V1/V2 선택" w="1191" h="528" >}}

<!-- TODO:
 - 기본은 V2, V1은 폴백 경로
 - HiSparse · 워터마킹은 V2 고정
 - 플랫폼 요건과 미지원 기능 확인
 - 근거: vllm/config/vllm.py (서로 다른 두 함수이고 걸렸을 때 동작도 다름)
-->

### V1 / V2 지원 범위

걸리는 조건이 길어서 [인덱스 페이지](/vllm/)에 목록으로 정리해두었다.
V2가 못 하면 V1로 조용히 폴백하고, V1이 못 하면 `ValueError`로 죽는다.

<!-- TODO: 이 차이(폴백 vs 예외)가 왜 갈리는지 한 문단 -->

## 8. 정리

<!-- TODO: 실무에서 어디를 먼저 보면 되는지 -->

---

**근거 코드**: `vllm/v1/engine/`, `vllm/v1/core/sched/scheduler.py`,
`vllm/v1/executor/multiproc_executor.py`, `vllm/v1/worker/gpu_model_runner.py`,
`vllm/v1/cudagraph_dispatcher.py`, `vllm/v1/spec_decode/`,
`vllm/config/vllm.py`, `vllm/config/compilation.py`, `vllm/compilation/backends.py`
