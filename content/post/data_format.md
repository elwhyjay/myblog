+++
title = 'Data_format'
date = 2025-12-28T21:59:14+09:00
draft = false
mathjax = true
tags = ["AI", "GPU", "LLM","CUDA"]
+++

# 현대 AI 데이터 타입: ML 엔지니어와 CUDA 개발자를 위한 기술 심층 분석

저정밀도 수치 포맷은 효율적인 AI 추론의 핵심이 되었으며, **FP8**은 Hopper GPU에서 **2배의 처리량 증가**를 제공하면서 FP16에 가까운 정확도를 유지하고, **INT4 가중치 전용(weight-only) 양자화**는 70B 파라미터 모델을 단일 48GB GPU에서 실행할 수 있게 해준다. 이 보고서는 15개 이상의 데이터 포맷에 대한 수학적 명세, 4세대 NVIDIA 아키텍처 전반에 걸친 하드웨어별 성능 특성, 그리고 양자화 알고리즘에 대한 정량적 분석을 제공한다. 이는 프로덕션 LLM 시스템을 배포하는 데 필수적인 지식이다.

## IEEE 754 기초와 표준 부동소수점의 진화

IEEE 754-2008 표준은 모든 최신 AI 데이터 타입이 구축되거나 수정된 수학적 기반을 정의한다. 정규화된 부동소수점 값의 경우:

$$v = (-1)^S \times 2^{(E-\text{bias})} \times \left(1 + \frac{M}{2^m}\right)$$

여기서 *S*는 부호 비트, *E*는 저장된 지수(exponent), *bias*는 지수 오프셋, *M*은 가수(mantissa), *m*은 가수 비트 폭이다.

FP32 (binary32)는 1개의 부호 비트, 8개의 지수 비트(bias=127), 23개의 가수 비트를 사용하여 ±3.4×10³⁸의 다이나믹 레인지와 약 7.22 십진수 자릿수의 정밀도를 제공한다. 머신 엡실론 ε = 2⁻²⁴ ≈ 5.96×10⁻⁸은 1.0에서 표현 가능한 가장 작은 차이를 정의한다.

FP16 (binary16)은 5개의 지수 비트(bias=15)와 10개의 가수 비트를 포함하여 16비트로 압축된다. 표현 가능한 최대 값은 **65,504**로 떨어지며, 정밀도는 약 3.31 십진수 자릿수로 낮아진다. 이 포맷은 혼합 정밀도(mixed-precision) 기법이 FP32와 동등한 수렴성을 보여준 후 학습에 널리 사용되게 되었다.

BF16 (Brain Floating Point)은 근본적으로 다른 접근 방식을 취한다. FP32의 8비트 지수를 유지(±3.4×10³⁸ 다이나믹 레인지 보존)하면서 가수를 7비트로 잘라낸다. 이 설계는 하위 16비트를 버리는 것만으로 FP32에서의 직접적인 절삭(truncation) 변환을 가능하게 하여, 기울기(gradient) 크기가 크게 변하는 학습 워크플로우에 이상적이다.

| Format | Sign | Exp | Mantissa | Bias | Max Value | Min Normal | Precision (ε) |
|--------|------|-----|----------|------|-----------|------------|---------------|
| FP32 | 1 | 8 | 23 | 127 | 3.4×10³⁸ | 1.2×10⁻³⁸ | 2⁻²⁴ |
| FP16 | 1 | 5 | 10 | 15 | 65,504 | 6.1×10⁻⁵ | 2⁻¹¹ |
| BF16 | 1 | 8 | 7 | 127 | 3.4×10³⁸ | 1.2×10⁻³⁸ | 2⁻⁸ |
| TF32 | 1 | 8 | 10 | 127 | 3.4×10³⁸ | 1.2×10⁻³⁸ | 2⁻¹¹ |

TF32 (TensorFloat-32)는 텐서 코어(Tensor Cores)를 위한 NVIDIA의 실용적인 타협안이다. 총 19비트(8 지수 + 10 가수)로 FP32의 다이나믹 레인지와 FP16 수준의 정밀도를 달성한다. 결정적으로, TF32는 저장 포맷이 아닌 연산 모드이다. 입력은 FP32로 저장되고, 곱셈을 위해 TF32 정밀도로 반올림된 다음, 다시 FP32로 누적된다.

## 정밀도와 처리량의 경계를 재정의하는 FP8 포맷

NVIDIA/Intel/ARM 공동 사양은 서로 다른 신경망 연산에 최적화된 두 가지 상호 보완적인 8비트 포맷을 정의한다.

FP8 E4M3 (4 지수, 3 가수 비트, bias=7)은 좁은 다이나믹 레인지 내에서 정밀도를 극대화한다. 최대 값은 ±448이며 최소 정규 값은 2⁻⁶ ≈ 0.0156이다. 이 명세는 무한대(infinity) 인코딩을 의도적으로 제거하고 E=1111, M=111을 NaN으로 예약하여 IEEE 호환 대안에 비해 사용 가능한 범위를 확장했다.

FP8 E5M2 (5 지수, 2 가수 비트, bias=15)는 정밀도를 희생하여 FP16 수준의 다이나믹 레인지(±57,344)를 제공한다. 이 포맷은 IEEE 호환 무한대 및 NaN 표현을 유지하므로, 크기 변동성이 정밀도 요구 사항을 초과하는 기울기(gradient)에 적합하다.

권장 사용 패턴: 순전파(가중치 및 활성화)에는 E4M3, 역전파(기울기)에는 E5M2. 제한된 다이나믹 레인지를 보상하기 위해 텐서별 스케일링 팩터를 사용한다:

$$x_{\text{scaled}} = x \times s_{\text{tensor}}, \quad s_{\text{tensor}} = \frac{\max(|x|)}{448}$$

| Format | Max Value | Min Normal | Precision | Special Values | Primary Use |
|--------|-----------|------------|-----------|----------------|-------------|
| FP8 E4M3 | ±448 | 0.0156 | 2⁻⁴ | No ∞, limited NaN | Forward pass |
| FP8 E5M2 | ±57,344 | 6.1×10⁻⁵ | 2⁻³ | IEEE-compliant | Backward pass |

## 8비트 미만 포맷을 통한 극단적인 압축

INT8 양자화는 아핀 변환(affine transformation)을 사용하여 연속적인 값을 256개의 이산 레벨로 매핑한다:

$$Q(x) = \text{round}\left(\frac{x}{s}\right) + z, \quad s = \frac{\max - \min}{2^8 - 1}$$

Signed INT8은 [-128, +127] 범위를, Unsigned는 [0, 255] 범위를 커버한다. 양자화 오차는 다음과 같다:

$$\text{SQNR} \approx 6.02b + 1.76 \text{ dB}$$

8비트의 경우 약 50 dB의 신호 대 양자화 잡음비(SQNR)를 제공한다.

INT4는 16개의 이산 레벨(signed: [-8, +7])로 더 압축하며, 보통 패킹되어(바이트당 두 값) 저장된다. 32-128 요소 크기의 블록 단위 양자화(Group-wise quantization)는 그룹별로 별도의 스케일 팩터를 제공하여 균일한 텐서별 양자화에서 손실된 정확도를 복구한다.

NF4 (Normal Float 4-bit)는 정보 이론적 접근 방식을 취한다. 균일한 간격 대신, 16개의 코드북 값이 표준 정규 분포 하에서 동일한 면적을 생성하도록 배치된다. 값 {-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0}은 정규 분포를 따르는 신경망 가중치에 대한 예상 양자화 오차를 최소화하며, 이는 QLoRA 효율성의 기초가 된다.

FP4 E2M1 (NVIDIA Blackwell의 기본 포맷)은 bias=1인 2개의 지수 비트와 1개의 가수 비트를 사용하여 {0, 0.5, 1, 1.5, 2, 3, 4, 6} 및 그 음수 값을 생성한다. NVIDIA의 NVFP4 구현은 블록별 FP8 E4M3 스케일링 (16개 요소 블록)과 텐서별 FP32 스케일링을 추가한다:

$$x = x_q \times s_{\text{block}} \times s_{\text{tensor}}$$

## OCP Microscaling 포맷: 블록 부동소수점의 표준화

OCP(Open Compute Project)의 MX 사양(AMD, ARM, Intel, Meta, Microsoft, NVIDIA, Qualcomm 승인)은 32개 요소가 단일 E8M0 지수를 공유하는 블록 부동소수점 포맷을 정의한다.

| Format | Element Format | Element Bits | Block Size | Scale Format | Effective bits/value |
|--------|---------------|--------------|------------|--------------|---------------------|
| MXFP8 | E4M3 or E5M2 | 8 | 32 | E8M0 (8-bit) | 8.25 |
| MXFP6 | E3M2 or E2M3 | 6 | 32 | E8M0 (8-bit) | 6.25 |
| MXFP4 | E2M1 | 4 | 32 | E8M0 (8-bit) | 4.25 |
| MXINT8 | INT8 | 8 | 32 | E8M0 (8-bit) | 8.25 |

E8M0 스케일 포맷은 가수 없이 순수 지수만으로 2⁻¹²⁷에서 2¹²⁷까지의 2의 거듭제곱을 나타낸다. 32개 요소의 MXFP4 블록의 경우: 32×4 + 8 = 136비트, 즉 **요소당 4.25비트**로 스칼라 FP4 대비 6.25%의 오버헤드만 발생한다.

Microsoft Research의 주요 발견: MXFP4 가중치 + MXFP6 활성화를 사용하면 FP32 베이스라인의 0.3% 이내로 학습 수렴을 달성할 수 있다.

## NVIDIA 아키텍처의 발전과 포맷 

각 GPU 세대는 텐서 코어 기능과 데이터 타입 지원을 확장한다.

### Ampere (SM80/SM86) - 3세대 텐서 코어

A100은 투명한 FP32 가속기로서 TF32와, 2배의 유효 처리량을 위한 구조적 2:4 희소성(sparsity)을 도입했다. 주요 사양:

| Data Type | Dense TFLOPS | Sparse TFLOPS (2:4) |
|-----------|-------------|---------------------|
| TF32 | 156 | 312 |
| BF16/FP16 | 312 | 624 |
| INT8 | 624 TOPS | 1,248 TOPS |
| INT4 | 1,248 TOPS | 2,496 TOPS |

메모리 대역폭: 2,039 GB/s (HBM2e). WMMA 행렬 모양: FP16의 경우 m16n16k16, TF32의 경우 m16n16k8.

### Ada Lovelace (SM89) - 4세대 텐서 코어

L40S와 RTX 4090은 전체 트랜스포머 엔진 없이 FP8 지원을 추가했다. L40S는 733 TFLOPS (FP8 dense), 희소성 적용 시 1,466 TFLOPS를 달성한다. 메모리 대역폭은 864 GB/s(GDDR6)로 떨어져, HBM 시스템에 비해 연산 제한(compute-bound)적인 성격을 띤다.

### Hopper (SM90) - 트랜스포머 엔진의 데뷔

H100은 레이어별로 자동 FP8/FP16 정밀도 관리를 제공하는 트랜스포머 엔진과 함께 근본적인 도약을 보여준다.

| Data Type | Dense TFLOPS | Sparse TFLOPS (2:4) |
|-----------|-------------|---------------------|
| TF32 | 500 | 1,000 |
| BF16/FP16 | 1,000 | 2,000 |
| FP8 | 2,000 | 4,000 |
| INT8 | 2,000 TOPS | 4,000 TOPS |

메모리 대역폭: 3.35 TB/s (HBM3, H100) ~ 4.8 TB/s (HBM3e, H200). H200은 FP8 처리량을 3,958 TFLOPS(dense)까지 끌어올린다.

Hopper는 스레드 블록 클러스터(SM 간 협력)와 비동기 텐서 데이터 이동을 위한 TMA (Tensor Memory Accelerator)를 도입했다. 이는 이러한 처리량 수준에서 메모리 대기 시간을 숨기는 데 매우 중요하다.

### Blackwell (SM100) - 네이티브 FP4 및 MX 포맷

B200은 FP4를 지원하는 2세대 트랜스포머 엔진으로 성능을 다시 한 번 두 배로 높인다.

| Data Type | Dense TFLOPS | Sparse TFLOPS (2:4) |
|-----------|-------------|---------------------|
| TF32 | 1,200 | 2,250 |
| BF16/FP16 | 2,250 | 4,500 |
| FP8/FP6 | 4,500 | 9,000 |
| **FP4** | 9,000 | 18,000 |

메모리 대역폭: **8 TB/s** (HBM3e), 용량: 192GB. 10 TB/s NV-HBI 인터커넥트를 갖춘 2080억 트랜지스터 듀얼 다이 설계가 이러한 사양을 가능하게 한다.

## 양자화 알고리즘: 정확도와 효율성의 균형

### GPTQ: 2차 정보를 활용

GPTQ (Frantar et al., ICLR 2023)는 역헤시안(inverse Hessian) 정보를 사용하여 레이어별 재구성 오차를 최소화한다:

$$\arg\min_{\hat{W}} \|WX - \hat{W}X\|_2^2$$

헤시안 H = 2XXᵀ + λI는 가중치 섭동(perturbation)에 대한 출력 민감도를 포착한다. 각 양자화된 가중치에 대해:

<div>
$$\delta_F = -\frac{[H^{-1}]_{:q}}{[H^{-1}]_{qq}} \cdot (\operatorname{quant}(w_q) - w_q)$$
</div>

이 오차 보상은 남아있는 비양자화 가중치로 전파된다. GPTQ의 핵심 혁신: 고정 순서 양자화(열 단위)는 숄레스키 분해를 통해 역 헤세 행렬을 한 번만 계산하게 하여, 레이어당 복잡도를 O(d³)에서 O(d²)로 줄인다.

성능: 175B 모델을 약 4 GPU 시간 만에 3-4비트로 양자화함. FP16 대비 A100에서 3.25배, A6000에서 4.5배의 속도 향상을 달성.

### AWQ: 활성화 인식 중요 가중치 식별

AWQ (Lin et al., MLSys 2024)는 단 1%의 가중치(크기가 큰 활성화와 정렬된 가중치)를 보호하는 것만으로도 양자화 오차를 획기적으로 줄일 수 있음을 관찰했다:

$$\text{Importance}(w_j) \propto \mathbb{E}[|X_{:,j}|]$$

AWQ는 (하드웨어적으로 비효율적인) 혼합 정밀도 대신 동등한 채널별 스케일링을 적용한다:

$$Y = XW = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$

스케일 팩터는 이전 LayerNorm으로 융합(fused)될 수 있어 하드웨어 효율성을 유지하면서 낮은 활성화 채널의 양자화 오차를 "용인"한다. AWQ는 모델 크기 전반에 걸쳐 정확도 보존 면에서 GPTQ보다 일관되게 우수하다.

### SmoothQuant: 어려움을 활성화에서 가중치로 이동

LLM 활성화에는 일반적인 값보다 약 100배 큰 이상치(outliers)가 고정된 채널에 집중되어 있다. SmoothQuant (Xiao et al., ICML 2023)는 채널별 스케일링을 도입한다:

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

이동 인자(migration factor) α는 난이도 분포를 제어한다. α=0.5는 활성화/가중치 양자화의 균형을 맞추고(OPT, BLOOM에 최적), α=0.75는 심각한 이상치가 있는 모델(GLM-130B)에 적합하다. 이는 INT8 GEMM 하드웨어 가속을 사용하는 W8A8 양자화를 가능하게 하여, 2배의 메모리 감소와 함께 최대 1.56배의 속도 향상을 제공한다.

## 메모리 대역폭이 지배하는 LLM 추론 경제학

연산 강도(Arithmetic intensity)(FLOPs/byte)는 추론이 메모리 제한적인지 연산 제한적인지를 결정한다:

$$\text{Critical Batch Size} = \frac{\text{Accelerator FLOPS/s}}{\text{Memory Bandwidth}} \times \frac{\operatorname{bits_{param}}}{\operatorname{bits_{activation}}}$$

H100에서 BF16 사용 시: B_crit ≈ 280 토큰. 이 임계값 아래에서는 추론이 메모리 제한(memory-bound) 상태이며, 성능은 데이터 전송 감소에 비례하므로 가중치 양자화가 매우 효과적이다.

KV-캐시 메모리는 시퀀스 길이와 배치 크기에 따라 선형적으로 증가한다:

$$\text{KV Size} = 2 \times L \times n_{kv} \times d_h \times \text{seq\_len} \times \text{batch} \times \text{bytes}$$

Llama-2-13B의 경우 BF16에서 시퀀스 길이 8192일 때: 시퀀스당 6.7GB. 4개 시퀀스만 되어도 KV-캐시가 모델 파라미터 메모리를 초과한다.

### KV-캐시 양자화 기법

KIVI (ICML 2024)는 비대칭 분포를 활용한다. Key는 고정 채널 이상치를 가지므로 채널별 양자화를 사용하고, Value는 뚜렷한 패턴이 없으므로 토큰별 양자화를 사용한다. 결과: 2비트 KV-캐시로 2.6배의 메모리 감소와 2.35-3.47배의 처리량 향상 달성.

Hopper에서의 FP8 KV-캐시는 최소한의 정확도 영향으로 2배 압축을 제공한다. NVIDIA는 더 나은 정확도 보존을 위해 KV-캐시에 INT8보다 FP8을 권장한다.

### 추측 디코딩(Speculative decoding)을 통한 자기회귀 생성 가속

EAGLE (특징 레벨 추측)은 토큰 대신 두 번째 상위 레이어의 활성화를 예측하여, Medusa의 ~0.6 대비 ~0.8의 초안 정확도(draft accuracy)를 달성한다. 다중 레이어 융합을 사용하는 EAGLE-3는 동일한 출력 분포를 유지하면서 최대 6.5배의 속도 향상(무손실 가속)에 도달한다.

| Method | Draft Accuracy | Typical Speedup | Architecture |
|--------|---------------|-----------------|--------------|
| Standard Speculative | 0.4-0.6 | 1.5-2× | Separate draft model |
| Medusa | ~0.6 | ~2× | Multiple prediction heads |
| EAGLE | ~0.8 | 2-3× | Feature-level prediction |
| EAGLE-3 | Higher | Up to 6.5× | Multi-layer fusion |

## 프레임워크 생태계를 통한 실전 배포

### TorchAO: PyTorch 네이티브 양자화

TorchAO는 커널 융합을 위해 `torch.compile`과 직접 통합된다:

```python
from torchao.quantization import quantize_, Int4WeightOnlyConfig
quantize_(model, Int4WeightOnlyConfig(group_size=32))
```

INT4/INT8 가중치 전용, FP8 동적 양자화 및 QAT 워크플로우를 지원한다. 성능: Llama-3-8B에서 INT4 + 2:4 희소성은 67.7%의 메모리 감소와 함께 2.37배의 처리량을 달성한다.

### vLLM 및 SGLang: 서빙 처리량 최적화

vLLM은 효율적인 KV-캐시 관리를 위해 PagedAttention과 함께 GPTQ, AWQ, FP8, INT8을 지원한다. FP8 W8A8은 2배의 메모리 감소와 최대 1.6배의 처리량 향상을 달성한다.

SGLang은 자동 KV-캐시 접두사 재사용을 위한 RadixAttention을 추가하여, 멀티 턴/퓨샷 워크로드에서 베이스라인 대비 최대 6.4배의 처리량을 제공한다. GPU당 3.56배 더 많은 토큰을 위한 실험적인 FP4 KV-캐시를 지원한다.

### TensorRT-LLM: NVIDIA 하드웨어 활용 극대화

권장 양자화 계층 구조: FP8 (최고의 정확도/성능)로 시작하여, INT8 SmoothQuant로 대체, 메모리 제약이 있을 경우 INT4 AWQ/GPTQ 사용.

| Batch Size | Bottleneck | Recommended Format |
|------------|-----------|-------------------|
| ≤4 | Memory bandwidth | INT4 weight-only (AWQ) |
| 5-15 | Mixed | FP8 or INT8 SmoothQuant |
| ≥16 | Compute + Memory | FP8 (W8A8) |

### llama.cpp: 엣지 배포 활성화

K-quants를 사용하는 GGUF 포맷은 품질 인식 양자화를 제공한다:

| Type | Size (7B) | Perplexity Increase | Recommendation |
|------|-----------|---------------------|----------------|
| Q8_0 | 8.54 GB | Minimal | Maximum quality |
| Q5_K_M | 5.73 GB | +0.035 | Quality/size balance |
| Q4_K_M | 4.92 GB | +0.054 | Default choice |
| Q3_K_M | 4.02 GB | +0.244 | Memory-constrained |

중요도 행렬 (imatrix) 보정은 Q3 이하에서 필수적이며, 가중치 민감도에 따라 비트를 할당하기 위해 활성화 통계를 사용한다.

## 정량적 벤치마크를 통한 배포 결정 가이드

### 정밀도별 처리량 비교

H100에서의 Llama 3-8B:
- FP16: 135.79 tokens/sec (베이스라인)
- INT8: 158.90 tokens/sec (1.17배)
- INT4: 211.50 tokens/sec (1.56배)

Atom W4A4 양자화: 동등한 메모리에서 FP16 대비 7.73배, INT8 대비 2.53배 처리량.

Mistral 7B FP8 대 FP16: 출력 토큰 33% 향상, TTFT 8.5% 감소, VRAM: 7GB 대 16GB.

### 정확도 저하 패턴

2B-405B 모델에 대한 포괄적인 평가 ([Jemin Lee et al., 2025](https://www.ijcai.org/proceedings/2025/0902.pdf)):
- 모든 양자화 방식은 표준 벤치마크에서 0-2%의 정확도 차이를 보임
- 가중치 전용 양자화에서 AWQ가 GPTQ보다 일관되게 우수함
- 더 큰 모델(70B+)이 공격적인 양자화를 더 잘 견딤 (4-bit Llama-2-13B가 FP16 Llama-2-7B를 능가)
- 작업 민감도는 다양함: 일반 지식 작업보다 TruthfulQA 및 지시 따르기(instruction-following)에서 더 높은 저하를 보임

### 프로덕션 배포를 위한 메모리 요구 사항

| Model | FP16 | INT8 | INT4 (AWQ) |
|-------|------|------|------------|
| Llama 3.1-8B | ~16 GB | ~8 GB | ~4 GB |
| Llama 3.1-70B | ~140 GB | ~70 GB | ~35 GB |
| Llama 3.1-405B | ~810 GB | ~405 GB | ~203 GB |

배포 구성: 8B는 단일 A10G/L4에 적합; 70B FP16은 4× A100 또는 2× A100에서 AWQ 필요; 405B FP8은 8× H100 또는 8× A100에서 AWQ 필요.

## 결론

AI 수치 포맷의 지형은 단순한 IEEE 754 준수에서 신경망 특성에 최적화된 도메인 특화 타입의 풍부한 생태계로 진화했다. 세 가지 핵심 통찰이 드러난다:

첫째, 포맷 선택은 메모리/연산 병목 체계에 결정적으로 의존한다. 메모리 대역폭이 지배적인 작은 배치 크기에서는 INT4 가중치 전용이 뛰어나고, 연산 활용도가 중요한 큰 배치에서는 FP8 W8A8이 최적이다.

둘째, OCP Microscaling 명세는 블록 부동소수점 접근 방식에 대한 업계의 수렴을 보여주며, MXFP4는 공유 지수 메커니즘을 통해 4배의 메모리 감소로 FP32와 동등한 학습 품질을 달성한다.

셋째, 프로덕션 배포는 하드웨어 기능을 포맷 요구 사항과 일치시켜야 한다. FP8 텐서 코어는 Hopper/Ada(SM89+)를 필요로 하고, FP4 네이티브 지원은 Blackwell(SM100)을 필요로 하며, 최적의 커널 선택은 배치 크기와 시퀀스 길이에 따라 달라진다.

2025년 NVIDIA 하드웨어를 목표로 하는 실무자를 위한 제언: 최고의 정확도/성능 균형을 위해 Hopper에서 FP8로 시작하고, 메모리 제약 시나리오에는 AWQ INT4를 사용하며, 도구 성숙도에 따라 Blackwell에서의 NVFP4 채택을 준비하라. 다이나믹 레인지 계산, 정밀도 손실 모델링, 양자화 오차 공식과 같은 수학적 기초는 확장되는 포맷 환경에서 정보에 입각한 트레이드오프 결정을 내리기 위한 분석적 프레임워크를 제공한다.

# THIS ARTICLE IS AUTO GENERATED FROM CLAUDE.