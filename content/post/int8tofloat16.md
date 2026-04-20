+++
title = 'Int4 to FP16 dequantization optimization'
date = 2025-12-02T17:44:26+09:00
draft = false
mathjax = true
tags = ['Quantization', 'GPU', 'Optimization','pinned']
+++

## Introduction

Quantization은 현재 LLM 분야에서 중요한 기술로 자리잡았다. 특히 추론을 최적화하는데 있어서 가중치를 int4, int8 등의 저정밀도 정수형으로 양자화하는 기법이 널리 사용되고 있다. 이 기법을 통해 모델의 메모리 사용량을 줄이고, 메모리 대역폭 요구사항을 낮추며, 하드웨어 가속기의 연산 처리량을 극대화할 수 있다. 하지만 양자화 기술은 만능이 아니고 많은 경우에 최적화가 요구될 수 있다.

이 글에서는 QServe에서 소개된 [Kim et al.](https://arxiv.org/abs/2211.10017) 의 논문에서 제시된 int4 양자화된 KV cache를 FP16 활성화 값으로 역양자화하는 비트 최적화 기법에 대해 살펴본다. 역양자화는 때로 과도한 연산량을 요구할 수 있기 때문에, 이를 효율적으로 처리하는 것이 중요하다. 이 최적화 기법은 computing resource를 절약하는 효과적인 방법중 하나로 이해하고 비슷한시나리오에서 사용가능할것 같다.

---

## FP16 (반정밀도 부동소수점) 기본 구조

FP16은 실수(부동소수점 수)를 표현하는 16비트 포맷이다. 이는 표준 단정밀도(FP32, 32비트) 대비 메모리 효율성 및 연산 처리량 증대에 유리하다.

FP16은 IEEE 754 표준에 따라 다음과 같이 16비트를 세 부분으로 나누어 구성된다:

- **부호 (Sign)**: 1 비트
- **지수 (Exponent)**: 5 비트 (바이어스 $Bias=15$)
- **가수/유효숫자 (Mantissa/Significand)**: 10 비트

FP16으로 표현되는 값 $V$의 일반적인 공식은 다음과 같다:

$$V = (-1)^{Sign} \times 2^{Exponent - Bias} \times (1 + Fraction)$$

### 암묵적 선행 비트의 역할

유효숫자는 일반적으로 **정규화된 수(Normalized Number)**에서 암묵적인 선행 **$1$**을 가정하여 11비트의 정밀도를 확보한다.

- **정규화된 수**: 지수 필드가 $0$이나 최대값($31$)이 아닐 때, 선행 비트는 **$1$**로 가정된다.
- **비정규화된 수 (Denormalized Number)**: 지수 필드가 **모두 $0$**일 때, 선행 비트는 **$0$**으로 가정된다. 이는 $0$ 주변의 매우 작은 수를 표현하여 점진적인 언더플로우를 가능하게 한다.

---

## 정수 변환과 $2^{10}$ 분기점

정수 $N$을 FP16으로 변환할 때, $2^{10}=1024$는 정확도와 표현 간격이 바뀌는 중요한 분기점이다.

1. **$N < 1024$인 경우**: 정규화 시 지수 $E \le 9$를 갖는다. 가수에 필요한 비트 수가 10비트 이하여서, $1024$ 미만의 모든 정수는 FP16으로 **오차 없이 정확하게 표현된다.**

2. **$N \ge 1024$인 경우**: 정규화 시 지수 $E \ge 10$을 갖는다.
    - $N=1027$을 예로 들면, 정규화 형태는 $1.0000000011_2 \times 2^{10}$이다.
    - $V = (1 + Fraction) \times 2^{10}$의 형태에서, 가수의 최소 단위 $ULP = 2^{-10}$이 $2^{10}$과 곱해지면서 표현 간격 $\Delta V$는 $1$이 된다.
        $$\Delta V = 2^{-10} \times 2^{10} = 1$$
    - 따라서 $1024$ 이상의 정수는 **$1$ 간격으로 떨어져 있는 정수**만 오차 없이 표현할 수 있다.


## Int4 $\to$ FP16 역양자화 최적화 기법

대규모 MoE(Mixture of Experts) 모델의 추론 최적화 과정에서, 4/8비트 정수형 가중치($Int8/Int4$)를 FP16 활성화 값으로 역양자화(Dequantize)하는 과정의 성능 향상을 위해 FP16의 비트 패턴 특징이 활용된다. 이 방법은 느린 네이티브 $Int \to Float$ 변환(I2F)을 대체한다.

### 핵심 관찰 사항

1. **관찰 1**: FP16에서 $1024 < X < 2048$ 범위의 정수 $X$는 $1024$가 지수 비트에 표현되고, $int(X-1024)$는 **가수 비트에 직접 저장된다**.

2. **관찰 2**: $0 \le Y < 1024$인 정수 $Y$에 대해, $Y+1024$의 FP16 표현은 $1024$의 16진수 표현($0x6400$)에 $Y$를 **OR 연산**하여 쉽게 만들 수 있다.

### Int4 역양자화 과정 최적화

1. **오프셋 추가**: 부호 있는 $Int4$ 가중치에 $\mathbf{128}$을 더하여 $unsigned$ $Int4$ 값($W_{+}$)을 만든다.

2. **고속 FP16 생성**: $W_{+}$의 값($e_i$)에 **$1024$를 더한 형태**($e_i + 1024$)의 FP16 비트 패턴을 **관찰 2**를 통해 **비트 연산**으로 생성한다.

3. **오프셋 제거**: 생성된 FP16 값은 $1024$ (비트 트릭 오프셋)와 $128$ (부호 변환 오프셋)을 포함하고 있다. 따라서 **총 오프셋 $\mathbf{1152}$**를 $FP16$ 부동소수점 뺄셈 연산으로 제거하여 원래의 부호 있는 $Int8$ 값을 $FP16$으로 복구한다.

이러한 최적화는 $Int \to Float$ 변환을 고속 ALU 및 $FP16$ 명령어로 대체하며, 역양자화 단계를 GEMM 커널에 융합하여 메모리 트래픽 병목 현상을 해소한다.


## Int4 역양자화 커널 구현 예제

다음은 Int4 양자화 가중치를 FP16으로 역양자화하는 CUDA 커널 구현이다.
출처는 다음 [github](https://github.com/mit-han-lab/omniserve)를 참조하라. 

```c
inline __device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source) {
    uint4 result;

    uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
    static constexpr uint32_t TOP_MASK              = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction.
    const uint32_t top_i4s = i4s >> 8;

    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[0])
                    : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[1])
                    : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[2])
                    : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[3])
                    : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // FP16 magic numbers
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;  // {1024, 1024}
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;       // {1/16, 1/16}
    static constexpr uint32_t NEG_64 = 0xd400d400;              // {-64, -64}

    // Finally, we construct the output numbers.
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

    return result;
}
```

### 구현 분석

이 커널은 다음과 같은 특징을 갖는다:

- $Int4$로 양자화된 8개의 가중치 요소($e_0$ ~ $e_7$)를 담고 있는 단일 $uint32\_t$ 변수(`source`)를 입력받아, $half2$ (32비트 레지스터에 FP16 2개) 형식의 4개 레지스터(`uint4 result`)에 담긴 $FP16$ 결과로 역양자화한다.
- **최적화 목표**: $Int \to Float$ 변환 명령어 대신 **비트 연산 (LOP3)**과 **고속 $FP16$ 연산 (SUB.F16X2, FMA.RN.F16X2)**을 사용하여 처리량(throughput)을 높인다.
- `source` ($uint32\_t$)는 8개의 $Int4$ 가중치 요소($e_0$ ~ $e_7$)를 담고 있으며, $\mathbf{8}$이 더해져 **부호 없는($W_{+}$) 상태**이다.

### 주요 상수 정의

| **상수** | **값** | **의미 (FP16 최적화 관점)** |
| --- | --- | --- |
| $\text{I4s\_TO\_F16s\_MAGIC\_NUM}$ | $\mathbf{0x64006400}$ | $half2$로 표현된 $\mathbf{\{1024, 1024\}}$의 비트 패턴 (관찰 2) |
| $\text{BOTTOM\_MASK}$ | $\mathbf{0x000f000f}$ | $Int4$ 요소 중 하위 비트들을 마스킹하여 추출 |
| $\text{TOP\_MASK}$ | $\mathbf{0x00f000f0}$ | $Int4$ 요소 중 상위 비트들을 마스킹하여 추출 |

### $LOP3$ 명령어 분석

`asm volatile("lop3.b32" %0, %1, %2, %3, %4;\n")`는 $PTX$의 논리 연산자 $LOP3$를 사용하여 다음 연산을 수행한다. 이는 $AND$, $OR$, $NOT$ 등을 조합하는 고성능 단일 명령어이다.

$$h[i] = (i4s\quad AND\quad  MASK)\quad  OR\quad  \mathbf{0x64006400}$$

1. **마스킹 ($AND$)**: 입력 $i4s$에서 해당하는 $Int4$ 요소들의 비트만 추출한다.

2. **OR 연산**: 추출된 $Int4$ 값($Y$)에 $\mathbf{0x64006400}$ (1024)를 $OR$ 연산한다. 이 연산은 논문의 **관찰 2**를 활용하여 $\mathbf{\{e_i+1024, e_{i+1}+1024\}}$의 $FP16$ 비트 패턴을 **고속으로** 생성한다.

총 4개의 $LOP3$ 명령어로 8개의 $Int4$ 요소가 $FP16$ 비트 패턴으로 저장된다. 특히 $elt_{23}$과 $elt_{67}$은 시프트 없이 처리되기 때문에, $top\_i4s$에 필요한 **단 하나의 시프트 명령**만 사용된다.

### 최종 $FP16$ 값 복구

이제 비트 패턴으로 만들어진 임시 $FP16$ 값에 오프셋 제거 및 스케일링을 수행한다.

| **상수** | **값** | **의미** |
| --- | --- | --- |
| $\text{FP16 \_ TOP\_ MAGIC\_ NUM}$ | $\mathbf{0x64006400}$ | $half2$로 표현된 $\mathbf{\{1024, 1024\}}$ |
| $\text{ONE\_SIXTEENTH}$ | $\mathbf{0x2c002c00}$ | $half2$로 표현된 $\mathbf{\{1/16, 1/16\}}$ |
| $\text{NEG\_64}$ | $\mathbf{0xd400d400}$ | $half2$로 표현된 $\mathbf{\{-64, -64\}}$ |

### $Int4$의 총 오프셋 계산

논문에 따르면, $Int4$는 원래 부호 있는($[-8, 7]$) 값이었다.


- **부호 변환 오프셋**: $8$을 더해 $unsigned$ $Int4$ ($[0, 15]$)로 만든다.
- **비트 트릭 오프셋**: $1024$를 더해 $FP16$ 비트 패턴을 만든다.

따라서 총 오프셋은 $1024 + 8 = \mathbf{1032}$이다.

### 연산 명령어 분석

코드는 4개의 $half2$ 쌍을 두 종류의 연산으로 처리한다: $elt_{01}, elt_{45}$는 $SUB$, $elt_{23}, elt_{67}$은 $FMA$ (Fused Multiply-Add)이다.

**A. $elt_{01}$ 및 $elt_{45}$ 처리 (SUB):**

```c
// Convert elt_01
asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
```

- $\text{FP16\_TOP\_MAGIC\_NUM}$은 $\mathbf{\{1024, 1024\}}$이다.
- 이 연산은 $FP16$ 값에서 $1024$를 뺀다.
- 구현에서는 논문의 $Int4$ 최적화 규칙($1032$를 빼야 함)을 직접 따르지 않고, **$1024$만 빼고 있다**. 나머지 $\mathbf{8}$에 대한 처리는 다른 부분(스케일링)에 통합되어 있다.

**B. $elt_{23}$ 및 $elt_{67}$ 처리 (FMA):**

```c
// Convert elt_23
asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
```

$FMA$ 연산은 $A \times B + C$ 형태이다. 
$$result = h[1] \times \lbrace 1/16, 1/16 \rbrace + \lbrace -64, -64 \rbrace$$
- $h[1]$ (임시 $FP16$ 값)은 $\mathbf{\{e_{2}+1024, e_{3}+1024\}}$를 나타낸다.
- $e_i$는 $Int4$ 가중치로, 최종 역양자화는 $W_{dq} = e_i \times S$ (Scale)로 표현된다.

이 $FMA$ 연산은 $Int8$ 역양자화 알고리즘과는 다른 스케일링 기반의 역양자화 공식을 따르는 것이다.: $W_{dq} = IntToFloat(W_{quantized}) \times Scale$.

$\mathbf{ \lbrace 1/16, 1/16 \rbrace}$은 스케일 값으로 사용되고, $\mathbf{\lbrace -64, -64 \rbrace}$는 오프셋을 상쇄하는 역할을 한다. 이는 $SUB$ 연산과 달리 전체 스케일링이 $FMA$로 융합되어 처리되는 복잡한 로직을 내포하고 있다.



## 결론

FP16 포맷의 구조적 특징, 특히 $2^{10}$을 기준으로 정수가 표현되는 방식과 비트 패턴이 값에 대응되는 방식을 활용하여, 대규모 AI 모델 추론 시 $Int4$ 양자화 가중치의 역양자화 과정을 효율적으로 수행할 수 있다. 



## References

1. [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532)
2. [Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production](https://arxiv.org/abs/2211.10017)
3. [IEEE 754 Standard for Floating-Point Arithmetic](https://ieeexplore.ieee.org/document/8766229)
4. [NVIDIA CUDA Programming Guide - PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
