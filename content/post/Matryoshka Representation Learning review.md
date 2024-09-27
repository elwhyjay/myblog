+++
title = 'Matryoshka Representation Learning Review'
date = 2024-09-27T00:00:29+09:00
draft = true
mathjax = true
tags = ['NLP','Machine-Learning','paper_review']
+++

blog도 개설한겸 논문한편을 리뷰해보고자 한다. 정말 오랜만에 작성하는 리뷰이다. 이 논문의 제목은
Matryoshka Representation Learning 으로 Neurips'22 에 accept된 논문이다. 

학습된 representation (예를 들어 encoder로 embedding된 sentence 등)은 다양한 다운스트림 task에 활용된다. Bert 계열의 encoder의 embedding출력값이 768차원으로 고정되듯 각 다운스트림 task에서 고정된 표현은 부족하거나 과할 수 있다. 이 때 적절한 차원으로의 축소 역시 그 크기를 정하는데 많은 리소스를 사용해야만 한다. 이 논문은 마트료시카 표현 학습 방법(MRL)을 제시함으로써 인코더가 다양한 정보로 표현을 임베딩해 다운스트림 task에 적응하는 방법을 제시한다. MRL을 통해 학습된 표현은 단일모델로 학습된 동일한 차원의 표현만큼 잘 학습한다.

방법을 살펴보자. 

우선 $d$ 차원의 임베딩이있을때, 집합 $\mathcal{M}\subset [d]$ 은 representation 의 크기집합이다. 입력 데이터에 대해 $x$ (input domain $\mathcal{X}$), 우리의 학습목표는  $d$-dimensional representation vector $z \in \mathbb{R}^d$ 를 학습하는 것이다. 모든 $m\in\mathcal{M}$ 에대해, MRL 은 $z$ 의 처음 $m$ 차원 임베딩 벡터, $z_{1:m}\in\mathbb{R}^m$ 가 독립적으로 $x$의 transferable한 능력과 일반적인 표현능력을 갖추도록 학습한다.  deep neural network $F(\, \cdot\,  ;\theta_F)\colon \mathcal{X} \rightarrow \mathbb{R}^d$ 는 $x$ 를 $z$로 맵핑하는 모델이라고 할때  $z \coloneqq F(x; \theta_F)$ 와 같이 적을 수 있다. 다양한 정밀도(granularity)가 학습되는데 그 크기는  $\lvert \mathcal{M} \rvert \leq \lfloor{\log(d)}\rfloor$ 이다. 저자들은 $\mathcal{M}$ 을 반감으로 구성하였다. e.g) $\{2048,1024,512,...\}$.  






![figure1](/img/Maytyoshka_img1.png)*Figure 1* 


attnetion masking 이나 dynamic attention 같이 입력에 대해서만 

--- 
# Reference
- Kusupati, Aditya, et al. "Matryoshka representation learning." Advances in Neural Information Processing Systems 35 (2022): 30233-30249. https://arxiv.org/pdf/2205.13147v4 
- https://github.com/RAIVNLab/MRL
- https://huggingface.co/blog/matryoshka
- Li, Xianming, et al. "2d matryoshka sentence embeddings." arXiv preprint arXiv:2402.14776 (2024).https://arxiv.org/pdf/2402.14776v1
- https://sbert.net/examples/training/matryoshka/README.html
- https://velog.io/@hoon_lander/posts
--- 