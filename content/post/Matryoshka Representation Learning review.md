+++
title = 'Matryoshka Representation Learning Review'
date = 2024-09-27T00:00:29+09:00
draft = true
mathjax = true
tags = ['NLP','Machine-Learning','paper_review']
+++

blog도 개설한겸 논문한편을 리뷰해보고자 한다. 정말 오랜만에 작성하는 리뷰이다. 이 논문의 제목은
Matryoshka Representation Learning 으로 Neurips'22 에 accept된 논문이다. 

학습된 representation (예를 들어 encoder로 embedding된 sentence 등)은 다양한 다운스트림 task에 활용된다. Bert 계열의 encoder의 embedding출력값이 768차원으로 고정되듯 각 다운스트림 task에서 고정된 표현은 부족하거나 과할 수 있다. 이 때 적절한 차원으로의 축소 역시 그 크기를 정하는데 많은 리소스를 사용해야만 한다. 이 논문은 마트료시카 표현 학습 방법(MRL)을 제시함으로써 인코더가 다양한 정보로 표현을 임베딩해 다운스트림 task에 적응하는 방법을 제시한다. MRL은 이를 동해 coarse한 표현부터 fine한 표현까지 학습할 수 있다.





![figure1](/img/Maytyoshka_img1.png)*Figure 1*

--- 
# Reference
- Kusupati, Aditya, et al. "Matryoshka representation learning." Advances in Neural Information Processing Systems 35 (2022): 30233-30249. https://arxiv.org/pdf/2205.13147v4 
- https://github.com/RAIVNLab/MRL
- https://huggingface.co/blog/matryoshka
- Li, Xianming, et al. "2d matryoshka sentence embeddings." arXiv preprint arXiv:2402.14776 (2024).
- https://sbert.net/examples/training/matryoshka/README.html
- https://velog.io/@hoon_lander/posts
--- 