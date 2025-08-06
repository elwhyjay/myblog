+++
title = 'Llm_error'
date = 2024-11-01T19:56:05+09:00
draft = true
tags = ['LLM', 'Error']
+++

LLM 을 다루다 만나는 error를 모아보겠다

### Error 1. tokenizer template....
```bash
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating 
```
이런 에러는 template이 없을때 발생하는데. 이때는 tokenizer.chat_template을 설정해주면 된다. 주로 mistral,llama2에서 발생한다. 
```python
tokenizer.chat_template = "..."
``` 
더 자세한 chat_template사용방법은 [여기](https://huggingface.co/docs/transformers/main/en/chat_templating)를 참고하자.