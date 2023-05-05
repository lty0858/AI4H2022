# NLP(Natural language processing) 自然語言處理 
- NLP(Natural language processing) 自然語言處理
- NLTK 工具集 |LTP 工具集
- 文字向量化

## RNN
- RNN Model
  - RNN
  - LSTM
  - bidirectional LSTM
  - GRU
- Text classification(文章分類)
  - 資料集
    IMDB large movie review dataset
  - 各種分析技術
    - 【TensorFlow 官方教學課程】[Text classification with an RNN](https://www.tensorflow.org/text/tutorials/text_classification_rnn)
    - 【TensorFlow 官方教學課程】 [Classify text with BERT](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)
- Time Series(時間序列分析)
  - [Modern Time Series Forecasting with Python(2022)](https://www.packtpub.com/product/modern-time-series-forecasting-with-python/9781803246802) [GITHUB](https://github.com/PacktPublishing/Modern-Time-Series-Forecasting-with-Python)

## Language Model
- 靜態詞向量預訓練模型
  - 神經網路語言模型
  - Word2vec 詞向量
    - 經典論文[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention|Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio](https://arxiv.org/abs/1502.03044)
    - 【TensorFlow 官方教學課程】[Word embeddings](https://www.tensorflow.org/text/guide/word_embeddings)
  - GloVe 詞向量word2vec

- sequence-to-sequence (seq2seq) model
  - [教電腦寫作：AI球評——Seq2seq模型應用筆記(PyTorch + Python3)](https://gau820827.medium.com/%E6%95%99%E9%9B%BB%E8%85%A6%E5%AF%AB%E4%BD%9C-ai%E7%90%83%E8%A9%95-seq2seq%E6%A8%A1%E5%9E%8B%E6%87%89%E7%94%A8%E7%AD%86%E8%A8%98-pytorch-python3-31e853573dd0) 

## Pretrained Model == > Large Language Models (LLMs)

- Transformer 2017
  - 經典論文[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - BOOKS [Transformers for Natural Language Processing - Second Edition(2022)](https://www.packtpub.com/product/transformers-for-natural-language-processing-second-edition/9781803247335) [GITHUB](https://github.com/Denis2054/Transformers-for-NLP-2nd-Edition)
  - BOOKS [Mastering Transformers(2021)](https://www.packtpub.com/product/mastering-transformers/9781801077651) [GITHUB](https://github.com/PacktPublishing/Mastering-Transformers)
  - huggingface[Transformers|State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX](https://huggingface.co/docs/transformers/index)
- BERT(Bidirectional Encoder Representations from Transformers) 2018
  - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova](https://arxiv.org/abs/1810.04805)
  - 參考書籍
    - [Getting Started with Google BERT(2021)](https://www.packtpub.com/product/getting-started-with-google-bert/9781838821593)  [GITHUB](https://github.com/PacktPublishing/Getting-Started-with-Google-BERT)
    - [基於 BERT 模型的自然語言處理實戰|李金洪](https://www.tenlong.com.tw/products/9787121414084?list_name=sp)
  - BERT Variants: ALBERT, RoBERTa, ELECTRA, and SpanBERT ........

![LLMhistory.JPG](./LLMhistory.JPG)

- REVIEW
  - [Large-scale Multi-Modal Pre-trained Models: A Comprehensive Survey(2023)](https://arxiv.org/abs/2302.10035) [GITHUB](https://github.com/wangxiao5791509/MultiModal_BigModels_Survey)
  - [On the Opportunities and Risks of Foundation Models(2021)](https://arxiv.org/abs/2108.07258)
  - [A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT(2023)](https://arxiv.org/abs/2302.09419)
  - [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond(202304)](https://arxiv.org/abs/2304.13712) 
    - [有用的GITHUB網址](https://github.com/Mooler0410/LLMsPracticalGuide)  [有用的PPT](https://github.com/Mooler0410/LLMsPracticalGuide/blob/main/source/figure_gif.pptx)
- BOOKS [全中文自然語言處理：Pre-Trained Model 方法最新實戰|車萬翔、郭江、崔一鳴 著(2022)](https://www.tenlong.com.tw/products/9789860776942?list_name=srh)  
