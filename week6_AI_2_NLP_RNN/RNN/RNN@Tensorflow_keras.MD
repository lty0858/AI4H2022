# RNN@Tensorflow/keras

- [tf.keras.layers.Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding)
- [tf.keras.layers.TextVectorization ](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization)
- 模型建構
  - 最常見的三大類型
    - [tf.keras.layers.SimpleRNN](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN)
    - [tf.keras.layers.LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
    - [tf.keras.layers.GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
  - 雙向 [tf.keras.layers.Bidirectional](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional)

# 資料預處理Data preprocessing
- [Working with preprocessing layers ](https://www.tensorflow.org/guide/keras/preprocessing_layers)
- [NLP-台語羅馬字: Word Embeddings Using Keras](https://ithelp.ithome.com.tw/articles/10254164)


# tf.keras.layers.Embedding
- [tf.keras.layers.Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding)
- [A Detailed Explanation of Keras Embedding Layer](https://www.kaggle.com/code/rajmehra03/a-detailed-explanation-of-keras-embedding-layer)
```python
tf.keras.layers.Embedding(
    input_dim,
    output_dim,
    embeddings_initializer='uniform',
    embeddings_regularizer=None,
    activity_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    input_length=None,
    **kwargs
)
```

# tf.keras.layers.TextVectorization
- [tf.keras.layers.TextVectorization ](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization)
- [範例 :Word embeddings](https://www.tensorflow.org/text/guide/word_embeddings)
```python
tf.keras.layers.TextVectorization(
    max_tokens=None,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    ngrams=None,
    output_mode='int',
    output_sequence_length=None,
    pad_to_max_tokens=False,
    vocabulary=None,
    idf_weights=None,
    sparse=False,
    ragged=False,
    **kwargs
)

```

# TensorFlow Text 
- TensorFlow Text provides a collection of text related classes and ops ready to use with TensorFlow 2.0. 
- The library can perform the preprocessing regularly required by text-based models, and includes other features useful for sequence modeling not provided by core TensorFlow.
- The benefit of using these ops in your text preprocessing is that they are done in the TensorFlow graph. 
- You do not need to worry about tokenization in training being different than the tokenization at inference, or managing preprocessing scripts.
- The tensorflow_text package includes TensorFlow implementations of many common tokenizers. 
- This includes three subword-style tokenizers:
  - text.BertTokenizer - The BertTokenizer class is a higher level interface. 
    - It includes BERT's token splitting algorithm and a WordPieceTokenizer. It takes sentences as input and returns token-IDs.
  - text.WordpieceTokenizer - The WordPieceTokenizer class is a lower level interface. 
    - It only implements the WordPiece algorithm. You must standardize and split the text into words before calling it. 
    - It takes words as input and returns token-IDs.
  - text.SentencepieceTokenizer - The SentencepieceTokenizer requires a more complex setup. 
    - Its initializer requires a pre-trained sentencepiece model. 
    - See the google/sentencepiece repository for instructions on how to build one of these models. It can accept sentences as input when tokenizing.
- [Tokenizing with TF Text](https://www.tensorflow.org/text/guide/tokenizers)
- [Subword tokenizers](https://www.tensorflow.org/text/guide/subwords_tokenizer)
- [BERT Preprocessing with TF Text](https://www.tensorflow.org/text/guide/bert_preprocessing_guide)
  - This tutorial will show how to use TF.Text preprocessing ops to transform text data into inputs for the BERT model and inputs for language masking pretraining task described in "Masked LM and Masking Procedure" of BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 
  - The process involves tokenizing text into subword units, combining sentences, trimming content to a fixed size and extracting labels for the masked language modeling task.

# [TensorFlow Models NLP library](https://github.com/tensorflow/models/tree/master/official/nlp/modeling)
- [ customize_encoder.ipynb](https://colab.research.google.com/github/tensorflow/models/blob/master/docs/nlp/index.ipynb)

# word embedding
- GloVe: Global Vectors for Word Representation
  - [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

# 各式各樣的embedding
- [Doc2vec的算法原理、代码实现及应用启发](https://zhuanlan.zhihu.com/p/336921474)
- [【Graph Embedding】Struc2Vec：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56733145)
- Word2Vec是非常經典的工作或應用，包括我們安全領域也有相關擴展，比如二進位、審計日誌、惡意程式碼分析的Asm2Vec、Log2Vec、Token2Vec
