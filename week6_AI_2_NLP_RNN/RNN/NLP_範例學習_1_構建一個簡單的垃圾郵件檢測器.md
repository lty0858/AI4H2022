# 參考資料
- [Advanced Natural Language Processing with TensorFlow 2(2021)Ashish Bansal](https://www.packtpub.com/product/advanced-natural-language-processing-with-tensorflow-2/9781800200937)
  - TensorFlow2的高級自然語言處理 ch1
- [Mastering Transformers(2021) Savaş Yıldırım , Meysam Asgari-Chenaghlu](https://www.packtpub.com/product/mastering-transformers/9781801077651)

# 自然語言處理(NLP)範例學習:構建一個簡單的垃圾郵件檢測器
  - 典型的`文本處理的工作流程(text processing workflow)`
  - `數據收集(Data collection)`和`標記(labeling)`
  - 文本預處理pre-processing text:文本規範化(Text normalization)，包括大小寫規範化(case normalization)、文本標記化(text tokenization)、詞幹提取(stemming)和詞形還原(lemmatization)
    - 對已文本規範化的數據集進行建模
    - 向量化文本(Vectorizing text)
    - 使用向量化文本對數據集進行建模
- 第一步:`數據收集(Data collection)`和`標記(labeling)`
  - `數據收集(Data collection)`:在文本領域中，有大量數據可查。
    - 一種常見的方法是使用函式庫(如scrapy或 Beautiful Soup)從網路上抓取數據。
    - 數據通常是未標記的，因此不能直接用於監督模型。
    - 不過這個數據還是很有用的。通過使用遷移學習，可以使用無監督或半監督方法訓練語言模型，並且可以進一步與特定於手頭任務的小型訓練數據集一起使用。
    - 後續:使用 BERT 嵌入的遷移學習時，使用 BiLSTM、CRF 和 Viterbi 解碼的`命名實體識別Named Entity Recognition(NER)`。 
  - `標記(labeling)`:文本數據數據收集步驟中的來源被標記為正確的類
    - 任務1:建立一個電子郵件的垃圾郵件分類器
      - 收集大量電子郵件。
      - 標記步驟將附加垃圾郵件或非垃圾郵件標籤到每封電子郵件。
    - 任務2:推文上的情緒檢測。
      - 數據收集步驟將涉及收集許多推文。
      - 使用充當基本事實的標籤標記每條推文。
    - 任務3:收集新聞文章
      - 標籤是文章的摘要。
    - 任務4:電子郵件自動回覆功能
      - 需要收集許多帶有回覆的電子郵件。
      - 標籤將是近似回覆的短文本。
    - 如果正在處理沒有太多公共數據的特定域，您可能必須自己執行這些步驟。 
  - 收集標記數據 ==> 公開可用的數據集。
    - 電子郵件數據集上構建垃圾郵件檢測系統:使用加州大學歐文分校提供的 SMS Spam Collection 數據集。
      - 每條 SMS 都被標記為“SPAM垃圾郵件”或“HAM正常郵件”
      - 加州大學歐文分校是機器學習數據集的重要來源。
      - http://archive.ics.uci.edu/ml/datasets.php
      - NLP公開可用的數據集: https://github.com/niderhoff/nlp-datasets
- 使用 Google Colaboratory GPU


# 範例程式


```python
# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
#from tf.keras.models import Sequential
#from tf.keras.layers import Dense
import os
import io

tf.__version__

path_to_zip = tf.keras.utils.get_file("smsspamcollection.zip",
                  origin="https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
                  extract=True)

# Unzip the file into a folder
!unzip $path_to_zip -d data

ls ./data
```
```
lines = io.open('/content/data/SMSSpamCollection').read().strip().split('\n')
lines[0]

spam_dataset = []
count = 0
for line in lines:
  label, text = line.split('\t')
  if label.lower().strip() == 'spam':
    spam_dataset.append((1, text.strip()))
    count += 1
  else:
    spam_dataset.append(((0, text.strip())))

print(spam_dataset[0])
print("Spam: ", count)
```
### Normalization functions
```
import pandas as pd 
df = pd.DataFrame(spam_dataset, columns=['Spam', 'Message'])


import re

def message_length(x):
  # returns total number of characters
  return len(x)

def num_capitals(x):
  _, count = re.subn(r'[A-Z]', '', x) # only works in english
  return count

def num_punctuation(x):
  _, count = re.subn(r'\W', '', x)
  return count
df['Capitals'] = df['Message'].apply(num_capitals)
df['Punctuation'] = df['Message'].apply(num_punctuation)
df['Length'] = df['Message'].apply(message_length)
df.describe()

train=df.sample(frac=0.8,random_state=42) #random state is a seed value
test=df.drop(train.index)
train.describe()

test.describe()
```
# Model Building
```
# Basic 1-layer neural network model for evaluation
def make_model(input_dims=3, num_units=12):
  model = tf.keras.Sequential()

  # Adds a densely-connected layer with 12 units to the model:
  model.add(tf.keras.layers.Dense(num_units, 
                                  input_dim=input_dims, 
                                  activation='relu'))

  # Add a sigmoid layer with a binary output unit:
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', 
                metrics=['accuracy'])
  return model

x_train = train[['Length', 'Punctuation', 'Capitals']]
y_train = train[['Spam']]

x_test = test[['Length', 'Punctuation', 'Capitals']]
y_test = test[['Spam']]
x_train

model = make_model()
model.fit(x_train, y_train, epochs=10, batch_size=10)

model.evaluate(x_test, y_test)
```
```
#y_train_pred = model.predict_classes(x_train)
# y_train_pred = model.predict(x_train)
y_train_pred = (model.predict(x_train)> 0.5).astype("int32")

tf.math.confusion_matrix(tf.constant(y_train.Spam), 
                         y_train_pred)

sum(y_train_pred)

# y_test_pred = model.predict_classes(x_test)
# 
y_test_pred =(model.predict(x_test)> 0.5).astype("int32")

tf.math.confusion_matrix(tf.constant(y_test.Spam), y_test_pred)
```
## Tokenization and Stop Word Removal
```
sentence = 'Go until jurong point, crazy.. Available only in bugis n great world'
sentence.split()
```
```
!pip install stanza  # StanfordNLP has become https://github.com/stanfordnlp/stanza/
```
```
import stanza
en = stanza.download('en')

en = stanza.Pipeline(lang='en')

sentence

tokenized = en(sentence)
len(tokenized.sentences)

for snt in tokenized.sentences:
  for word in snt.tokens:
    print(word.text)
  print("<End of Sentence>")
```
# Dependency Parsing Example
```
en2 = stanza.Pipeline(lang='en')
pr2 = en2("Hari went to school")
for snt in pr2.sentences:
  for word in snt.tokens:
    print(word)
  print("<End of Sentence>")
```
# Japanese Tokenization Example
```
jp = stanza.download('ja')

jp = stanza.Pipeline(lang='ja')

jp_line = jp("選挙管理委員会")
for snt in jp_line.sentences:
  for word in snt.tokens:
    print(word.text)
```
# Adding Word Count Feature
```
def word_counts(x, pipeline=en):
  doc = pipeline(x)
  count = sum( [ len(sentence.tokens) for sentence in doc.sentences] )
  return count
#en = snlp.Pipeline(lang='en', processors='tokenize')
df['Words'] = df['Message'].apply(word_counts)
df.describe()
```
```
#train=df.sample(frac=0.8,random_state=42) #random state is a seed value
#test=df.drop(train.index)

train['Words'] = train['Message'].apply(word_counts)
test['Words'] = test['Message'].apply(word_counts)
x_train = train[['Length', 'Punctuation', 'Capitals', 'Words']]
y_train = train[['Spam']]

x_test = test[['Length', 'Punctuation', 'Capitals' , 'Words']]
y_test = test[['Spam']]

model = make_model(input_dims=4)

model.fit(x_train, y_train, epochs=10, batch_size=10)

model.evaluate(x_test, y_test)
```
### Stop Word Removal
```
!pip install stopwordsiso
```

```
import stopwordsiso as stopwords

stopwords.langs()
```

```
sorted(stopwords.stopwords('en'))
```
```
en_sw = stopwords.stopwords('en')

def word_counts(x, pipeline=en):
  doc = pipeline(x)
  count = 0
  for sentence in doc.sentences:
    for token in sentence.tokens:
        if token.text.lower() not in en_sw:
          count += 1
  return count

train['Words'] = train['Message'].apply(word_counts)
test['Words'] = test['Message'].apply(word_counts)
x_train = train[['Length', 'Punctuation', 'Capitals', 'Words']]
y_train = train[['Spam']]

x_test = test[['Length', 'Punctuation', 'Capitals' , 'Words']]
y_test = test[['Spam']]

model = make_model(input_dims=4)
#model = make_model(input_dims=3)

model.fit(x_train, y_train, epochs=10, batch_size=10)
```
### POS Based Features
```
en = stanza.Pipeline(lang='en')

txt = "Yo you around? A friend of mine's lookin."
pos = en(txt)
```

```
def print_pos(doc):
    text = ""
    for sentence in doc.sentences:
        for token in sentence.tokens:
            text += token.words[0].text + "/" + \
                    token.words[0].upos + " "
        text += "\n"
    return text

print(print_pos(pos))
```
```
en_sw = stopwords.stopwords('en')

def word_counts_v3(x, pipeline=en):
  doc = pipeline(x)
  count = 0
  for sentence in doc.sentences:
    for token in sentence.tokens:
        if token.text.lower() not in en_sw and \
        token.words[0].upos not in ['PUNCT', 'SYM']:
          count += 1
  return count

print(word_counts(txt), word_counts_v3(txt))
```
```
train['Test'] = 0
train.describe()
```

```
def word_counts_v3(x, pipeline=en):
  doc = pipeline(x)
  totals = 0.
  count = 0.
  non_word = 0.
  for sentence in doc.sentences:
    totals += len(sentence.tokens)  # (1)
    for token in sentence.tokens:
        if token.text.lower() not in en_sw:
          if token.words[0].upos not in ['PUNCT', 'SYM']:
            count += 1.
          else:
            non_word += 1.
  non_word = non_word / totals
  return pd.Series([count, non_word], index=['Words_NoPunct', 'Punct'])

x = train[:10]
x.describe()
```

```
train_tmp = train['Message'].apply(word_counts_v3)
train = pd.concat([train, train_tmp], axis=1)
train.describe()
```

```
test_tmp = test['Message'].apply(word_counts_v3)
test = pd.concat([test, test_tmp], axis=1)
test.describe()
```

```
z = pd.concat([x, train_tmp], axis=1)
z.describe()
```

```
z.loc[z['Spam']==0].describe()
```

```
z.loc[z['Spam']==1].describe()
```

```
aa = [word_counts_v3(y) for y in x['Message']]

ab = pd.DataFrame(aa)
ab.describe()
```


### Lemmatization
```python

text = "Stemming is aimed at reducing vocabulary and aid un-derstanding of" +\
       " morphological processes. This helps people un-derstand the" +\
       " morphology of words and reduce size of corpus."

lemma = en(text)
lemmas = ""
for sentence in lemma.sentences:
        for token in sentence.tokens:
            lemmas += token.words[0].lemma +"/" + \
                    token.words[0].upos + " "
        lemmas += "\n"

print(lemmas)
```


# 使用 sklearn

### Count Vectorization
```python
corpus = [
          "I like fruits. Fruits like bananas",
          "I love bananas but eat an apple",
          "An apple a day keeps the doctor away"
]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

vectorizer.get_feature_names()
```

```python
X.toarray()


from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(X.toarray())


query = vectorizer.transform(["apple and bananas"])

cosine_similarity(X, query)
```


### TF-IDF Vectorization
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X.toarray())

pd.DataFrame(tfidf.toarray(), 
             columns=vectorizer.get_feature_names())
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

tfidf = TfidfVectorizer(binary=True)
X = tfidf.fit_transform(train['Message']).astype('float32')
X_test = tfidf.transform(test['Message']).astype('float32')
X.shape
```

```python
from keras.utils import np_utils

_, cols = X.shape
model2 = make_model(cols)  # to match tf-idf dimensions
lb = LabelEncoder()
y = lb.fit_transform(y_train)
dummy_y_train = np_utils.to_categorical(y)
model2.fit(X.toarray(), y_train, epochs=10, batch_size=10)
```

```python
model2.evaluate(X_test.toarray(), y_test)
```

```python
train.loc[train.Spam == 1].describe() 
```


## 使用 Word Vectors
```python

# memory limit may be exceeded. Try deleting some objects before running this next section
# or copy this section to a different notebook.
!pip install gensim
```

```python
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

## 檢視有多少api
api.info()

## 載入要使用的
model_w2v = api.load("word2vec-google-news-300")

## 
model_w2v.most_similar("cookies",topn=10)

##
model_w2v.doesnt_match(["USA","Canada","India","Tokyo"])


##
king = model_w2v['king']
man = model_w2v['man']
woman = model_w2v['woman']

queen = king - man + woman  
model_w2v.similar_by_vector(queen)
```


```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
