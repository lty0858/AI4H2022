## Agenda
- A.單字向量化（vectorizing）
  - 標準流程與實作 == > class Vectorizer的撰寫
  - 使用tensorflow.keras.layers.TextVectorization
- B.文字處理模型
  - 範例資料集的處理
  - B1.詞袋模型(bag-of words model)
  - B2.序列模型(sequence model)
## A.單字向量化（vectorizing）
- 單字向量化（vectorizing）是一種將文字轉換成數值張量的過程
- 有許多種做法，但都遵循相同的模板 ==> 參看教科書圖11.1
﹒- 1.將文字標準化standardize:使其更易於處理。
    - 例如轉換成小寫字母，或刪除標點符號等。
  - 2.斷詞（tokenization）: 把文字分割成一個個單元（稱之為token），如字元、單字或單字組
﹒- 3.把token 轉換成一個數值向量（如： one-hot 向量）。
    - 我們通常會先為資料中的所有token 加上索引
- 11-2-1 文字標準化（text standardization)
  - 待處理的兩段句子
```
sunset came. i was staring at the Mexico sky. Isnt nature splendid ？？」
Sunset came; I started at the Mexico sky. Isn’t natur splendid? J
```
  - 文字標準化（text standardization)
    - 文字標準化是特徵工程的一種基礎形式，目的是消除掉不希望模型去處理的編碼差異。
    - 該步驟不僅出現在機器學習領域，若我們想建橋一個搜尋引擎，也要做同樣的事情。
    - 標準化做法:
      - 將文字轉換成小寫字母並刪除標點符號。
      - 將特殊字元轉換成標準字母，例如用「e」取代「已」等
      - 字根提取(stemming) ：將一個詞彙的不同變形（如一個動詞的詞形變化）轉換成單一的適用表示法。
        - was staring 和 stared  == >變成 stare 
    - 兩個句子會變成：
```
sunset came i was staring at the mexico sky isnt nature splendid 
sunset came i stared at the mexico sky isnt nature splendid 
```
- 11-2-2 拆分文字（斷詞tokenization)
  - 3 種不同的方法來進行tokenization
     - 1.單字層級的tokenization (word-level tokenization) 
       -  token 間是以空格（或標點符號）分隔的子字串。
       -  該做法的其中一種變形，是在適當情況下進一步將單字拆分成「子字（subword）」
          -  將staring 視為「star＋ ing 」
          -  將called 視為「call+ ed 」
     - 2.N-gram tokenization: token 是由N 個連續單字構成的組合
       - 「the cat 」或「he was 」都是2-gram 的token(bigram)
     - 3.字元層級的tokenization (character-level tokenization）：每個字元就是一個token 。
       - 實際案例中很少使用
       - 只有在特定情境（例如﹒文字生成或語音辨識）才會看到。

## 標準流程與實作 == > class Vectorizer的撰寫
```python

# Natural-language processing: 
# Preparing text data
# Text standardization
# Text splitting (tokenization)
# Vocabulary indexing
# Using the TextVectorization layer

import string

class Vectorizer:
    def standardize(self, text):
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)

    def tokenize(self, text):
        text = self.standardize(text)
        return text.split()

    def make_vocabulary(self, dataset):
        self.vocabulary = {"": 0, "[UNK]": 1}
        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        self.inverse_vocabulary = dict(
            (v, k) for k, v in self.vocabulary.items())

    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence):
        return " ".join(
            self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence)

vectorizer = Vectorizer()

dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]

vectorizer.make_vocabulary(dataset)

test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = vectorizer.encode(test_sentence)
print(encoded_sentence)

decoded_sentence = vectorizer.decode(encoded_sentence)
print(decoded_sentence)

```
## 使用tensorflow.keras.layers.TextVectorization
- [tf.keras.layers.TextVectorization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization)
```
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
```python
from tensorflow.keras.layers import TextVectorization
text_vectorization = TextVectorization(
    output_mode="int",
)
```

```python
import re
import string
import tensorflow as tf

# 自訂的標準化函數
def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor) //將字串轉換成小寫
    //用空字串取代標點符號
    return tf.strings.regex_replace(
        lowercase_string, f"[{re.escape(string.punctuation)}]", "")

# 自訂的tokenization函式
def custom_split_fn(string_tensor):
    return tf.strings.split(string_tensor) //使用空格來拆分字串


text_vectorization = TextVectorization(
    output_mode="int",
    standardize=custom_standardization_fn,
    split=custom_split_fn,
)


dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]

# 為一個文字語料庫中的單字建立索引
# 只需用一個能生成字串的Dataset物件來呼叫TextVectorization 層的adapt()方法：
text_vectorization.adapt(dataset)

# 顯示詞彙表來
# 可使用text_vectorization.get_vocabulary()取得運算出的詞彙表

vocabulary = text_vectorization.get_vocabulary()
test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = text_vectorization(test_sentence)
print(encoded_sentence)

# 對一個句子進行編碼(變成整數序列)，然後再解碼(將整數序列轉換回單字)
inverse_vocab = dict(enumerate(vocabulary))
decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
print(decoded_sentence)
```
# B.文字處理模型
- 核心關鍵問題:
  - 如何表示個別單字
  - 如何表示單字順序
- 文字處理模型
  - 詞袋模型(bag-of words model)
    - 拋棄順序、把文字視作無序的單字集合來處理 
  - 序列模型(sequence model):
    - 單字應該嚴格按照出現的順序來處理，一次處理一個
    - RNN循環模型
    - 新的混合的方法：Transformer 架構
      - Transformer 是不直接處理順序的，但它會將單字位置的資訊注入所處理的表示法中。
      - 如此一來，它便能看到一個句子中的不同部分(不同於RNN)，同時仍考慮到順序
    - 2015 年才開始提升

## 範例資料集: IMDB movie reviews data(IMDB 影評分類任務)
- 下載資料
```
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz

//原本aclImdb中還有一個train/unsup 子目錄，但用不到不，於是便將其刪除
!rm -r aclImdb/train/unsup
!cat aclImdb/train/pos/4077_10.txt
```
- 檢視aclImdb的目錄結構
  - train/pos中有12,500 個文字檔，每個檔案都包含一篇具正面評價的影評文字，可作為訓練資料使用。
  - train/neg／中有12,500 個文字檔，每個檔案都包含一篇具負面評論的訓練資料，同樣也有12,500 個檔案。
  - 我們一共有25000 個文字檔能用於訓練
  - 另外，用於測試的文字檔同樣也是25000 個 
```python
import os, pathlib, shutil, random

base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"

for category in ("neg", "pos"):
    os.makedirs(val_dir / category)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)
    //準備一個驗證集：取出20% 的訓練文字檔，並放到一個新的目錄（aclImdb/val）中
    num_val_samples = int(0.2 * len(files))
    val_files = files[-num_val_samples:]
    for fname in val_files:
        shutil.move(train_dir / category / fname,
                    val_dir / category / fname)
```
- 使用keras.utils.`text`_dataset_from_directory批次處理文字資料
  - 就像使用`image`_dataset_from_directory()來創建影像及其標籤的批次資料嗎
```python
from tensorflow import keras
batch_size = 32

train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size
)
val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size
)
test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

# Displaying the shapes and dtypes of the first batch

for inputs, targets in train_ds:
    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    print("inputs[0]:", inputs[0])
    print("targets[0]:", targets[0])
    break
```
# B1.詞袋模型(bag-of words model)
- 詞袋法(The bag-of-words approach): 將單字視為一組`集合(set)` ==>無順序

## 第一種測試:使用`Single words (unigrams)` with binary encoding
- unigram 組成的詞袋
  - the cat sat on the mat 這個句子就會變成：
  - (”cat”,"mat”,"on"’”sat"’”the”)
- 用一個簡單向量來呈現文字文件中的所有內容，其中的每個項目可以反映特定單字是否存在。
- 使用多元編碼（multi-hot編碼）就可把文字文件編成一個向量，其長度等同於詞彙表中的單字數量。
- 在該向量中，幾乎所有項目都是用「O 」來代表文件中未出現的單字，少數的「l 」則代表那些有出現的單字。
- Preprocessing datasets with a `TextVectorization` layer
  - max_tokens=20000:
    - 限制使用20000個最常出現的單字。
    - 若未指定則預設會為訓練資料中的每固單字建立索引，但這樣可能會多出幾萬個只出現兩次的詞彙，而這些詞彙通常沒有乘載什麼有用的資訊。
    - 20000是做文字分類時的合理詞彙量
  - 將輸出token 編碼成multi_hot向量  ==> output_mode="multi_hot"
```python
text_vectorization = TextVectorization(
    max_tokens=20000,
    output_mode="multi_hot",
)

# 準備個只產生原始文字輸入的資料集(沒有標籤)
text_only_train_ds = train_ds.map(lambda x, y: x)

# 建立索引:使用text_only_train_ds來建立索引
text_vectorization.adapt(text_only_train_ds)


# num_parallel_calls=4: 明確指定num_parallel_calls值來使用多個CPU 核心
binary_1gram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
binary_1gram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
binary_1gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
        
# 檢視輸出結果(the output of binary unigram dataset)

for inputs, targets in binary_1gram_train_ds:
    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    print("inputs[0]:", inputs[0])
    print("targets[0]:", targets[0])
    break

# 建構可重複使用的建模函式get_model()

from tensorflow import keras
from tensorflow.keras import layers

def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# Training and testing the binary unigram model

model = get_model()
        
model.summary()
        
callbacks = [
    keras.callbacks.ModelCheckpoint("binary_1gram.keras",
                                    save_best_only=True)
]

model.fit(binary_1gram_train_ds.cache(),
          validation_data=binary_1gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)
        
model = keras.models.load_model("binary_1gram.keras")

print(f"Test acc: {model.evaluate(binary_1gram_test_ds)[1]:.3f}")
```
- binary_1gram_train_ds.cache():
- 呼叫cache()以在記憶體中對資料集進行快取
- 這作法只會在第1 個epcch 中進行預先處理，並在接下來的epoch 中重用預先處理完的文字
- 這個做法只有在資料量小到能裝進記憶體時才適用

## 第二種測試 ==> 使用`Bigrams` with `binary` encoding
- Bigrams組成的詞袋
  - the cat sat on the mat 這個句子就會變成：
  - {"the”,”the cat”, ”cat”,"cat sat”,"sat","sat on ”,”on”,”on the","the mat","mat "}
- TextVectorization層可以傳回任意N 值的N-gram，只要將增rams 參數設定為所需的N 即可
```python

# Configuring the TextVectorization layer to return bigrams

text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="multi_hot",
)

# Training and testing the binary bigram model

text_vectorization.adapt(text_only_train_ds)
binary_2gram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
binary_2gram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
binary_2gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

model = get_model()
        
model.summary()
        
callbacks = [
    keras.callbacks.ModelCheckpoint("binary_2gram.keras",
                                    save_best_only=True)
]
        
model.fit(binary_2gram_train_ds.cache(),
          validation_data=binary_2gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)
        
model = keras.models.load_model("binary_2gram.keras")
        
print(f"Test acc: {model.evaluate(binary_2gram_test_ds)[1]:.3f}")
```
## 第三種測試 ==> Bigrams with `TF-IDF` encoding
- 可藉由計算每個單字或N-gram 的出現次數，為表示法再增添一些資訊。
- 使用文字資料中，單字或N-gram 出現次數的直方圓(histogram):
- { "the": 2,”the cat": l ,”cat": 1, "cat sat”: 1, "sat": 1,"sat on”: 1 ,”on": 1, "on the":1,＂the mat": l, "mat”: 1}
- 進行文字分類時，知道特定單字的出現次數非常關鍵。
- 任何有一定長度的影評都可能包含terrible 一詞，但如果影評中出現了很多次terrible，就很可能是負面影評。
- 注意底下所使用的output_mode參數
  - output_mode="count" ==> 單純計算出線次數
  - output_mode="tf_idf"==> TF-IDF 正規化(TF-IDF normalization)
    - 已經內建在TextVectorization層中:只要將output_mode 參數設定為tf_idf,就可使用 
```python
# Configuring the TextVectorization layer to return token counts

text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="count"
)
```
```python
# Configuring TextVectorization to return TF-IDF-weighted outputs

text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="tf_idf",
)
```      
#### Training and testing the TF-IDF bigram model
```
text_vectorization.adapt(text_only_train_ds)

tfidf_2gram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
tfidf_2gram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
tfidf_2gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

model = get_model()
        
model.summary()
        
callbacks = [
    keras.callbacks.ModelCheckpoint("tfidf_2gram.keras",
                                    save_best_only=True)
]
        
model.fit(tfidf_2gram_train_ds.cache(),
          validation_data=tfidf_2gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)
        
model = keras.models.load_model("tfidf_2gram.keras")
        
print(f"Test acc: {model.evaluate(tfidf_2gram_test_ds)[1]:.3f}")
```
### 匯出一個處理原始字串的模型
- 在前述案例中，將文字標準化、單字拆分和建立索引都當作tf.data 工作流的一部分。
- 但不同模型使用其文字處理層則...
- 也就是:若我們想匯出一個不需要該工作流的獨立運作模型，就要確保它有自己的文字處理層
- （否則就要在賞際運作的環境中重新處理一次，這很有挑戰性，也可能導致訓練資料跟實際運作的資料之間的微妙差異）。
- 這個問題不難解決。只要創建包含TextVectorization 層的一個新模型，並加入剛剛訓練的模型即可
```
inputs = keras.Input(shape=(1,), dtype="string")
processed_inputs = text_vectorization(inputs)
outputs = model(processed_inputs)

inference_model = keras.Model(inputs, outputs)
        
import tensorflow as tf

raw_text_data = tf.convert_to_tensor([
    ["That was an excellent movie, I loved it."],
])
        
predictions = inference_model(raw_text_data)
        
print(f"{float(predictions[0] * 100):.2f} percent positive")
```

# B2.序列模型(sequence model)
- Processing words as a sequence: The sequence model approach
- 單字順序的重要性：對具順序的特徵進行人工處理（如： N-gram）能讓準確度大大提升
- 深度學習的歷史是與人工的特徵工程背道而馳的。
- 深度學習的目標，就是要讓模型僅從看過的資料中自己學習特徵。
- 如果不人工做出包含順序資訊的特徵，要如何讓模型處理原始單字序列，並自己找出這樣的特徵呢？
- 序列模型sequence model
- 實作一個序列模型
  - 1.先用整數索引序列來表示輸入樣本（一個整數代表一個單字）
  - 2.將每個整數對應到一個向量以取得向量序列。
  - 3.把這些向量序列送入能讓「相鄰向量中的特徵產生關聯性」的堆疊層中，例如lD 卷積神經網路、RNN 或Transformer 等。
- 在2016 至2017 年左右的一段時間裡，雙向RNN （尤其是雙向LSTM）被認為是序列模型的最頂尖成果
- 現今的序列模型幾乎都是用Transformer 來建構
- 奇怪的是,lD 卷積網路在NLP中從來都不是很流行

## 第一個測試:雙向RNN(雙向LSTM)
```python

# Downloading the data

!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
!rm -r aclImdb/train/unsup
```
# Preparing the data
```python
import os, pathlib, shutil, random
from tensorflow import keras

batch_size = 32

base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"

for category in ("neg", "pos"):
    os.makedirs(val_dir / category)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)
    num_val_samples = int(0.2 * len(files))
    val_files = files[-num_val_samples:]
    for fname in val_files:
        shutil.move(train_dir / category / fname,
                    val_dir / category / fname)

train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size
)

val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size
)
test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

text_only_train_ds = train_ds.map(lambda x, y: x)
```
# Preparing integer sequence datasets ==>整數序列
- max_length = 600:為了控制輸入大小，我們只會採用評論中的前600個單字。
- 這是一個很合理的做法，因為影評的平均長度為233 字,只有5% 的評論會超過600字
```
from tensorflow.keras import layers

max_length = 600
max_tokens = 20000

text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)

text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
int_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
int_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
```
#  使用 one-hot encoded vector sequences建立的sequence model
- 建構模型:要把整數序列轉換成向量序列
  - 使用最簡單的方式:對整數進行one-hot 編碼
  - embedded = tf.one_hot(inputs, depth=max_tokens)
  - 再把這些one-hot 向量傳進一個簡單的雙向LSTM
- inputs = keras.Input(shape=(None,), dtype="int64")
  - 一筆輸入就是一個整數序列
  - shape=(None,) 表示序列長度不固定，不過本例實際輸出的序列長度均為600
- embedded = tf.one_hot(inputs, depth=max_tokens) 
  - depth = max_tokens = 20000: 將每個整數值都編碼為20,000維的one-hot 向量
```
import tensorflow as tf

inputs = keras.Input(shape=(None,), dtype="int64")

embedded = tf.one_hot(inputs, depth=max_tokens)

x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
              
model.summary()

# Training 

callbacks = [
    keras.callbacks.ModelCheckpoint("one_hot_bidir_lstm.keras",
                                    save_best_only=True)
]

model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)

model = keras.models.load_model("one_hot_bidir_lstm.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
```
教科書作者的評論:
- 1.這個模型的訓練速度非常慢
  - 這是因為我們的輸入相當大﹒每個輸入樣本都被編碼成一個大小為（600, 20000）的矩陣
  - 因此一篇影評就有12,000,000 個浮點數，可見雙向LSTM 的工作量非常大
- 2.模型的測試準確度只達到87%，遠不及前面模型。

# 使用 word embeddings建立的sequence model
- word embeddings的好處與歷史
- 獲得word embeddings的兩種方法:
  - 1.針對欲解決的任務（如文件分類或情感預測）進行詞word embeddings的學習
    - 會從隨機的詞向量開始,然後像學習神經網路權重一樣,學習單字向量的表示法
  - 2.預訓練詞依入法(pretrained word embedding)
    - 將預先學習好（使用與當前問題不同的機器學習任務）的詞嵌入向量載入模型中使用
- Learning word embeddings with the Embedding layer
```python

# 初始化Embedding layer(Instantiating an Embedding layer)

embedding_layer = layers.Embedding(input_dim=max_tokens, output_dim=256)

# 重新訓練Embedding layer(Model that uses an Embedding layer trained from scratch)

inputs = keras.Input(shape=(None,), dtype="int64")

embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)

x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
              
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("embeddings_bidir_gru.keras",
                                    save_best_only=True)
]

model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)

model = keras.models.load_model("embeddings_bidir_gru.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
```

# 使用masking enabled的 word embeddings建立的sequence model
- padding and masking ==> Embedding layer with masking enabled
```python

inputs = keras.Input(shape=(None,), dtype="int64")

embedded = layers.Embedding(
    input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)
    
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
              
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("embeddings_bidir_gru_with_masking.keras",
                                    save_best_only=True)
]

model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)

model = keras.models.load_model("embeddings_bidir_gru_with_masking.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
```

## 使用pretrained word embeddings建立的sequence model
- 使用pretrained word embeddings:GloVe word-embeddings
  - 1.下載pretrained word embeddings:GloVe word-embeddings
  - 2.建構 GloVe word-embeddings matrix
  - 3.使用layers.Embedding處理GloVe word-embeddings matrixembedding_matrix並建置成embedding_layer
  - 4.使用 pretrained Embedding layer建構模型

- 1.下載pretrained word embeddings:GloVe word-embeddings
```
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip
```
```
import numpy as np
path_to_glove_file = "glove.6B.100d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print(f"Found {len(embeddings_index)} word vectors.")
```
- 2.建構 GloVe word-embeddings matrix
```
embedding_dim = 100

vocabulary = text_vectorization.get_vocabulary()

word_index = dict(zip(vocabulary, range(len(vocabulary))))

embedding_matrix = np.zeros((max_tokens, embedding_dim))

for word, i in word_index.items():
    if i < max_tokens:
        embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
```        
- 3.使用layers.Embedding()處理GloVe word-embeddings matrix並建置成embedding_layer
```        
embedding_layer = layers.Embedding(
    max_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
    mask_zero=True,
)
```
- 4.使用 pretrained Embedding layer建構模型
```
inputs = keras.Input(shape=(None,), dtype="int64")

embedded = embedding_layer(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
              
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("glove_embeddings_sequence_model.keras",
                                    save_best_only=True)
]

model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)

model = keras.models.load_model("glove_embeddings_sequence_model.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
```
