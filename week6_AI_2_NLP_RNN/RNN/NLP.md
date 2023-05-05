

#

NLP論文讀取順序
one-hot編碼時代
簡介

# one-hot編碼
- 在提出詞向量（Distributed representation， Word embedding， word representation）之前所有的神經網路模型（或者傳統的機器學習）對詞數據的處理都是將詞轉換為one-hot編碼進行處理。NLP 中最直觀，也是到目前為止最常用的詞表示方法是 One-hot Representation
- 這種方法把每個詞表示為一個很長的向量。
- 這個向量的維度是詞表大小，其中絕大多數元素為 0，只有一個維度的值為 1，這個維度就代表了當前的詞。
　
舉個栗子：
　　“話筒”表示為 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 …]
　　“麥克”表示為 [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 …]
　　每個詞都是茫茫 0 海中的一個 1。

- 這種 One-hot Representation 如果採用稀疏方式存儲，會是非常的簡潔：也就是給每個詞分配一個數字 ID。
- 比如剛才的例子中，話筒記為 3，麥克記為 8（假設從 0 開始記）。
- 如果要程式設計實現的話，用 Hash 表給每個詞分配一個編號就可以了。
- 這麼簡潔的表示方法配合上最大熵、SVM、CRF 等等演算法已經很好地完成了 NLP 領域的各種主流任務。

這種編碼方式存在的問題

這種表示方法也存在一個重要的問題就是**“詞彙鴻溝”**現象：任意兩個詞之間都是孤立的（也就是無法衡量兩個詞之間的相似性關係，無法用定量的方法計算，因為如果把每個詞看成向量的話那麼任意兩個詞之間是，正交的也就是毫無關係的）。光從這兩個向量中看不出兩個詞是否有關係，哪怕是話筒和麥克這樣的同義詞也不能倖免於難。


# Language Model
- 經典論文
- 2003JMLR–A Neural Probabilistic Language Model–作者：Yoshua Bengio

解決的問題:本文聚焦於三個主要的問題：
- 1.維度災難(特別是離散變數)，在高維下，資料的稀缺性導致統計語言模型存在很多為0的條件概率
- 2.語言模型的參數個數隨著階數呈指數增長，所以一般這個模型的階數不會很高，這樣n-gram無法建立長遠的關係
- 3.n-gram無法建模出多個相似詞的關係

解決問題的方法

本文為解決上述的三個問題提出了幾個新概念：

- 1.詞向量：本文中的詞向量是通過學習得到的，對所有的文檔提取單詞製作一個詞彙表，每個單詞有一個唯一的索引，即詞彙表每行代表一個單詞的embedding（詞向量），每個詞的索引可以看為一個單位向量（其實就是one-hot編碼），通過學習得到的詞向量就可以體現出相似詞語之間的關係，並且one-hot向量維度大小與詞典成正比，稠密向量大小是固定值（50~300），所以不會存在維度過大的問題，導致維度災難。
- 2.NNML：一個三層的神經網路，結合了詞向量後，通過不斷訓練採用BP演算法來對連接權和詞向量舉證進行更新，這是一個非常簡單的淺層網路但是也可以在簡單的預測問題上獲得較好的結果。


奠基之處: 本文最重要的貢獻就是為詞向量的出現埋下了伏筆

2011–Natural Language Processing (Almost) from Scratch–作者：Ronan Collobert

解決的問題

本文前part-of-speech tagging, chunking, named entity recognition, semantic role labeling使用的是原來那種 man-made 的輸入特徵，並且需要很多的先驗知識，無法自己進行特徵工程。

沒有一個統一的模型能夠解決上述提到的問題，本文嘗試使用一個模型來對這些問題進行解決

解決的方法

提出了一種新的模型，這種模型有兩種不同的模式分別用來解決不同的問題

1.window approach：window approach是基於n-gram模型的改造，視窗大小為n，中心的那個詞為中心詞，上下文各(n-1)/2個詞。

2.sentence approach是利用卷積獲取上下文並將其變成大小一致的中間表示（通過修改卷積核的大小和步伐實現）。

兩個模型最後都是最大化softmax輸出的正確標籤類別。window approach適用於POS,CHUNK,NER, sentence approach 適用於LRS。通過這兩個模型學習沒有標籤的資料從而實現無監督的學習目標特徵。

奠基之處:使用神經網路實現NER、LSR等自然語言處理的任務

### 詞向量時代

詞向量（Distributed representation， Word embedding， word representation）
- Hinton 在 1986 年的論文《Learning distributed representations of concepts》
- 實際發展爆炸的時期是在《Efficient Estimation of Word Representations in Vector Space》和《Distributed Representations of Words and Phrases and their Compositionality》兩篇論文發表之後，兩篇論文都介紹了word2Vec的方法。


詞向量的意義

現在的深度學習所使用的詞向量應該叫做Distributed representation表示的一種低維實數向量（過高的維度在深度學習之中是不利的，可能會發生維度爆炸的問題）。這種向量一般長成這個樣子：[0.792, −0.177, −0.107, 0.109, −0.542, …]。維度以 50 維和 100 維比較常見。這種向量的表示不是唯一的，後文會提到目前計算出這種向量的主流方法。（個人認為）Distributed representation 最大的貢獻就是讓相關或者相似的詞，在距離上更接近了。向量的距離可以用最傳統的歐氏距離來衡量，也可以用 cos 夾角來衡量。

里程碑論文

2013–Efficient Estimation of Word Representations in Vector Space–Mikolov

解決的問題
Hinton等其他人提出的訓練Distributed reputation的方法的複雜度仍然較高，如果在十分龐大的資料集上進行訓練的話是十分耗費時間和資源的，希望能夠提出一種更好的訓練出詞向量的方法

解決的方法
提出了兩種簡單線性模型：
- 1.CBOW：連續詞袋模型這個模型與Hinton提出的NNML很像只是去掉了非線性的隱藏層（對於NNML來說非線性的隱藏層是導致模型複雜度最主要的因素），並且不同於NNML的地方還有使用了目標詞語前後的詞語（NNMl僅僅使用了前面n個詞）。
- 2.skip—gram:和CBOW模型很相似，但它不是根據上下文預測當前單詞，而是根據同一句話中的另一個單詞最大限度地分類一個單詞。更準確地說，我們使用每個當前單詞作為一個具有連續投影層的對數線性分類器的輸入，並在當前單詞前後的一定範圍內預測單詞。


3. 里程碑

本文對詞向量訓練方法的改進，十分大的刺激了詞向量的使用，並且推動了NLP的發展可以說是里程碑事件也不為過

2013----Distributed Representations of Words and Phrases and their Compositionality–Mikolov

做出的改進:
- 1.分層softmax：在skip-gram的softmax層進行的是完全softmax而有一種Hinton等人提出的分層softmax可以很好的節約訓練的時間，有效的減少時間複雜度
- 2.負採樣（NEG）：對分層softmax我們仍然可以替換為另一種高效的方法雜訊對比估計(NCE)。它假設一個好的模型應該能夠通過 Logistic 回歸來區分資料和雜訊。由於我們只關心高品質的向量表示所以可以對NCE進行簡化為NEG。
- 3.頻繁詞採樣：為了抵消罕見詞和高頻詞之間的不平衡，我們使用簡單的二次抽樣：訓練集中的每個單詞Wi將有一定概率被丟棄

# Seq2Seq(2013)

seq2seq:
- seq2seq是一種自然語言的任務，中文翻譯過來也就是序列到序列任務
- 最經典的seq2seq任務就是機器翻譯任務（如機器翻譯、文本摘要、會話建模、圖像字幕等場景中的任務）

奠基性論文

2013–Generating Sequences With Recurrent Neural Networks–作者：Graves, Alex ，bengio

解決的問題
原則上，足夠大的RNN應該足以產生任意複雜的序列。然而，在實踐中，標準RNN無法存儲關於過去輸入的資訊很長時間。這種失憶症削弱了RNN模擬長距記憶的能力，也會損壞其穩定性，這種問題是所有條件生成模型都有的，不確定性的問題對於真實資料來說尤其嚴重。

解決的方法
- 1.傳統的解決方法：為條件模型提出的一種補救方法是將雜訊注入到預測中，然後再將其送回模型，從而將模型的魯棒性提高到驚人的投入。
- 2.本文提出的創新：本文提出了一種新型的模型來解決普通的RNN不具有長距記憶能力的問題，LSTM的結構如圖所示：


奠基之處

由於本文提出的LSTM模型具有長效記憶的能力所以對於seq2seq問題具有相當大的幫助，對後來的encoder-decoder模型起到了極為大的影響，可以說是NLP歷史上的一個大轉折

NIPS 2014–Sequence to Sequence Learning with Neural Networks–作者：Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le

解決的問題

1.一般的序列學習的簡單策略是用RNN將輸入映射為定長的向量，然後用另一個RNN將向量映射成目標序列。雖然RNN原則上可以工作，但是由於序列任務對之前的資料也有很大的依賴，過長距離的依賴使得RNN很難訓練成功

2.在自然語言處理中有很多工都可以歸約為序列到序列的任務，例如：

1.機器翻譯問題
　　2.語音辨識
　　3.圖片添加注釋
　　4.問答系統
　　5.文本摘要

這些問題都： 1）輸入和輸出可能都是不同領域 2）輸入和輸出可能長度不一致。

解決的方法

模型的三大創新點：
- 1.兩個LSTM：我們使用了兩個不同的LSTM，一個用輸入序列，一個用於輸出序列。因為這樣做可以增加模型參數的數量，但計算代價可忽略不計，並且很自然的可以在多語言對上訓練LSTM。（算是引用）
- 2.深層LSTM：我們發現深層LSTM明顯優於淺層LSTM，所以我麼選擇了四層的LSTM。
- 3.reverse input：我們發現顛倒輸入句子的單詞順序是非常有價值的。（算一個小技巧）

奠基之處:提出了Seq2Seq模型也就是等價於編碼和解碼模型即encode-decode，對於不同的問題編碼和解碼可以選擇不同的模型，以後大多數的任務都沿用和改進了這種方法

# encode-decode模型
- [Neural Machine Translation by Jointly Learning to Align and Translate(2015)](https://arxiv.org/abs/1409.0473)
- [ICLR 2015–Neural Machine Translation by Jointly Learning to Align and Translate–作者：Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio]()

encode-decode其實就是編碼和解碼的意思，自從seq2seq的論文發表以來，就出現了encode-decode的模型結構，通過各種理論及實驗證明這種結構在seq2seq的任務上具有很好的效果

```
編碼模型需要解決問題是將變長的輸入編碼成定長向量，可選模型包括：
　　1.CNN
　　2.RNN，包括LSTM、GRU以及其他變形單元
　　3.BIRNN雙向RNN，被證明在多種任務中效果優於RNN
　　4.fasttext：將詞向量進行求和

解碼模型可以選擇
　　1.RNN
　　2.AttentionModel
```

要解決的問題:
- 基於encode-decode的機器翻譯在進行編碼時需要將輸入轉換為定長的向量，這可能使神經網路難以處理長句子，尤其是那些比訓練語料庫中更長的句子。
- Cho（2014）發現，隨著輸入句子長度的增加，基本的encoder-decoder的性能會迅速下降。（encoder-decoder的缺點是難以處理長句子）

解決方法:
- 引入了一種對encoder-decoder模型的拓展。
- 每當生成的模型在翻譯中生成一個單詞的時候，它會（軟）搜索源句中最相關資訊集中的位置。
- 然後，該模型根據與源句位置相關的上下文向量和之前產生的所有目標詞來預測目標詞。（引入attention機制處理長句子）
- 這些方法與基本的encoder-decoder最大的區別是它不試圖將整個輸入序列編碼成一個定長的向量。
- 相反，它將輸入序列編碼成向量，然後當解碼翻譯的時候自我調整地選擇向量的子集。
- 這使得神經翻譯模型避免把源句的所有資訊，不管它的長度，壓扁成一個定長的向量。
- 我們發現這可以讓模型更好的處理長句子。（改進的模型與原來的區別）

- 本文的最大貢獻便是引入了注意力機制，這也是一個劃時代的發現，注意力機制的實現很大程度上的提高了模型處理長句子的能力

## Attention時代

- 奠基論文:[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention(2015)](https://proceedings.mlr.press/v37/xuc15.html)
- Attention詳細解讀：《一文讀懂Attention》

Attention: 是一種處理長句的手段，這種方法可以聚焦於一句中的關鍵字語，從而達到減少輸入的效果

要解決的問題:
- image caption是用自然語言描述圖片或者是視頻的技術，但是之前的方法是定位識別其中的物體，再生成對應的句子，這種方法只能生成較為相似的句子，效率不高。
- 後來也有人利用深度學習例如卷積神經網路搭配RNN或者LSTM，但是效果均差不多


解決方法:
- 本文的創新點在於引用了注意力機制，分為soft attention機制和hard attention機制
- soft attention機制訓練過程使用的是標準的BP演算法，hard attention機制通過最大化變分下界（變分自編碼器）實現。

奠基之處:本文主要是在image caption中引入了Attention機制並且很大程度的提高了image caption任務的效果

## Transformer(2017)
- 2017 ML– Attention Is All You Need–作者：A Vaswani，N Shazeer，N Parmar
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

解決的問題:以往nlp裡大量使用RNN結構和encoder-decoder結構，RNN及其衍生網路的缺點就是慢，問題在於前後隱藏狀態的依賴性，無法實現並行

解決方法:
- 本文提出的”Transformer”框架完全摒棄了遞迴結構，依賴注意力機制，挖掘輸入和輸出之間的關係
- 這樣做最大的好處是能夠平行計算
 
Transformer架構:
- Encoder
  - Transformer模型的Encoder由6個基本層堆疊起來
  - 每個基本層包含兩個子層
    - 第一個子層是一個注意力機制
    - 第二個是一個全連接前向神經網路。
  - 對兩個子層都引入了殘差邊以及layer normalization。
- Decoder
  - Transformer模型的Decoder也由6個基本層堆疊起來
  - 每個基本層除了Encoder裡面的那兩個以外，還增加了一層注意力機制
  - 同樣引入殘差邊以及layer normalization。

奠基之處:本文提出了Transform框架，也正式提出了注意力機制，這種完全利用注意力機制的模型，為之後的模型提供了一個非常好的範本

## Bert時代(2018)
- 2018 Computation and Language  BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding–作者：J Devlin, MW Chang, K Lee, K Toutanova
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- 預訓練(pre-training)
  - 如需要搭建一個網路模型來完成一個特定的圖像分類的任務。
  - 首先，你需要隨機初始化參數，然後開始訓練網路，不斷調整直到網路的損失越來越小。
  - 在訓練的過程中，一開始初始化的參數會不斷變化。
  - 當你覺得結果很滿意的時候，你就可以將訓練模型的參數保存下來，以便訓練好的模型可以在下次執行類似任務時獲得較好的結果。

- 微調法(fine—tuning):用別人的參數、修改後的網路和自己的資料進行訓練，使得參數適應自己的資料

所以，預訓練 就是指預先訓練的一個模型或者指預先訓練模型的過程；微調 就是指將預訓練過的模型作用於自己的資料集，並使參數適應自己資料集的過程

里程碑論文


解決的問題

1.如果使用預訓練模型處理其他任務，那人們想要的肯定不止某個詞左邊的資訊，而是左右兩邊的資訊。而考慮到這點的模型ELMo只是將left-to-right和right-to-left分別訓練拼接起來。

2.如果從頭開始一個模型的訓練需要大量的語料與時間

3.獲取比詞更高級別的句子級別的語義表徵較難

4.多工下的遷移學習現有模型難以實現

如何解決的

作者用了一個加mask的雙向Transformer


	2.採用了pre-training的方法

	3.BERT加入了Next Sentence Prediction來和Masked-LM一起做聯合訓練

	4.BERT設計了更通用的輸入層和輸出層
1
2
3
4
5
里程碑
* 引入了Masked LM，使用雙向LM做模型預訓練。
* 為預訓練引入了新目標NSP，它可以學習句子與句子間的關係。
* 進一步驗證了更大的模型效果更好： 12 --> 24 層。　
* 為下游任務引入了很通用的求解框架，不再為任務做模型定制。
* 刷新了多項NLP任務的記錄，引爆了NLP無監督預訓練技術。

BERT是穀歌團隊糅合目前已有的NLP知識集大成者，刷新11條賽道彰顯了無與倫比的實力，且極容易被用於多種NLP任務。宛若一束煙花點亮在所有NLP從業者心中。更為可貴的是穀歌選擇了開源這些，讓所有從業者看到了在各行各業落地的更多可能性。

如果想詳細瞭解bert可讀：《一文讀懂Bert》


https://blog.csdn.net/m0_46868094/article/details/114486057


## RoBERTa

## BART(2019)
- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)

## GPT-3

- GPT-3 is the third generation Generative Pre-trained Transformer model developed by OpenAI. 
- GPT-3 has 175B parameters and is trained on very large-scale internet data. 
- It has been found to achieve high performance in different classification and generation tasks
- GPT-3 architecture can also be used in different software engineering tasks that involve both natural and programming language understanding. 

- [GPT-3 Demo Showcase, 300+ Apps, Examples, & Resources]()

- OpenAI Codex 
*  Codex is a GPT-3 like model which is trained on large-scale GitHub data.
- Codex is trained on over a dozen of programming languages like Python, Java, PHP, JavaScript, and so on. 
- The video demo of OpenAI shows that Codex could generate source code from a given requirement specified in Natural Language. 
- 新聞:
  - [OpenAI 推出機器學習工具 Codex，能直接將英文轉成程式碼！(2021)](https://buzzorange.com/techorange/2021/08/12/codex-translate-english-into-code/)
  - [OpenAI開放可將自然語言轉為程式碼的AI系統Codex(2021)](https://www.ithome.com.tw/news/146142)
- Since then researchers have been investigating Codex for the automation of several SE tasks like code generation, code repair, security bug-fix, simulation modeling.
- The official documentation of Codex mentions that it is also capable of automatic documentation generation. 
- However, we are not aware of any systematic evaluation of Codex to produce code documentation.
- [Automatic Code Documentation Generation Using GPT-3(2022)](https://arxiv.org/abs/2209.02235)

- [GitHub Copilot]()
GitHub Copilot uses the OpenAI Codex to suggest code and entire functions in real-time, right from your editor.







# 參考資料
- [[论文阅读] (05) NLP知识总结及NLP论文撰写之道——Pvop老师](https://blog.csdn.net/Eastmount/article/details/109825633)
