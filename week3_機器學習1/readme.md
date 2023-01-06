# 機器學習


# 1.機器學習演算法 [開發套件 scikit-learn](https://scikit-learn.org/stable/)

## 監督式學習Supervised learning  [scikit-learn支援的演算法](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- 回歸Regression  [scikit-learn]()
  - 常用回歸Regression
    - Ordinary Least Squares
      - [不用套件硬功夫](https://github.com/m0-k1/Linear-Regression_model)
      - 範例研讀
        - [07_Simple_and_Multiple_Regression](https://github.com/sandipanpaul21/Machine-Learning-in-Python-Code/blob/master/07_Simple_and_Multiple_Regression.ipynb) 
          - [範例筆記](https://github.com/sandipanpaul21/ML-Notes-Daywise)
        - [karthickai/Linear-Regression](https://github.com/karthickai/Linear-Regression) 
    - Ridge regression and classification
    - Lasso regression
    - Nearest Neighbors Regression 
- 分類Classification ==> 二元分類(有病|沒病,正常|異常) VS 多元分類(不同等級A|B|C|D|E|...的水果)
  - Nearest Neighbors
  - Naive Bayes
  - Decision Trees決策樹
  - [Support Vector Machines 支援向量機](https://scikit-learn.org/stable/modules/svm.html#support-vector-machines)
    - SVC, NuSVC and LinearSVC are classes capable of performing binary and multi-class classification on a dataset.
  - Ensemble methods(超熱門)
    - [集成式學習：Python 實踐！Hands-On Ensemble Learning with Python](https://www.tenlong.com.tw/products/9789863126942?list_name=srh)
    - [Ensemble Machine Learning Cookbook(2019)](https://www.tenlong.com.tw/products/9781789136609?list_name=srh)
  - 效能評估 see [Classification metrics]()
    - The `sklearn.metrics` module implements several loss, score, and utility functions to measure classification performance
    - 3.3.2.2. Accuracy score
    - 3.3.2.7. Classification report
## 非監督式學習Unsupervised learning [scikit-learn支援的演算法 2. Unsupervised learning](https://scikit-learn.org/stable/unsupervised_learning.html)
- 叢集演算法 see [scikit-learn 2.3. Clustering]()
  - K-means [2.3.2. K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
  - Hierarchical clustering see [2.3.6. Hierarchical clustering]()
  - DBSCAN clustering algorithm
    - [Demo of DBSCAN clustering algorithm](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py)
  - 效能評估 see [2.3.10. Clustering performance evaluation]()
    - 2.3.10.1. Rand index
    - 2.3.10.2. Mutual Information based scores
    - 2.3.10.3. Homogeneity, completeness and V-measure
    - 2.3.10.4. Fowlkes-Mallows scores
    - 2.3.10.5. Silhouette Coefficient
    - 2.3.10.6. Calinski-Harabasz Index
    - 2.3.10.7. Davies-Bouldin Index
    - 2.3.10.8. Contingency Matrix
    - 2.3.10.9. Pair Confusion Matrix
- Principal component analysis (PCA)
  - [2.5. Decomposing signals in components (matrix factorization problems)](https://scikit-learn.org/stable/modules/decomposition.html)
- [Anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection)|  Novelty Detection | Outlier Detection 孤立子偵測 see [2.7. Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
  - Local Outlier Factor (LOF) 2000
    - 論文 [LOF: Identifying Density-Based Local Outliers ](https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf) 
  - 孤立森林(Isolation Forest)2008 
    - [論文](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest)
    - [WIKI](https://en.wikipedia.org/wiki/Isolation_forest) 
    - [sklearn.ensemble.IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
      - from sklearn.ensemble import IsolationForest
      - X = [[-1.1], [0.3], [0.5], [100]]
      - clf = IsolationForest(random_state=0).fit(X)
      - clf.predict([[0.1], [0], [90]])
    - [Anomaly detection using Isolation Forest – A Complete Guide](https://www.analyticsvidhya.com/blog/2021/07/anomaly-detection-using-isolation-forest-a-complete-guide/)
  - 深度學習
    - GAN 
    - 自動編碼器（Autoencoder）與 VAE
  - 其他
    - 自組織對映演算法（SOM） 
  - 推薦書籍
    - [非監督式學習｜使用 Python (Hands-On Unsupervised Learning Using Python)](https://www.tenlong.com.tw/products/9789865024062?list_name=srh)
    - [Hands-On Unsupervised Learning with Python](https://www.tenlong.com.tw/products/9781789348279?list_name=srh) 簡中譯本[Python 無監督學習](https://www.tenlong.com.tw/products/9787115540720?list_name=srh)
    - [Applied Unsupervised Learning with Python(2019)](https://www.tenlong.com.tw/products/9781789952292?list_name=srh) 

## 半監督式學習Semi-supervised learning

## 強化學習Reinforcement learning

# 2.機器學習開發平台 
- [scikit-learn: machine learning in Python](https://scikit-learn.org/)

# 3.機器學習應用場域
- 智慧資安
- 智慧金融
- 智慧製造

# 4.機器學習學習
- [Kaggle: Your Machine Learning and Data Science Community](https://www.kaggle.com/)
  - [Datasets](https://www.kaggle.com/) 
    - 學習常用資料集
      - [Breast Cancer Wisconsin (Diagnostic) Data Set@Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/code)
        - [分析1:thesis-v1](https://www.kaggle.com/code/ahsanadiba/thesis-v1/notebook) 
    - 資訊安全
      - kddcup99
        - 1998年美國國防部高級規劃署（DARPA）在MIT林肯實驗室進行了一項入侵檢測評估項目。
        - 林肯實驗室建立了模擬美國空軍局域網的一個網絡環境，收集了9周時間的 TCPdump網絡連接和系統審計數據，仿真各種用戶類型、各種不同的網絡流量和攻擊手段，使它就像一個真實的網絡環境
        - 這些TCPdump採集的原始數據被分爲兩個部分：7周時間的訓練數據 大概包含5,000,000多個網絡連接記錄，剩下的2周時間的測試數據大概包含2,000,000個網絡連接記錄。
        - [參看參數說明KDD CUP 99數據集分析](https://www.twblogs.net/a/5c9e8158bd9eee7523887b96)
        - [入侵检测之KDDCUP99数据集分析](https://blog.csdn.net/qq_38384924/article/details/97128744)
        - [A detailed analysis of the KDD CUP 99 data set](https://ieeexplore.ieee.org/document/5356528)
        - [sklearn.datasets.fetch_kddcup99](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html#sklearn.datasets.fetch_kddcup99)
        - [KDD Cup 1999 analysis@Kaggle](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data/code)
        - [研讀Anomaly Detection using KDD99 ](https://www.kaggle.com/code/tsenglung/anomaly-detection-using-kdd99/edit)
        - 分析 [Evaluation of outlier detection estimators](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html#sphx-glr-auto-examples-miscellaneous-plot-outlier-detection-bench-py)
      - [UNSW_NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) 
      - [IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv)
    - 圖片image classification
      - [Cats and Dogs image classification](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification)
      - [Cotton plant disease](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification)
      - [Pothole Detection Dataset](https://www.kaggle.com/datasets/rajdalsaniya/pothole-detection-dataset)
    - 自然語言處理
      - [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  
  # 其他
  - https://github.com/leemengtaiwan/deep-learning-resources
