# [Google Colab技巧](https://github.com/TaiwanHolyHigh/AI4H2022/blob/main/week4_%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%922/GoogleColabUsage.md)
# [scikit-learn](./scikit-learn.md)
  - [資料集匯入與產生](./Datasets.md)
# 非監督學習 Unsupervised Learning
- Unsupervised Learning [scikit-learn支援的Unsupervised learning演算法](./Unsupervised.md)
- clustering
  - [k-mean Unsupervised Learning](./K-means.md)
  - other clustering algorithm 
    - [DBSCAN](./DBSCAN.md)
    - [affinity propagation clustering algorithm](./AffinityPropagationclustering.md)
    - [叢集演算法大車拚](ClusteringALL.md)
  - [Ensemble Learning with k-means](./EnsembleLearning_k-means.md)
- Dimension reduction
  - Linear dimensionality reduction  ==> [PCA](./PCA.md)
  - NON-Linear dimensionality reduction(Manifold Laerning) ==> [t-Distributed Stochastic Neighbor Embedding (t-SNE) 2008](./tsne.md)
- Anomaly DETECTION
  - [區域性異常因子(Local Outlier Factor)](./lof.md)
  - [IsolationForest 2012](./IsolationForest.md)
  - [Extended Isolation Forest (EIF) 2018](./EIF.md)
  - 其他主題
    - [Python Outlier Detection (PyOD)](https://github.com/yzhao062/pyod)  pip install pyod
    - [ODDS (Outlier Detection Datasets)](http://odds.cs.stonybrook.edu) 
    - [ADBench: Anomaly Detection Benchmark(2022)](https://www.andrew.cmu.edu/user/yuezhao2/papers/22-neurips-adbench.pdf)
    - [Handbook of Anomaly Detection: With Python Outlier Detection — (1) Introduction](https://medium.com/dataman-in-ai/handbook-of-anomaly-detection-with-python-outlier-detection-1-introduction-c8f30f71961c)
# Ensemble Learning
- Ensemble Learning
- 非生成式演算法:[投票法（Voting）](./Voting.md)
- 非生成式演算法:堆疊法（Stacking）
- 生成式演算法:自助聚合法（Bootstrap Aggregation） [Bagging](./Bagging.md)
- 生成式演算法:提升法（Boosting）
- 生成式演算法:適應提升（Adaptive Boosting, AdaBoost） [AdaBoost](./AdaBoost.md)
- 生成式演算法:梯度提升（Gradient Boosting） [Gradient Boosting](./GradientBoosting.md)
- 生成式演算法:[隨機森林（Random Forest）](./RF.md)

# 半監督學習(semi-supervised Learning)
- [半監督學習semi-supervised Learning](./Semi-supervised_learning.md)
  - [semi-supervised Learning | scikit-learn](https://scikit-learn.org/stable/modules/semi_supervised.html)
  - [Semi-Supervised Learning, Explained with Examples](https://www.altexsoft.com/blog/semi-supervised-learning/)
  - [[free online book] Semi Supervised Learning](http://www.acad.bg/ebook/ml/MITPress-%20SemiSupervised%20Learning.pdf)
- Self Training
- LabelPropagation models
  - scikit-learn provides two label propagation models: LabelPropagation and LabelSpreading. 
- co-training
- Deep Semi-Supervised Learning Algorithms
  - consistency training
  - [Unsupervised Data Augmentation for Consistency Training UDA(2019)](https://arxiv.org/abs/1904.12848)  [GITHUB](https://github.com/google-research/uda)
    - [論文閱讀筆記: Unsupervised Data Augmentation for Consistency Training (2019)](https://good74152.medium.com/%E8%AB%96%E6%96%87%E9%96%B1%E8%AE%80%E7%AD%86%E8%A8%98-unsupervised-data-augmentation-for-consistency-training-2019-a72cc30b5f8c) 
    - [整理對於UDA與MixMatch的一些想法](https://good74152.medium.com/%E6%95%B4%E7%90%86%E5%B0%8D%E6%96%BCuda%E8%88%87mixmatch%E7%9A%84%E4%B8%80%E4%BA%9B%E6%83%B3%E6%B3%95-cf31721f4120)
  - Google [MixMatch: A Holistic Approach to Semi-Supervised Learning(2019)](https://arxiv.org/abs/1905.02249) [GITHUB](https://github.com/google-research/mixmatch)
  - [GANN: Graph Alignment Neural Network for Semi-Supervised Learning(2023)]()

## Imbalanced-learn
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)
- [Examples](https://imbalanced-learn.org/stable/auto_examples/index.html)
- [API reference](https://imbalanced-learn.org/stable/references/index.html)
- [SMOTE(Synthetic Minority Oversampling Technique)](./SMOTE.md)
- 重要專案 [Credit Card Fraud Detection | Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  - 【TensorFlow 官方範例】[Classification on imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
  - [A Comparative Study of Credit Card Fraud Detection Using the Combination of Machine Learning Techniques with Data Imbalance Solution(2021)](https://www.researchgate.net/publication/353017833_A_Comparative_Study_of_Credit_Card_Fraud_Detection_Using_the_Combination_of_Machine_Learning_Techniques_with_Data_Imbalance_Solution) 
  - [Reproducible Machine Learning for Credit Card Fraud detection - Practical handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html)
- [A Gentle Introduction to Imbalanced Classification](https://machinelearningmastery.com/what-is-imbalanced-classification/)
