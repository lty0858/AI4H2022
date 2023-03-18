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
  - [IsolationForest](./IsolationForest.md)
  - 其他主題
    - [Python Outlier Detection (PyOD)](https://github.com/yzhao062/pyod)  pip install pyod
    - [ODDS (Outlier Detection Datasets)](http://odds.cs.stonybrook.edu) 
    - [ADBench: Anomaly Detection Benchmark(2022)](https://www.andrew.cmu.edu/user/yuezhao2/papers/22-neurips-adbench.pdf)
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
  - [Unsupervised Data Augmentation for Consistency Training UDA(2019)](https://arxiv.org/abs/1904.12848)  [GITHUB](https://github.com/google-research/uda)
  - Google [MixMatch: A Holistic Approach to Semi-Supervised Learning(2019)](https://arxiv.org/abs/1905.02249)
