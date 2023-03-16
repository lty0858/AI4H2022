## [scikit-learn支援的演算法 2. Unsupervised learning](https://scikit-learn.org/stable/unsupervised_learning.html)
- 2.1. Gaussian mixture models
- 2.2. Manifold learning
- 👍2.3. Clustering
- 2.4. Biclustering: Spectral Co-Clustering | Spectral Biclustering
- 👍2.5. Decomposing signals in components (matrix factorization problems)](https://scikit-learn.org/stable/modules/decomposition.html)
  - Principal component analysis (PCA) | Kernel Principal Component Analysis (kPCA)|Truncated singular value decomposition and latent semantic analysis
  - Dictionary Learning | Factor Analysis |Independent component analysis (ICA) | Non-negative matrix factorization (NMF or NNMF) |Latent Dirichlet Allocation (LDA)
- 2.6. Covariance estimation
- 👍[2.7. Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- 2.8. Density Estimation
- 👍2.9. Neural network models (unsupervised)

# [Anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection)|  Novelty Detection | Outlier Detection 孤立子偵測 
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

# 深度學習
- GAN 
- 自動編碼器（Autoencoder）與 VAE
- 其他
  - 自組織對映演算法（SOM） 

# 非監督學習(參考資訊)
- [Hands-on Unsupervised Learning Using Python](https://www.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/) 
  - [GITHUB](https://github.com/aapatel09/handson-unsupervised-learning)
  - 繁體中譯本 [非監督式學習｜使用 Python](https://www.tenlong.com.tw/products/9789865024062?list_name=srh)
- [Hands-On Unsupervised Learning with Python(2019)](https://www.packtpub.com/product/hands-on-unsupervised-learning-with-python/9781789348279)
  - 簡體中譯本 [Python 無監督學習](https://www.tenlong.com.tw/products/9787115540720?list_name=srh)
  - [GITHUB](https://github.com/PacktPublishing/Hands-on-Unsupervised-Learning-with-Python)
- [Applied Unsupervised Learning with Python](https://www.packtpub.com/product/applied-unsupervised-learning-with-python/9781789952292)
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)
  - Topic Modeling
  - Market Basket Analysis
  - Hotspot Analysis 
- [The Unsupervised Learning Workshop(2020)](https://www.packtpub.com/product/the-unsupervised-learning-workshop/9781800200708) [GITHUB](https://github.com/PacktWorkshops/The-Unsupervised-Learning-Workshop)
