#
- K-means 運作的流程步驟：
  - 1.首先設定要分成多少群：K
  - 2.然後在特徵空間中隨機設定K個群心。
  - 3.計算每一個資料點到K個群心的距離 ( 基本上使用 L2距離，但也是可以換成別的。)
  - 4.將資料點分給距離最近的那個群心。
  - 5.在所有資料點都分配完畢後，每一群再用剛剛分配到的資料點算平均(means)來更新群心。
  - 6.最後不斷重複3–5 的動作，直到收斂 ( 每次更新後群心都已經不太會變動 ) 後結束。
# 
```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
kmeans.labels_

kmeans.predict([[0, 0], [12, 3]])

kmeans.cluster_centers_
```


# Ensemble Learning with k-means
- [Hands-On Ensemble Learning with Python: Build highly optimized ensemble machine learning models using scikit-learn and Keras](https://www.packtpub.com/product/hands-on-ensemble-learning-with-python/9781789612851) [GITHUB](https://github.com/PacktPublishing/Hands-On-Ensemble-Learning-with-Python)
  - 繁體中譯本[集成式學習：Python 實踐！整合全部技術，打造最強模型](https://www.tenlong.com.tw/products/9789863126942?list_name=srh) CH8
- !pip install openensembles
```python
# Libraries and data loading
import openensembles as oe
import numpy as np
import pandas as pd
import sklearn.metrics

from sklearn.datasets import load_breast_cancer

## TSNE降維
#from sklearn.manifold import TSNE
#t = TSNE()

bc = load_breast_cancer()

# Create the data object
cluster_data = oe.data(pd.DataFrame(bc.data), bc.feature_names)

# cluster_data = oe.data(pd.DataFrame(t.fit_transform(bc.data)), [0,1])

np.random.seed(123456)
```

- oe_co_occurence.py 
```python
# --- SECTION 3 ---
# Create the ensembles and calculate the homogeneity score
for K in [2, 3, 4, 5, 6, 7]:
    for ensemble_size in [3, 4, 5]:
        ensemble = oe.cluster(cluster_data)
        for i in range(ensemble_size):
            name = f'kmeans_{ensemble_size}_{i}'
            ensemble.cluster('parent', 'kmeans', name, K)

        preds = ensemble.finish_co_occ_linkage(threshold=0.5)
        print(f'K: {K}, size {ensemble_size}:', end=' ')
        print('%.2f' % sklearn.metrics.homogeneity_score(
                bc.target, preds.labels['co_occ_linkage']))
```

- oe_graph_closure.py 
```python
# --- SECTION 3 ---
# Create the ensembles and calculate the homogeneity score
for K in [2, 3, 4, 5, 6, 7]:
    for ensemble_size in [3, 4, 5]:
        ensemble = oe.cluster(cluster_data)
        for i in range(ensemble_size):
            name = f'kmeans_{ensemble_size}_{i}'
            ensemble.cluster('parent', 'kmeans', name, K)

        preds = ensemble.finish_graph_closure(threshold=0.5)
        print(f'K: {K}, size {ensemble_size}:', end=' ')
        print('%.2f' % sklearn.metrics.homogeneity_score(
                bc.target, preds.labels['graph_closure']))
```

- oe_vote.py  VS oe_vote_tsne.py 
```python
# --- SECTION 3 ---
# Create the ensembles and calculate the homogeneity score
for K in [2, 3, 4, 5, 6, 7]:
    for ensemble_size in [3, 4, 5]:
        ensemble = oe.cluster(cluster_data)
        for i in range(ensemble_size):
            name = f'kmeans_{ensemble_size}_{i}'
            ensemble.cluster('parent', 'kmeans', name, K)

        preds = ensemble.finish_majority_vote(threshold=0.5)
        print(f'K: {K}, size {ensemble_size}:', end=' ')
        print('%.2f' % sklearn.metrics.homogeneity_score(
                bc.target, preds.labels['majority_vote']))
```


