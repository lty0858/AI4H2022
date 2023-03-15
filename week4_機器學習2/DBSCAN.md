# DBSCAN 1996
- Density-based spatial clustering of applications with noise (DBSCAN)
- [重要值得看的wiki](https://en.wikipedia.org/wiki/DBSCAN)
- [DBSCAN Clustering Algorithm in Machine Learning(2022)](https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html#:~:text=low%20point%20density.-,Density%2DBased%20Spatial%20Clustering%20of%20Applications%20with%20Noise%20(DBSCAN),is%20containing%20noise%20and%20outliers.)
# [DBSCAN 範例 frpm sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
```python
from sklearn.cluster import DBSCAN
import numpy as np

X = np.array([[1, 2], [2, 2], [2, 3],[8, 7], [8, 8], [25, 80]])

clustering = DBSCAN(eps=3, min_samples=2).fit(X)

clustering.labels_

clustering
```
- [Demo of DBSCAN clustering algorithm官方網站範例]()
# [DBSCAN 範例]()
```python
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

from sklearn.cluster import DBSCAN
import numpy as np

X = np.array([[1, 2], [2, 2], [2, 3],[8, 7], [8, 8], [25, 80]])

clustering = DBSCAN(eps=3, min_samples=2).fit(X)

clustering.labels_

clustering
```
