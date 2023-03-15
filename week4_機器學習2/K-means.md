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
