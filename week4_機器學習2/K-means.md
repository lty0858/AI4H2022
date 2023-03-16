#
- K-means 運作的流程步驟：
  - 1.首先設定要分成多少群：K
  - 2.然後在特徵空間中隨機設定K個群心。
  - 3.計算每一個資料點到K個群心的距離 ( 基本上使用 L2距離，但也是可以換成別的。)
  - 4.將資料點分給距離最近的那個群心。
  - 5.在所有資料點都分配完畢後，每一群再用剛剛分配到的資料點算平均(means)來更新群心。
  - 6.最後不斷重複3–5 的動作，直到收斂 ( 每次更新後群心都已經不太會變動 ) 後結束。
# [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
kmeans.labels_

kmeans.predict([[0, 0], [12, 3]])

kmeans.cluster_centers_
```

# [[演算法] K-means 分群 (K-means Clustering)](https://ithelp.ithome.com.tw/articles/10209058)
```python
import numpy as np
import matplotlib.pyplot as plt

# 群集中心和元素的數量
seed_num = 3
dot_num = 20

# 初始元素
x = np.random.randint(0, 500, dot_num)
y = np.random.randint(0, 500, dot_num)
# 初始群集中心
kx = np.random.randint(0, 500, seed_num)
ky = np.random.randint(0, 500, seed_num)


# 兩點之間的距離
def dis(x, y, kx, ky):
    return int(((kx-x)**2 + (ky-y)**2)**0.5)

# 對每筆元素進行分群
def cluster(x, y, kx, ky):
    team = []
    for i in range(3):
        team.append([])
    mid_dis = 99999999
    for i in range(dot_num):
        for j in range(seed_num):
            distant = dis(x[i], y[i], kx[j], ky[j])
            if distant < mid_dis:
                mid_dis = distant
                flag = j
        team[flag].append([x[i], y[i]])
        mid_dis = 99999999
    return team

# 對分群完的元素找出新的群集中心
def re_seed(team, kx, ky):
    sumx = 0
    sumy = 0
    new_seed = []
    for index, nodes in enumerate(team):
        if nodes == []:
            new_seed.append([kx[index], ky[index]])
        for node in nodes:
            sumx += node[0]
            sumy += node[1]
        new_seed.append([int(sumx/len(nodes)), int(sumy/len(nodes))])
        sumx = 0
        sumy = 0
    nkx = []
    nky = []
    for i in new_seed:
        nkx.append(i[0])
        nky.append(i[1])
    return nkx, nky

# k-means 分群
def kmeans(x, y, kx, ky, fig):
    team = cluster(x, y, kx, ky)
    nkx, nky = re_seed(team, kx, ky)
    
    # plot: nodes connect to seeds
    cx = []
    cy = []
    line = plt.gca()
    for index, nodes in enumerate(team):
        for node in nodes:
            cx.append([node[0], nkx[index]])
            cy.append([node[1], nky[index]])
        for i in range(len(cx)):
            line.plot(cx[i], cy[i], color='r', alpha=0.6)
        cx = []
        cy = []
    
    # 繪圖
    feature = plt.scatter(x, y)
    k_feature = plt.scatter(kx, ky)
    nk_feaure = plt.scatter(np.array(nkx), np.array(nky), s=50)
    plt.savefig('/yourPATH/kmeans_%s.png' % fig)
    plt.show()

    # 判斷群集中心是否不再更動
    if nkx == list(kx) and nky == (ky):
        return
    else:
        fig += 1
        kmeans(x, y, nkx, nky, fig)


kmeans(x, y, kx, ky, fig=0)
```
#  KMeans
- [Hands-On Ensemble Learning with Python: Build highly optimized ensemble machine learning models using scikit-learn and Keras](https://www.packtpub.com/product/hands-on-ensemble-learning-with-python/9781789612851) [GITHUB](https://github.com/PacktPublishing/Hands-On-Ensemble-Learning-with-Python)
  - 繁體中譯本[集成式學習：Python 實踐！整合全部技術，打造最強模型](https://www.tenlong.com.tw/products/9789863126942?list_name=srh) CH8-2
```PYTHON
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.manifold import TSNE


np.random.seed(123456)

bc = load_breast_cancer()
tsne = TSNE()

data = tsne.fit_transform(bc.data)
reds = bc.target == 0
blues = bc.target == 1
plt.scatter(data[reds, 0], data[reds, 1], label='malignant')
plt.scatter(data[blues, 0], data[blues, 1], label='benign')
plt.xlabel('1st Component')
plt.ylabel('2nd Component')
plt.title('Breast Cancer dataa')
plt.legend()


plt.figure()
plt.title('2, 4, and 6 clusters.')
for clusters in [2, 4, 6]:
    km = KMeans(n_clusters=clusters)
    preds = km.fit_predict(data)
    plt.subplot(1, 3, clusters/2)
    plt.scatter(*zip(*data), c=preds)

    classified = {x: {'m': 0, 'b': 0} for x in range(clusters)}

    for i in range(len(data)):
        cluster = preds[i]
        label = bc.target[i]
        label = 'm' if label == 0 else 'b'
        classified[cluster][label] = classified[cluster][label]+1

    print('-'*40)
    for c in classified:
        print('Cluster %d. Malignant percentage: ' % c, end=' ')
        print(classified[c], end=' ')
        print('%.3f' % (classified[c]['m'] /
                        (classified[c]['m'] + classified[c]['b'])))

    print(metrics.homogeneity_score(bc.target, preds))
    print(metrics.silhouette_score(data, preds))
```




