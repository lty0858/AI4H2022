# TSNE:t-隨機鄰近嵌入法
- t-distributed stochastic neighbor embedding
- Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(11)
- 👍[Van der Maaten官方網址](https://lvdmaaten.github.io/tsne/) 
  - 有許多補充資料 
- [Visualizing Data Using t-SNE(2013)](https://www.youtube.com/watch?v=RJVL80Gg3lA&list=UUtXKDgv1AVoG88PLl8nGXmw)
- [機器學習_學習筆記系列(78)：t-隨機鄰近嵌入法(t-distributed stochastic neighbor embedding)](https://tomohiroliu22.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98%E7%B3%BB%E5%88%97-78-t-%E9%9A%A8%E6%A9%9F%E9%84%B0%E8%BF%91%E5%B5%8C%E5%85%A5%E6%B3%95-t-distributed-stochastic-neighbor-embedding-a0ed57759769)
- 教學影片[t-SNE(T-distributed Stochastic Neighbourhood Embedding)](https://www.youtube.com/playlist?list=PLupD_xFct8mHqCkuaXmeXhe0ajNDu0mhZ)
- t-SNE is a tool to visualize high-dimensional data. 
- It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. 
- t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results.


# [sklearn.manifold.TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)
```python

import numpy as np
from sklearn.manifold import TSNE

X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(X)

X_embedded.shape
```
