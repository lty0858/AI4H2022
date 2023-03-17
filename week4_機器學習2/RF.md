#
- Bagging 通常用於減少模型的方差。
- 它通過創建一組基礎學習器來實現它，每個基礎學習器都在原始訓練集的一個獨特的引導樣本上進行訓練。這迫使基礎學習者之間的多樣性。
- 隨機森林通過不僅在每個基礎學習器的訓練樣本上而且在特徵中引入隨機性來擴展裝袋。
- 此外，它們的性能類似於提陞技術，儘管它們不需要像提升方法那樣多的微調。



```
選擇m (每個節點將考慮的特徵數) Select the number of features m that will be considered at each node

對於每個基學習器，執行以下操作：
創建引導訓練樣本
選擇要拆分的節點
隨機選擇m 個特徵
從m中選擇最好的特徵和分割點
將節點拆分為兩個節點
從步驟 2-2 重複直到滿足停止條件，例如最大樹深度
```

# scikit-learn 既實現了傳統的隨機森林樹，也實現了額外的樹。

## 使用隨機森林分類集成 對 `手寫數字資料集`  進行分類  see ch7.2
```python
# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np

digits = load_digits()


train_size = 1500
train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

np.random.seed(123456)

# --- SECTION 2 ---
# Create the ensemble
ensemble_size = 500
ensemble = RandomForestClassifier(n_estimators=ensemble_size, n_jobs=4)

# --- SECTION 3 ---
# Train the ensemble
ensemble.fit(train_x, train_y)

# --- SECTION 4 ---
# Evaluate the ensemble
ensemble_predictions = ensemble.predict(test_x)

ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

# --- SECTION 5 ---
# Print the accuracy
print('Random Forest: %.2f' % ensemble_acc)
```

- validation_curve  rf_classification_validation_curves.py
```python
# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()


train_size = 1500
train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

np.random.seed(123456)
# --- SECTION 2 ---
# Create the ensemble
ensemble_size = 500
ensemble = RandomForestClassifier(n_estimators=ensemble_size, n_jobs=4)

param_range = [10, 50, 100, 150, 200, 250, 300, 350, 400]
train_scores, test_scores = validation_curve(ensemble, train_x, train_y, 'n_estimators', param_range,
                       cv=10, scoring='accuracy')

# --- SECTION 3 ---
# Calculate the average and standard deviation for each hyperparameter
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


# --- SECTION 4 ---
# Plot the scores
plt.figure()
plt.title('Validation curves (Random Forest)')
# Plot the standard deviations
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="C1")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="C0")

# Plot the means
plt.plot(param_range, train_scores_mean, 'o-', color="C1",
         label="Training score")
plt.plot(param_range, test_scores_mean, 'o-', color="C0",
         label="Cross-validation score")

plt.xticks(param_range)
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')
plt.legend(loc="best")
```

## 使用隨機森林分類集成 對 `手寫數字資料集`  進行分類  see ch7.3  rf_regression.py 
```python
# --- SECTION 1 ---
# Libraries and data loading
from copy import deepcopy
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import numpy as np

diabetes = load_diabetes()

train_size = 400
train_x, train_y = diabetes.data[:train_size], diabetes.target[:train_size]
test_x, test_y = diabetes.data[train_size:], diabetes.target[train_size:]

np.random.seed(123456)

# --- SECTION 2 ---
# Create the ensemble
ensemble_size = 1000
ensemble = RandomForestRegressor(n_estimators=ensemble_size,
                                 min_samples_leaf=20, n_jobs=4)

# --- SECTION 3 ---
# Evaluate the ensemble
ensemble.fit(train_x, train_y)
predictions = ensemble.predict(test_x)

# --- SECTION 4 ---
# Print the metrics
r2 = metrics.r2_score(test_y, predictions)
mse = metrics.mean_squared_error(test_y, predictions)

print('Random Forest:')
print('R-squared: %.2f' % r2)
print('MSE: %.2f' % mse)
```
