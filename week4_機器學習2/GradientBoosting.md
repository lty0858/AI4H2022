# Gradient Boosting 1999
- [Gradient boosting|WIKI](https://en.wikipedia.org/wiki/Gradient_boosting)
- [Friedman, J. H. (February 1999). "Greedy Function Approximation: A Gradient Boosting Machine"](https://jerryfriedman.su.domains/ftp/trebst.pdf)

## Gradient Boosting實作大車拚
- Python 硬派實作
- 使用scikit-learn
- 使用XGBoost(eXtreme Gradient Boosting)
  - [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/) 
  - [Hands-On Gradient Boosting with XGBoost and scikit-learn(2020)](https://www.packtpub.com/product/hands-on-gradient-boosting-with-xgboost-and-scikit-learn/9781839218354)
- 使用LightGBM (Light Gradient Boosting Machine) 2017
  - 微軟團隊[LightGBM: A Highly Efficient Gradient Boosting Decision Tree ](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).
  - [Predicting Diabetes Using LightGBM and SHAP]()
- 使用CatBoost
  - [你听过 CatBoost 吗？本文教你如何使用 CatBoost 进行快速梯度提升](https://aijishu.com/a/1060000000143372) 
# 迴歸
- 資料集 [Diabetes(糖尿病) Dataset| Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
- [sklearn.datasets.load_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
### Python 硬派實作 see ch6.6
```python
# --- SECTION 1 ---
# Libraries and data loading
from copy import deepcopy
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

import numpy as np

diabetes = load_diabetes()

train_size = 400
train_x, train_y = diabetes.data[:train_size], diabetes.target[:train_size]
test_x, test_y = diabetes.data[train_size:], diabetes.target[train_size:]

np.random.seed(123456)

# --- SECTION 2 ---
# Create the ensemble

# Define the ensemble's size, learning rate and decision tree depth
ensemble_size = 50
learning_rate = 0.1
base_classifier = DecisionTreeRegressor(max_depth=3)

# Create placeholders for the base learners and each step's prediction
base_learners = []

# Note that the initial prediction is the target variable's mean

previous_predictions = np.zeros(len(train_y)) + np.mean(train_y)

# Create the base learners
for _ in range(ensemble_size):
    # Start by calcualting the pseudo-residuals
    errors = train_y - previous_predictions

    # Make a deep copy of the base classifier and train it on the
    # pseudo-residuals
    learner = deepcopy(base_classifier)
    learner.fit(train_x, errors)

    # Predict the residuals on the train set
    predictions = learner.predict(train_x)

    # Multiply the predictions with the learning rate and add the results
    # to the previous prediction
    previous_predictions = previous_predictions + learning_rate*predictions

    # Save the base learner
    base_learners.append(learner)

# --- SECTION 3 ---
# Evaluate the ensemble

# Start with the train set's mean
previous_predictions = np.zeros(len(test_y)) + np.mean(train_y)

# For each base learner predict the pseudo-residuals for the test set and
# add them to the previous prediction, after multiplying with the learning rate
for learner in base_learners:
    predictions = learner.predict(test_x)
    previous_predictions = previous_predictions + learning_rate*predictions

# --- SECTION 4 ---
# Print the metrics
r2 = metrics.r2_score(test_y, previous_predictions)
mse = metrics.mean_squared_error(test_y, previous_predictions)

print('Gradient Boosting:')
print('R-squared: %.2f' % r2)
print('MSE: %.2f' % mse)
```



# 分類


# [Gradient Boosting Algorithm: A Complete Guide for Beginners](https://www.analyticsvidhya.com/blog/2021/09/gradient-boosting-algorithm-a-complete-guide-for-beginners/)
- [資料集 Income classification@kaggle](https://www.kaggle.com/lodetomasi1995/income-classification)
  - [Random Forest Classifier + Feature Importance](https://www.kaggle.com/code/prashant111/random-forest-classifier-feature-importance) 
- [GITHUB](https://github.com/AnshulSaini17/Income_evaluation)
