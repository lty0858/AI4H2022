# 集成式學習（Ensemble Learning）
- 非生成式演算法
  - 投票法（Voting）
  - 堆疊法（Stacking）
- 生成式演算法
  - 自助聚合法（Bootstrap Aggregation）
  - 提升法（Boosting）
    - 適應提升（Adaptive Boosting, AdaBoost）
    - 梯度提升（Gradient Boosting）
  - 隨機森林（Random Forest）

# [sklearn.ensemble 子模組](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)

| Classifier | Regressor |
| ------ | -------|
| ensemble.AdaBoostClassifier([estimator, ...]) <br> An AdaBoost classifier.| ensemble.AdaBoostRegressor([estimator, ...]) <br> An AdaBoost regressor.|
| ensemble.BaggingClassifier([estimator, ...])<br> A Bagging classifier.| ensemble.BaggingRegressor([estimator, ...])<br> A Bagging regressor.|
|  ensemble.ExtraTreesClassifier([...]) <br>An extra-trees classifier.| ensemble.ExtraTreesRegressor([n_estimators, ...])<br>  An extra-trees regressor.|
| ensemble.GradientBoostingClassifier(`*[, ...]`) <br>Gradient Boosting for classification.|ensemble.GradientBoostingRegressor(`*[, ...]`) <br>Gradient Boosting for regression.|
| ensemble.IsolationForest(`*[, n_estimators, ...]`) <br> Isolation Forest Algorithm.| ensemble.RandomForestClassifier([...]) <br> A random forest classifier.|
| ensemble.RandomForestRegressor([...]) <br> A random forest regressor.| ensemble.RandomTreesEmbedding([...])<br> An ensemble of totally random trees.|
| ensemble.StackingClassifier(estimators[, ...])<br> Stack of estimators with a final classifier.| ensemble.StackingRegressor(estimators[, ...])<br> Stack of estimators with a final regressor.|
| ensemble.VotingClassifier(estimators, `*[, ...]`) <br>Soft Voting/Majority Rule classifier for unfitted estimators.| ensemble.VotingRegressor(estimators, `*[, ...]`)<br>Prediction voting regressor for unfitted estimators.|
| ensemble.HistGradientBoostingRegressor([...]) <br>Histogram-based Gradient Boosting Regression Tree. |ensemble.HistGradientBoostingClassifier([...])<br> Histogram-based Gradient Boosting Classification Tree.|
```
