# 集成式學習（Ensemble Learning）
- 非生成式演算法
  - 投票法（Voting）
  - 堆疊法（Stacking）

# [sklearn.ensemble](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)

| Classifier | Regressor |
| ------ | -------|
| ensemble.AdaBoostClassifier([estimator, ...]) <br> An AdaBoost classifier.| ensemble.AdaBoostRegressor([estimator, ...]) <br> An AdaBoost regressor.|

ensemble.BaggingClassifier([estimator, ...])

A Bagging classifier.

ensemble.BaggingRegressor([estimator, ...])

A Bagging regressor.

ensemble.ExtraTreesClassifier([...])

An extra-trees classifier.

ensemble.ExtraTreesRegressor([n_estimators, ...])

An extra-trees regressor.

ensemble.GradientBoostingClassifier(*[, ...])

Gradient Boosting for classification.

ensemble.GradientBoostingRegressor(*[, ...])

Gradient Boosting for regression.

ensemble.IsolationForest(*[, n_estimators, ...])

Isolation Forest Algorithm.

ensemble.RandomForestClassifier([...])

A random forest classifier.

ensemble.RandomForestRegressor([...])

A random forest regressor.

ensemble.RandomTreesEmbedding([...])

An ensemble of totally random trees.

ensemble.StackingClassifier(estimators[, ...])

Stack of estimators with a final classifier.

ensemble.StackingRegressor(estimators[, ...])

Stack of estimators with a final regressor.

ensemble.VotingClassifier(estimators, *[, ...])

Soft Voting/Majority Rule classifier for unfitted estimators.

ensemble.VotingRegressor(estimators, *[, ...])

Prediction voting regressor for unfitted estimators.

ensemble.HistGradientBoostingRegressor([...])

Histogram-based Gradient Boosting Regression Tree.

ensemble.HistGradientBoostingClassifier([...])

Histogram-based Gradient Boosting Classification Tree.
```
