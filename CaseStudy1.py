# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:06:38 2024

@author: tobys
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

#Initial model selection
df = pd.read_csv('C:/users/tobys/Downloads/train.csv')

X, y = df.iloc[:, :-1].values, df['Quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#scale data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

models = []
models.append(('LogReg', LogisticRegression()))
models.append(('SVC', SVC()))
models.append(('Decision Tree', DecisionTreeClassifier()))

# evaluate each model in turn
results = []
names = []
metrics = []
scoring = ['accuracy', 'f1']
for name, model in models:
    kfold = StratifiedKFold(n_splits=10)
    cv_results = cross_validate(model, X_train_std, y_train, cv=kfold, scoring=scoring)
    accuracy, f1 = cv_results['test_accuracy'], cv_results['test_f1']
    results = results + [i for i in accuracy] + [i for i in f1]
    names = names + [name for i in range(0,20)]
    metrics = metrics + ['accuracy' for i in range(0,10)] + ['f1' for i in range(0,10)]

dict = {'Model': names, 'Scoring Metric': metrics, 'Score': results}
results_df = pd.DataFrame(dict)  
sns.boxplot(data = results_df, x="Model", y="Score", hue='Scoring Metric')

#Learning curve
train_sizes, train_scores, test_scores = learning_curve(estimator=SVC(), X=X_train_std, y=y_train, train_sizes=np.linspace( 0.1, 1.0, 10), cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o',
markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.03])

#Parameter tuning
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'C': param_range, 'kernel': ['linear']}, {'C': param_range, 'gamma': param_range, 'kernel': ['rbf']}]
gs = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='accuracy', cv=10)
gs.fit(X_train_std, y_train)
print(gs.best_score_)
print(gs.best_params_)

#feature selection
sns.heatmap(df.corr())
results = []
num_features = []
metrics = []
from sklearn.feature_selection import SequentialFeatureSelector
for i in range(3,7):
    sfs = SequentialFeatureSelector(SVC(kernel='rbf', gamma=0.1), n_features_to_select=i)
    sfs.fit(X_train_std, y_train)
    X_sfs = sfs.transform(X_train_std)
    kfold = StratifiedKFold(n_splits=10)
    kfold_results = cross_validate(SVC(kernel='rbf', gamma=0.1), X_sfs, y_train, cv=kfold, scoring=scoring)
    accuracy, f1 = kfold_results['test_accuracy'], cv_results['test_f1']
    results = results + [j for j in accuracy] + [j for j in f1]
    num_features = num_features + [i for j in range(0,20)]
    metrics = metrics + ['accuracy' for j in range(0,10)] + ['f1' for j in range(0,10)]
feature_dict = {'Number of Features': num_features, 'Scoring Metric': metrics, 'Score': results}
feature_df = pd.DataFrame(feature_dict)  
sns.boxplot(data = feature_df, x="Number of Features", y="Score", hue='Scoring Metric')