from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import logging

param_grid = {
    'SVR__kernel': ['linear', 'rbf', 'poly'],   
    'SVR__C': [0.1, 1, 10, 100],               
    'SVR__gamma': ['scale', 'auto'],            
    'SVR__epsilon': [0.01, 0.1, 0.5, 1]         
}


grid_search = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)


def gridSearch(pipeline):
    grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2
    )
    return grid_search

