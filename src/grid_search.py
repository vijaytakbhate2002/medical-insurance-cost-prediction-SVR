from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],   
    'C': [0.1, 1, 10, 100],               
    'gamma': ['scale', 'auto'],            
    'epsilon': [0.01, 0.1, 0.5, 1]         
}

grid_search = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

