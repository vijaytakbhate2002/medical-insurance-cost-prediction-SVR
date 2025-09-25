from data_processing import column_transformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from grid_search import grid_search
from config import BEST_PARAMS
import logging


training_pipe = Pipeline(
    steps=[
        ('processor', column_transformer),
        ('SVR', SVR(
            **BEST_PARAMS
        ))
    ]
)

