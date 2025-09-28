from .data_processing import column_transformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from .grid_search import grid_search
import logging
import json


with open("src/params.json", 'r') as file:
    data = dict(json.load(file))
    data = {key.replace("SVR__", ''): value for key, value in data.items()}


training_pipe = Pipeline(
    steps=[
        ('processor', column_transformer),
        ('SVR', SVR(
            **data
        ))
    ]
)

