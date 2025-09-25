import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import CAT_COLS, NUM_COLS, TARGET_COLS



class LabelEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        self.encoders = {
            col: LabelEncoder().fit(X[col]) for col in X.columns
        }
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col, encoder in self.encoders.items():
            X_copy[col] = encoder.transform(X_copy[col])
        return X_copy



column_transformer = ColumnTransformer(
    transformers=[
        ('cat', LabelEncoding(), CAT_COLS),
        ('num', MinMaxScaler(), NUM_COLS)
    ], 
    remainder='passthrough' 
)


processing_pipe = Pipeline(steps=[
    ('processor', column_transformer)
])

