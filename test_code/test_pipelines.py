import pytest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)
sys.path.append(PROJECT_ROOT)

from data_handling import dataDumper, dataLoader
from grid_search import grid_search
from model_validation import ModelValidation
import training_pipeline_runner
import processing_pipeline_runner
import config


if os.path.exists("raw_data\\insurance.csv"):
    df = pd.read_csv("raw_data\\insurance.csv")
else:
    df = pd.read_csv("test_code\\raw_data\\insurance.csv")


def test_processing_pipeline():
    X, y = processing_pipeline_runner.process_data(df=df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    
def test_training_pipeline():
    X, y = processing_pipeline_runner.process_data(df=df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    training_pipe, validation_dict = training_pipeline_runner.train_model(activate_grid_search=True)
    assert isinstance(training_pipe, Pipeline)
    assert isinstance(validation_dict, dict)
