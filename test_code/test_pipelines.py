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


df = pd.read_csv("data/raw_data/insurance.csv")


@pytest.mark.parametrize("row", df.to_dict(orient="records")[:5]) 
def test_training_pipeline(row):
    X, y = processing_pipeline_runner.process_data()
    assert X is not None
    assert y is not None

    training_pipe, validation_dict = training_pipeline_runner.train_model(activate_grid_search=True)
    assert isinstance(training_pipe, Pipeline)
    assert isinstance(validation_dict, dict)
