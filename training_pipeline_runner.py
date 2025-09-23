import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.data_handling import dataDumper, dataLoader
from src.training_pipeline import training_pipe
from src.config import CAT_COLS, NUM_COLS, TARGET_COLS, RAW_DATA_PATH, PROCESSED_X_PATH, PROCESSED_Y_PATH
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.grid_search import grid_search
from src import config
from src.model_validation import ModelValidation
from processing_pipeline_runner import process_data

def train_model(activate_grid_search:bool=True) -> Pipeline:

    """ 
        Args: activate_grid_search: bool
        Description: This function will train pipeline, generate validation metrics.
        Return: tuple(sklearn.pipeline.Pipeline, dict)
                function will return trained pipeline and dictionary with metrics
        """

    X, y = process_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if activate_grid_search:
        grid_search.fit(X, y)
        config.BEST_PARAMS = grid_search.best_params_

    training_pipe.fit(X_train, y_train)
    model_validation = ModelValidation(
                                        model=training_pipe,
                                        X_train=X_train,
                                        X_test=X_test,
                                        y_train=y_train,
                                        y_test=y_test
                                       )
    
    validation_dict = model_validation.metrics()

    print(training_pipe, validation_dict)
    return training_pipe, validation_dict




if __name__ == "__main__":
    train_model()


