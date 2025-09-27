import pandas as pd
import numpy as np
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
import logging

logging.basicConfig(
    filename='logs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)


logging.info("Logging is started ...")


def train_model(activate_grid_search:bool=True, return_data:bool=False) -> Pipeline:

    """ 
        Args: 
            activate_grid_search: bool (if true, it will find best parameters using grid search)
            return_data: bool
        Return: tuple(sklearn.pipeline.Pipeline, dict)
                function will return trained pipeline and dictionary with metrics
        Description: This function will train pipeline, generate validation metrics.
        """

    X, y = process_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if activate_grid_search:
        logging.info("Searching for best parameteres ...")
        grid_search.fit(X, y)
        config.BEST_PARAMS = grid_search.best_params_
        logging.info("Found best parameters are ", config.BEST_PARAMS)

    logging.info("Fitting training pipeline ...")
    training_pipe.fit(X_train, y_train)

    logging.info("Creating Instance of model validation ...")
    model_validation = ModelValidation(
                                        model=training_pipe,
                                        X_train=X_train,
                                        X_test=X_test,
                                        y_train=y_train,
                                        y_test=y_test
                                       )

    logging.info("Calling metrics function of model validation instance ...")
    validation_dict = model_validation.metrics()

    if return_data:
        logging.info(f"Returning training pipe and validation dictionary {validation_dict} with splitted data ...")
        return training_pipe, validation_dict, X_train, X_test, y_train, y_test
    logging.info(f"Returning training pipe and validation dictionary {validation_dict} ...")
    return training_pipe, validation_dict




if __name__ == "__main__":
    train_model(activate_grid_search=False)


