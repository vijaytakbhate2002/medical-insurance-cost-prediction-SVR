import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data_handling import dataDumper, dataLoader
from src.training_pipeline import training_pipe
from src.config import CAT_COLS, NUM_COLS, TARGET_COLS, RAW_DATA_PATH, PROCESSED_X_PATH, PROCESSED_Y_PATH
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.grid_search import grid_search, gridSearch
from src import config
from src.model_validation import ModelValidation
from processing_pipeline_runner import process_data
import logging
import json
import joblib


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


    logging.info("Splitting raw data to fit ...")
    df = dataLoader(path=RAW_DATA_PATH)
    raw_X, raw_y = df.drop(TARGET_COLS, axis='columns'), df[TARGET_COLS]


    logging.info("Scaling target column with minmax")
    target_scaling = MinMaxScaler()
    y_scaled = target_scaling.fit_transform(raw_y.values.reshape(-1, 1))
    joblib.dump(target_scaling, "data/target_scaling.pkl")


    y_scaled = y_scaled.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        raw_X, y_scaled, test_size=0.2, random_state=42
    )
    

    if activate_grid_search:
        logging.info("Searching best parameters with grid search CV...")
        grid_search = gridSearch(training_pipe)
        grid_search.fit(raw_X, y_scaled)

        logging.info("Loading best parameteres to src/params.json")
        with open("src/params.json", "w") as file:
            json.dump(grid_search.best_params_, file, indent=4)

        logging.info(f"Found best parameters are {grid_search.best_params_}")

    
    logging.info("Fitting raw datat to training pipeline ...")
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
        training_pipe.fit(raw_X, y_scaled)
        logging.info(f"Returning training pipe and validation dictionary {validation_dict} with splitted data ...")
        return training_pipe, validation_dict, X_train, X_test, y_train, y_test

    
    training_pipe.fit(raw_X, y_scaled)
    logging.info(f"Returning training pipe and validation dictionary {validation_dict} ...")
    return training_pipe, validation_dict


if __name__ == "__main__":
    train_model(activate_grid_search=True)


