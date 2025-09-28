import mlflow 
from mlflow import pyfunc
from src.config import MLFLOW_URI
import os
from training_pipeline_runner import train_model
from processing_pipeline_runner import process_data
from mlflow.models import infer_signature
from urllib import parse
import logging
import boto3
import json


with open("src/params.json", 'r') as file:
    data = dict(json.load(file))
    data = {key.replace("SVR__", ''): value for key, value in data.items()}


mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("medical-insurance-cost-prediction-SVR")

with mlflow.start_run():

    trained_pipeline, metrics, X_train, X_test, y_train, y_test = train_model(activate_grid_search=True, return_data=True)

    mlflow.log_params(data)

    mlflow.log_metrics(metrics=metrics)

    signature = infer_signature(X_train, trained_pipeline.predict(X_train))

    tracking_uri_sheme = parse.urlparse(mlflow.get_tracking_uri()).scheme
    

    if tracking_uri_sheme != "file":
        logging.info("logging model to remote uri tracking mlflow ... ")
        model_info = mlflow.sklearn.log_model(
            sk_model=trained_pipeline,
            signature=signature, 
            name="momedical_insurance_cost_prediction_SVRdel",
            input_example=X_train,
            registered_model_name="InsuranceCostPredictor"
        )
        logging.info("logging successful model to remote uri tracking mlflow ... ")
    else:
        logging.info("logging model to local uri tracking mlflow ... ")
        model_info = mlflow.sklearn.log_model(
            sk_model=trained_pipeline,
            name="medical_insurance_cost_prediction_SVR",
            signature=signature
        )
        logging.info("logging successful model to local uri tracking mlflow ... ")

