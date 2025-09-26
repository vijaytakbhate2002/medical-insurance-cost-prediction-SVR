import mlflow 
from src.config import MLFLOW_URI
import boto3
import os

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
os.environ['MLFLOW_TRACKING_USERNAME'] = 'USERNAME'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'PASSWORD'

mlflow.set_tracking_uri(MLFLOW_URI)

with mlflow.start_run():
    mlflow.log_param("Testing", 0.9)

