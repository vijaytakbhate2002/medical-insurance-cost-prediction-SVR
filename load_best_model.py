import mlflow
from mlflow import MlflowClient
from src.config import MLFLOW_URI
import logging
import mlflow.pyfunc
import os
import joblib
from src.config import BEST_MODEL_FOLDER


# os.environ['MLFLOW_TRACKING_USERNAME'] = 'YOUR USERNAME'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'YOUR PASSWORD'

# os.environ["AWS_ACCESS_KEY_ID"] = "YOUR AWS ACCESS KEY ID"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR AWS ACCESS KEY"
# os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


mlflow.set_tracking_uri(MLFLOW_URI)

MODEL_NAME = "InsuranceCostPredictor"
MODEL_VERSION = 13
MODEL_URI_VERSION = f"models:/{MODEL_NAME}/{MODEL_VERSION}" 

try:
    logging.info(f"Attempting to load model from URI: {MODEL_URI_VERSION}")
    print(f"Attempting to load model from URI: {MODEL_URI_VERSION}")
    loaded_model = mlflow.pyfunc.load_model(MODEL_URI_VERSION)
    joblib.dump(loaded_model, os.path.join(BEST_MODEL_FOLDER, MODEL_NAME + "_" + str(MODEL_VERSION) + ".pkl"))
    logging.info(f"✅ Model Version {MODEL_VERSION} loaded successfully!")
    
except Exception as e:
    logging.error(f"❌ An error occurred while loading Version {MODEL_VERSION}: {e}")