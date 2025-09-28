import os


WORKDIR_PATH = ""


RAW_DATA_PATH = "data/raw_data/insurance.csv"
PROCESSED_X_PATH = "data/processed_data/processed_X.csv"
PROCESSED_Y_PATH = "data/processed_data/processed_Y.csv"
CWD = os.getcwd()



CAT_COLS = ['sex', 'smoker', 'region']
NUM_COLS = ['age', 'bmi', 'children']
TARGET_COLS = ['charges']
TARGET_SCALER_PATH = "data/target_scaling.pkl"

MLFLOW_URI = "http://ec2-52-90-255-217.compute-1.amazonaws.com:5000/"

BEST_MODEL_FOLDER = "best_performing_model/"



