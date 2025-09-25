import os


WORKDIR_PATH = ""


RAW_DATA_PATH = "data/raw_data/insurance.csv"
PROCESSED_X_PATH = "data/processed_data/processed_X.csv"
PROCESSED_Y_PATH = "data/processed_data/processed_Y.csv"
CWD = os.getcwd()



CAT_COLS = ['sex', 'smoker', 'region']
NUM_COLS = ['age', 'bmi', 'children']
TARGET_COLS = ['charges']



BEST_PARAMS = {'C': 10, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf'}



