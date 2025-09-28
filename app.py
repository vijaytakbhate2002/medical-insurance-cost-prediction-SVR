import joblib
from src import config
import pandas as pd
from src.config import TARGET_COLS, TARGET_SCALER_PATH

target_scaler = joblib.load(TARGET_SCALER_PATH)
model = joblib.load("best_performing_model\InsuranceCostPredictor_13.pkl")

df = pd.read_csv(config.RAW_DATA_PATH)
X = df.drop(TARGET_COLS, axis='columns')
y = df[TARGET_COLS]

y_pred = model.predict(X)

y_pred_org = target_scaler.inverse_transform(y_pred.reshape(1, -1))
print(y_pred_org[:10], y[:10])


