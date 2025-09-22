import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from src.data_handling import dataDumper, dataLoader
from src.data_processing import pipeline
from src.config import CAT_COLS, NUM_COLS, TARGET_COLS, RAW_DATA_PATH, PROCESSED_X_PATH, PROCESSED_Y_PATH

df = dataLoader(path=RAW_DATA_PATH)

X = pipeline.fit_transform(df.drop(TARGET_COLS, axis='columns'), df[TARGET_COLS])
processed_cols = NUM_COLS + CAT_COLS 
X = pd.DataFrame(X, columns=processed_cols)


target_scaling = MinMaxScaler()
y = target_scaling.fit_transform(df[TARGET_COLS]).reshape(1, -1)[0]
y = pd.Series(y)


dataDumper(X, PROCESSED_X_PATH)
dataDumper(y, path=PROCESSED_Y_PATH)


