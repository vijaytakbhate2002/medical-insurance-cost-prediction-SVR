from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import pandas as pd
import logging



class ModelValidation:
    """
        Args: model: sklearn.pipeline.Pipeline, 
                X_train, X_test, y_train, y_test
                
        Description: Model Validation helps for generating validation metrics for model
        """
    def __init__(self, model:Pipeline, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.Series, y_test:pd.Series) -> None:
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test



    def metrics(self):
        """ 
            Args: None
            Description: Metrics computation function, used to compute MSE, 
            R2_Score (train and test), accuracy (train and test)
            """
        logging.info(f"""Shape of data:  
                     X_train {self.X_train.shape}
                     X_test {self.X_test.shape}
                     y_train {self.y_train.shape} 
                     y_test {self.y_test.shape}"""
                     )
        
        y_pred = self.model.predict(self.X_test)
        y_train_pred = self.model.predict(self.X_train)

        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_pred)

        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = root_mean_squared_error(self.y_test, y_pred)

        return {
            "mse":mse, 
            "mae": mae,
            "rmse":rmse,
            "train_r2":train_r2,
            "test_r2":test_r2
        }

