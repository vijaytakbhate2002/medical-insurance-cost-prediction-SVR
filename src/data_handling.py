import pandas as pd
import logging 



def dataLoader(path:str) -> pd.DataFrame:
    """ 
        Args: 
            path: str (path from where you want to load data)
        Return: 
            pd.DataFrame (return the data which is present on path)

        Description: Read data from given pand and return as dataframe
        """
    
    logging.info(f"Reading data from {path}")
    if '.csv' in path:
        return pd.read_csv(path)
    elif '.xlsx' in path:
        return pd.read_excel(path)
    else:
        logging.error(path + " Does not exists")
        raise FileNotFoundError(path + " Does not exists")
    


def dataDumper(df:pd.DataFrame, path:str) -> None:
    """ 
        Args: 
            path: str (path to where you want to dump data)
            df: pd.DataFrame (the dataframe which you want to store)
        Return: 
            None

        Description: Store the data into given path
        """
    logging.info(f"Dumping data to {path}")
    if '.csv' in path:
        df.to_csv(path)
    elif '.xlsx' in path:
        df.to_excel(path)
    else:
        logging.error(path + " Does not exists")
        raise ValueError(path + " Does not exists")