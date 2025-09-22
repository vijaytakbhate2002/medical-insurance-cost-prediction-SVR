import pandas as pd



def dataLoader(path:str) -> pd.DataFrame:
    """ Read data from given pand and return as dataframe"""
    if '.csv' in path:
        return pd.read_csv(path)
    elif '.xlsx' in path:
        return pd.read_excel(path)
    else:
        raise FileNotFoundError(path + " Does not exists")
    


def dataDumper(df:pd.DataFrame, path:str) -> None:
    """ Saves the dataframe in given path"""
    if '.csv' in path:
        df.to_csv(path)
    elif '.xlsx' in path:
        df.to_excel(path)
    else:
        raise ValueError(path + " Does not exists")