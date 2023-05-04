from sklearn.base import BaseEstimator,TransformerMixin
from typing import List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

class MinMaxScale(BaseEstimator,TransformerMixin):

    def __init__(self,variables:List[str]):
        self.variables = variables

    def fit(self,X:pd.DataFrame) -> pd.DataFrame:

        return self
    
    def transform(self,X:pd.DataFrame)->pd.DataFrame:

        df = X.copy()
        for columns in self.variables:
            df[columns] = MinMaxScaler().fit_transform(df[[columns]])
        return df 
