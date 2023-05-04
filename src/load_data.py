import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from  dataclasses import dataclass
import sys

path = os.path.join(os.getcwd(),'Data','diabetes.csv')

@dataclass
class Split_Data_Path:
    train_dataset_path = os.path.join('Data','train_dataset','train_df.csv')
    test_dataset_path = os.path.join('Data','test_dataset','test_df.csv')



class Load_Data:

    def __init__(self, Data_path):
        self.Data_path = Data_path
        self.split_data_path = Split_Data_Path()

    def retrive_data(self):
        df = pd.read_csv(self.Data_path)
        df_train,df_test = train_test_split(df,test_size = 0.2,random_state = 12)

        os.makedirs(os.path.dirname(self.split_data_path.train_dataset_path),exist_ok=True)
        df_train.to_csv(self.split_data_path.train_dataset_path,index= False,header = True)

        
        os.makedirs(os.path.dirname(self.split_data_path.test_dataset_path),exist_ok=True)

        df_test.to_csv(self.split_data_path.test_dataset_path,index= False,header = True)

        return df_train,df_test
    

if __name__ == "__main__":
    ld = Load_Data(Data_path = path)
    ld.retrive_data()



