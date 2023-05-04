
from sklearn.pipeline import Pipeline
from src.load_data import Load_Data,path
from pipelines.features import MinMaxScale
from config.core import config 

train_df,_ = Load_Data(Data_path = path).retrive_data()
columns = [columns for columns in train_df.columns if columns not in config['targetvariable'] ] 

Pipe = Pipeline([

    ('minmaxscaler',MinMaxScale(variables = columns),)

                ])

