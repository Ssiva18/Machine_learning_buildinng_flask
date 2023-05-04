from src.load_data import Load_Data,path
from sklearn.model_selection import train_test_split
from config.core import config
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from pipelines.pipeline import Pipe
from utils import evaluate_model
import pickle
import os

class Split_Data: 

    def __init__(self):
        self.train_df,self.test_df = Load_Data(Data_path = path).retrive_data()
        self.target = config['targetvariable']

    def splitting_data(self):

        input_feature_train_df = self.train_df.drop(config['targetvariable'],axis =1)
        target_feature_train_df = self.train_df[config['targetvariable']]

        input_feature_test_df = self.test_df.drop(config['targetvariable'],axis =1)
        target_feature_test_df = self.test_df[config['targetvariable']]


        return input_feature_train_df,input_feature_test_df,target_feature_train_df,target_feature_test_df

    def train_model(self):

        x_train,x_test,y_train,y_test = self.splitting_data()

        self.models = {

                  'decisiontree':DecisionTreeClassifier(),
                  'adaboostclassifier':AdaBoostClassifier(),
                  'gradientboostingclassifier':GradientBoostingClassifier(),
                  'logisticregression':LogisticRegression(),
                  'randomforest':RandomForestClassifier()

                  }
        
        self.params={

                "decisiontree": {
                    'criterion':['entropy', 'log_loss', 'gini'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },

                "randomforest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "gradientboostingclassifier":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "logisticregression":{},
                
                
                "adaboostclassifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
        x_train = Pipe.fit_transform(x_train)
        x_test = Pipe.transform(x_test)

        self.report,self.mod = evaluate_model(x_train=x_train,x_test=x_test,y_test=y_test,
                                y_train=y_train,models=self.models,params=self.params)
    
        return self.report,self.mod
        

    def Best_performance_of_model(self):
        
        self.report,self.mod = self.train_model()
        accuracies = []
        for i in self.report.keys():
            accuracies.append(self.report[i]['accuracy_score'])
        
        max_value = max(accuracies)
        for i in range(len(accuracies)):
            model_name = list(self.report.keys())[i]
            if max_value == self.report[model_name]['accuracy_score']:
                 self.model_name = model_name

                 return self.model_name
    
    def Save_Model(self):
         
         best_model = self.Best_performance_of_model()
         dump_model = self.mod[best_model]


         with open(os.path.join('save_models','model.pkl'),'wb') as files:
             pickle.dump(dump_model,files)

 

if __name__ == "__main__":
    sd = Split_Data()
    print(sd.Save_Model())
 



