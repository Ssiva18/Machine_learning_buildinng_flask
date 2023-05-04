from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def evaluate_model(x_train,y_train,x_test,y_test,models,params):

    report = {}
    mod = {}
    for i in models.keys():
        model = models[i]
        param = params[i]
        gs = GridSearchCV(model,param,cv=2)
        gs.fit(x_train,y_train)

        model.set_params(**gs.best_params_)
        model.fit(x_train,y_train)
        mod[i] = model

        #model.fit(X_train, y_train)  # Train model

        y_test_pred = model.predict(x_test)

        train_model_accuracy_score = accuracy_score(y_test, y_test_pred)
        train_model_precision_score = precision_score(y_test, y_test_pred)
        train_model_recall_score = recall_score(y_test, y_test_pred)
        train_model_f1_score = f1_score(y_test, y_test_pred)


        report[i] = {
                           'train_model_f1_score' :train_model_f1_score,
                            'recall_score':train_model_recall_score,
                            'precision_score':train_model_precision_score,
                            'accuracy_score':train_model_accuracy_score


        }

    return report,mod

