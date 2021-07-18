

from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score, log_loss

from sklearn.exceptions import NotFittedError

import numpy as np


def calculate_metrics(y_true, y_pred, verbose, mode):

    metrics = {}

    if (mode=='regression'):
        # Regression metrics

        metrics['explained_variance']=explained_variance_score(y_true, y_pred)
        metrics['mslogerror']=mean_squared_log_error(y_true, y_pred)
        metrics['mae']=mean_absolute_error(y_true, y_pred) 
        metrics['mse']=mean_squared_error(y_true, y_pred)
        metrics['rmse'] = round(np.sqrt(metrics['mse']),4)
        metrics['median_abs_error']=median_absolute_error(y_true, y_pred)
        metrics['r2']=r2_score(y_true, y_pred)

        if (verbose):

            print('explained_variance: ', round(metrics['explained_variance'],4))    
            print('mean_squared_log_error: ', round(metrics['mslogerror'],4))
            print('MAE: ', round(metrics['mae'],4))
            print('MSE: ', round(metrics['mse'],4))
            print('RMSE: ', round(np.sqrt(metrics['rmse']),4))
            print('Median Absolute Error: ', round(metrics['median_abs_error'],4))
            print('r2: ', round(metrics['r2'],4))

    if (mode=='classification'):
        # classification metrics 

        metrics['accuracy']=accuracy_score(y_true, y_pred > 0.5)
        metrics['precision']=precision_score(y_true, y_pred > 0.5)
        metrics['recall']=recall_score(y_true, y_pred > 0.5) 
        metrics['f1']=f1_score(y_true, y_pred > 0.5)
        metrics['log_loss'] = log_loss(y_true, y_pred > 0.5)
        metrics['roc_auc']=roc_auc_score(y_true, y_pred)
        
        if (verbose):

            print('accuracy: ', round(metrics['accuracy'],4))    
            print('precision: ', round(metrics['precision'],4))
            print('recall: ', round(metrics['recall'],4))
            print('f1: ', round(metrics['f1'],4))
            print('log_loss: ', round(metrics['log_loss'],4))
            print('roc_auc: ', round(metrics['roc_auc'],4))


    return metrics



def metrics(model, mode, X_train, X_test, y_train, y_test,verbose=True):

    try:
        model.predict(X_train)
    except NotFittedError as e:
        print('O modelo ainda não havia sido fittado ao conjunto de treino!')
        model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    if (mode=='classification'):    
        y_pred_train = model.predict_proba(X_train)[:,-1]
        y_pred_test = model.predict_proba(X_test)[:,-1]
    
    metrics= {}

    if verbose:
        print('\n### TREINO:###\n')
    metrics['train'] = calculate_metrics(y_train, y_pred_train,verbose,mode)

    if verbose:
        print('\n### TESTE:###\n')
    metrics['test'] = calculate_metrics(y_test, y_pred_test,verbose, mode)

    return metrics

