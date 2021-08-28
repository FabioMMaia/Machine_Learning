

from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.exceptions import NotFittedError

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def calculate_metrics(y_true, y_pred, verbose, mode):

    metrics = {}

    if (mode=='regression'):
        # Regression metrics

        metrics['explained_variance']=explained_variance_score(y_true, y_pred)
        metrics['mslogerror']=mean_squared_log_error(y_true, y_pred)
        metrics['mae']=mean_absolute_error(y_true, y_pred) 
        metrics['mse']=mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['median_abs_error']=median_absolute_error(y_true, y_pred)
        metrics['r2']=r2_score(y_true, y_pred)

        if (verbose):

            print('explained_variance: ', round(metrics['explained_variance'],4))    
            print('mean_squared_log_error: ', round(metrics['mslogerror'],4))
            print('MAE: ', round(metrics['mae'],4))
            print('MSE: ', round(metrics['mse'],4))
            print('RMSE: ', round(metrics['rmse'],4))
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

def plot_decision_function(classifier, X,y, proba=False, figsize=(20,10)):

    x1_axis,x2_axis = X.columns[0], X.columns[1]
    target=y.name

    fig, axis = plt.subplots(1,1, figsize=figsize)

    X1, X2 = np.meshgrid(np.arange(start= X[x1_axis].min() -1, stop= X[x1_axis].max() +1, step = 0.01),
                         np.arange(start= X[x2_axis].min()-1, stop= X[x2_axis].max() +1, step = 0.01))


    if (proba):
        predictions = classifier.predict_proba(np.array([X1.ravel(), X2.ravel()]).T)[:,1].reshape(X1.shape)
    else:
        predictions = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

    # plota 
    all = pd.concat([X, y], axis=1)
    all.plot(x = x1_axis, y =x2_axis, c=target, kind = 'scatter', cmap = 'Spectral', s=100, alpha =0.6, ax=axis)
    axis.contourf(X1,X2, predictions, cmap = 'Spectral', alpha = 0.2)
    axis.set_title('Superfície de Decisão')
    plt.show()
    
def plotta_cms(classifier, X_train, X_test, y_train, y_test, labels):

    fig, ax = plt.subplots(1,2,figsize=(12,12))

    plot_confusion_matrix(classifier, X_test, y_test, 
                        display_labels=labels, 
                        cmap =plt.cm.Blues,ax = ax[0],colorbar=False);
    ax[0].set_title('Teste');
    plot_confusion_matrix(classifier, X_train, y_train, 
                        display_labels=labels, 
                        cmap =plt.cm.Blues, ax= ax[1],colorbar=False);
    ax[1].set_title('Treino');
    plt.tight_layout()