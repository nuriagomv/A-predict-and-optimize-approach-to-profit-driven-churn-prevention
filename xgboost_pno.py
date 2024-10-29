# https://chatgpt.com/share/6716902f-39dc-800d-b4be-b43aa7033bb2

import xgboost as xgb
from sklearn.model_selection import ParameterGrid, KFold
import numpy as np
import pandas as pd
from functools import partial
import torch
import string
from f_aux import smooth_loss

def loss_xgb_pno(y_pred, dtrain, aux_params):
        
        (CLV_train, f, d, gamma) = aux_params
        y_true = dtrain.get_label()

        y_true, y_pred = torch.tensor(y_true), torch.tensor(y_pred, requires_grad=True)

        _, _, L = smooth_loss(y_true, y_pred, 10, torch.tensor(CLV_train), f, d, gamma)
        L = L.mean()
        L.backward()
        #L.backward(torch.ones_like(L))

        grad = y_pred.grad.detach().numpy()
        #print("grad: ",grad)
        #print("grad shape: ", grad.shape)
        # https://www.geeksforgeeks.org/how-to-compute-the-hessian-in-pytorch/
        #hess = torch.autograd.functional.hessian(smooth_loss, (y_true, y_pred, torch.tensor(10), torch.tensor(CLV_train), torch.tensor(f), torch.tensor(d), torch.tensor(gamma)))
        hess = torch.autograd.functional.hessian(lambda x: (smooth_loss(y_true, x, 10, torch.tensor(CLV_train), f, d, gamma)[2]).mean(),
                                                 y_pred)
        hess = np.diag(hess.detach().numpy())
        #print("hess: ", hess)
        #print("hess shape: ", hess.shape)
        
        return grad, hess


def eval_xgb_pno(y_pred, dtrain, aux_params):

    (CLV_train, f, d, gamma) = aux_params
    y_true = dtrain.get_label()

    y_true, y_pred = torch.tensor(y_true), torch.tensor(y_pred, requires_grad=True)

    _, _, L = smooth_loss(y_true, y_pred, 10, torch.tensor(CLV_train), f, d, gamma)
    L = L.detach().numpy().mean()
    
    return 'custom_error', L, False  # False means 'lower is better'


def ejecutar_xgboost(X, y, CLV, train_valid_idx, test_idx, f, d, gamma):

    X_train = X[train_valid_idx,:]
    y_train = y[train_valid_idx]
    CLV_train = CLV[train_valid_idx]
    
    # XGBoost parameters
    # Define a parameter grid
    param_grid = {
        'eta': [0.01, 0.1, 0.2],  # Learning rate
        'max_depth': [3, 6]
    }

    # Define number of boosting rounds and cross-validation folds
    num_boost_round = 100
    
    # Initialize KFold
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store results
    results = []

    # Perform grid search
    for params in ParameterGrid(param_grid):
        
        # Store scores for this parameter set
        fold_results = []

        for train_index, val_index in kf.split(X_train):
            # Get the training and validation data
            X_train_fold, X_val_fold = X_train[train_index,:], X_train[val_index,:]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            CLV_train_fold = CLV_train[train_index]
            CLV_valid_fold = CLV_train[val_index]

            aux_params = (CLV_train_fold, f, d, gamma)
            custom_obj = partial(loss_xgb_pno, aux_params=aux_params)
            custom_eval = partial(eval_xgb_pno, aux_params=aux_params)

            # Create DMatrix for train and validation
            dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dval_fold = xgb.DMatrix(X_val_fold, label=y_val_fold)

            # Train the model
            bst = xgb.train(params,
                            dtrain_fold,
                            num_boost_round=num_boost_round,
                            obj=custom_obj,
                            custom_metric=custom_eval
                            )

            # Make predictions on the validation set
            preds = bst.predict(dval_fold)
            aux_params_vl = (CLV_valid_fold, f, d, gamma)
            custom_eval = partial(eval_xgb_pno, aux_params=aux_params_vl)
            # Evaluate the custom metric
            metric_value = custom_eval(preds, dval_fold)[1]
            fold_results.append(metric_value)
            
        # Store mean score for this parameter set
        params['mean_score'] = np.mean(fold_results) #añado aqui el resultado para que se vea mejor
        results.append( params )

    results_df = pd.DataFrame(results).astype(dict([(k,type(v).__name__) for (k,v) in params.items()]))
    print("RESULTADOS CV XGBOOST: ")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
        print(results_df)
    best_params = results_df[results_df['mean_score'] == results_df['mean_score'].min()].iloc[:,:-1] 
    best_params = best_params.to_dict()
    best_params = dict( [(k,list(v.values())[0]) for (k,v) in best_params.items()] )
    #best_params = dict([(c,best_params.loc[0,c]) for c in best_params.columns])
    print("BEST PARAMS XGBOOST: ", best_params)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    aux_params = (CLV_train, f, d, gamma)
    custom_obj = partial(loss_xgb_pno, aux_params=aux_params)
    custom_eval = partial(eval_xgb_pno, aux_params=aux_params)
    bst = xgb.train(best_params,
                    dtrain, 
                    num_boost_round=num_boost_round,
                    obj=custom_obj,
                    custom_metric=custom_eval)
    
    X_test = X[test_idx,:]
    dtest = xgb.DMatrix(X_test)
    
    y_pred = bst.predict(dtest)

    return y_pred
