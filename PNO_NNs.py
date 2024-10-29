import torch
import os
import numpy as np
import random
from torch.utils.data import DataLoader#,Dataset
#from sampler import BalancedBatchSampler
from joblib import Parallel, delayed
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE #https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
#from collections import Counter

from arquitecturas_NN import NN, NN_logistic
from f_aux import loss, smooth_loss


def ejecutar_para_CV(value):
    
    ELEGIR_MODELO, X,y,CLV, train_idx, valid_idx, hidden_size, batch_size, epochs, lr, seed, f, d, gamma, loss_flag = value

    #print('Ejecutando iteracion con batch_size=',batch_size, ' epochs=',epochs, ' lr=',lr, ' seed=',seed)

    # Genera una semilla fija para que los experimentos sea repetibles.
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    X_train, y_train,CLV_train = torch.Tensor(X[train_idx,:]), torch.Tensor(y[train_idx]), torch.Tensor(CLV[train_idx])
    """
    oversample = SMOTE()
    X_CLV_train, y_train = oversample.fit_resample(np.c_[X_train,CLV_train], y_train)
    X_train = torch.Tensor( X_CLV_train[:,:-1] )
    CLV_train = torch.Tensor( X_CLV_train[:,-1] )
    counter = Counter(y_train)
    print('balanceo SMOTE oversampling: ', counter)
    """
    m_train = ( f + gamma*(d-CLV_train) )/( gamma*(d-CLV_train) - d )
    

    #AQUI ES DONDE TENEMOS QUE DIFERENCIAR NN, LOGISTIC
    if ELEGIR_MODELO == 'NN':
        red = NN(X.shape[1], hidden_size)
    if ELEGIR_MODELO == 'logistic':
        red = NN_logistic(X.shape[1])
    optimizador = torch.optim.Adam(params = red.parameters(), lr = lr)
    red.train()

    train_loader = DataLoader(list(zip(X_train, y_train, CLV_train)),
                              shuffle=True, batch_size=batch_size)
    #print('\n ENTRENAMIENTO \n')
    for e in range(epochs):
        #print('EPOCH: ', e)
        
        for X_batch, y_batch, CLV_batch in train_loader:
            #print(y_batch)
            y_pred = red.forward(X_batch).flatten()
            if loss_flag == 'smooth_loss':
                z_pred, z_opt, L = smooth_loss(y_batch, y_pred, 10, CLV_batch, f, d, gamma)
            else:
                z_pred, z_opt, L = loss(y_batch, y_pred, CLV_batch, f, d, gamma)
            #print('z_opt: ', z_opt)
            #print('z_pred: ', z_pred)
            #print('Loss: ', L)
            L = L.mean()
            #print('Mean Loss: ', L)
            #red.only_hidden.weight
            #red.only_hidden.weight.grad
            try:
                L.backward()
                #print('gradiente: ', red.only_hidden.weight.grad)
                optimizador.step()
                optimizador.zero_grad()
            except Exception as e:
                print(e) #Function 'ExpBackward0' returned nan values in its 0th output.
                #print(red.only_hidden.weight.grad)
                break

    #print('\n VALIDACION \n')
    red.eval()
    X_valid, y_valid, CLV_valid = torch.Tensor(X[valid_idx,:]), torch.Tensor(y[valid_idx]), torch.Tensor(CLV[valid_idx])
    y_pred = red.forward(X_valid).flatten()
    if loss_flag == 'smooth_loss':
        z_pred_valid, z_opt_valid, L_valid = smooth_loss(y_valid, y_pred, 10, CLV_valid, f, d, gamma)
    else:
        z_pred_valid, z_opt_valid, L_valid = loss(y_valid, y_pred, CLV_valid, f, d, gamma)
    #print('z_pred: ', torch.round(z_pred))
    #print('z_opt: ', z_opt)
    #print('Loss: ', torch.round(L))
    #print('Mean Loss: ', torch.round(L.mean()))
    #print('Total Loss: ', float(torch.round(L_valid.sum())))

    """
    actual = z_opt.detach().numpy()
    predicted = torch.round(z_pred).detach().numpy()
    cm = metrics.confusion_matrix(actual, predicted, labels=[1,0])
    #print('Confusion matrix of decision z:\n ', cm)
    ac = metrics.accuracy_score(actual, predicted)
    #print('Accuracy: ', round(ac, 2))
    #print('Precision: ', round(metrics.precision_score(actual, predicted), 2))
    sens = metrics.recall_score(actual, predicted)
    #print('Sensitivity (true "target" rate): ', round(sens, 2))
    spec = metrics.recall_score(actual, predicted, pos_label=0)
    #print('Specificity (true "dont target" rate): ', round(spec, 2))
    #print('F1 score: ', round(metrics.f1_score(actual, predicted), 2))
    """

    return [(hidden_size, batch_size, epochs, lr, loss_flag), (red, z_pred_valid, z_opt_valid, L_valid)]


def entrenar_y_validar(ELEGIR_MODELO,
                       X,y,CLV, f,d,gamma, train_valid_idx, test_idx):

    num_cores = os.cpu_count()
    print('NUM CORES: ', num_cores)

    tune_epochs = [10, 50, 100]#, 500, 1000, 5000]
    tune_lr = [0.01, 0.001, 0.0001]
    tune_batch_size = [32]#, 16, 1]
    tune_hidden_size = [int((X.shape[1]+1)/2)]
    try_loss_flags = ['smooth_loss']#, 'loss']
    seeds = range(10)

    diccionario_validaciones = {}
    for seed in seeds: #Monte Carlo cross-validation (MCCV)
        diccionario_validaciones[seed] = {}
    
        # Genera una semilla fija para que los experimentos sea repetibles.
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)

        #train_idx, valid_idx, test_idx = split(X,y, seed)
        train_idx, valid_idx = train_test_split(train_valid_idx,
                                                test_size=0.2 , shuffle=True,
                                                stratify=y[train_valid_idx])
        if seed == 0:
            print('Tamaño muestras entrenamiento, validación y test, respectivamente: ', len(train_idx), len(valid_idx), len(test_idx))
    
        values = []
        for epochs in tune_epochs:
            for lr in tune_lr:
                for batch_size in tune_batch_size:
                    for hidden_size in tune_hidden_size:
                        for loss_flag in try_loss_flags:
                            values.append( (ELEGIR_MODELO,
                                            X,y,CLV, train_idx, valid_idx, hidden_size, batch_size, epochs, lr, seed, f, d, gamma, loss_flag) )

        results = Parallel(n_jobs=num_cores)(delayed(ejecutar_para_CV)(value) for value in values)
        for result in results:
            [hyperp, red_and_valid] = result
            diccionario_validaciones[seed][hyperp] = red_and_valid

    
    # SELECCION DE LA MEJOR ITERACION DE VALIDACION
    total_loss, mean_total_loss = {}, {}
    hyperparams = list(diccionario_validaciones[seed].keys())
    for h in hyperparams:
        total_loss[h] = [float(L_valid.sum()) for (_, _, _, L_valid) in [diccionario_validaciones[seed][h] for seed in diccionario_validaciones.keys()]]
        mean_total_loss[h] = np.mean(np.array(total_loss[h]))
        
    best_hyperparams = min(mean_total_loss, key=mean_total_loss.get)
    best_seed = (total_loss[best_hyperparams]).index(min(total_loss[best_hyperparams]))

    print('MEAN_TOTAL_LOSS_VALIDATION:', mean_total_loss)
    print('\n Best hyperparams: ', best_hyperparams, ' and best seed: ', best_seed)

    
    #SACO PREDICCIÓN TEST
    (red_best, _,_, _) = diccionario_validaciones[best_seed][best_hyperparams]
    print('\n \n TEST \n')
    red_best.eval()
    X_test = torch.Tensor(X[test_idx,:])
    
    #IMPORTANTE DIFERENCIA ENTRE y_pred/z_pred
    y_pred = red_best.forward(X_test).flatten()
    
    return y_pred