#https://github.com/estripling/proflogit/tree/master

import numpy as np
import pandas as pd
import string
from proflogit.proflogit.base import ProfLogit
#from proflogit.proflogit.utils import load_data


def ejecutar_proflogit(X,y,CLV, train_idx, test_idx, f,d,gamma, CLV_promedio, case_label=0):

    X_train = {}
    for i in range(X.shape[1]):
        X_train[i] = X[train_idx,i]

    df = pd.DataFrame(X_train)
    df['y'] = y[train_idx]

    X_train = df.iloc[:,0:X.shape[1]]
    X_train.columns = list(string.ascii_lowercase)[:X.shape[1]]
    y_train = df.loc[:,'y']

    pfl = ProfLogit(rga_kws={'niter': 50, 'disp': True, 'random_state': 42,},
                    empc_kws={'f': f, 'd': d, 'clv': CLV_promedio, 'case_label': case_label},)
    pfl.fit(X_train, y_train)

    """
    #plot
    import matplotlib.pyplot as plt
    best_so_far_solutions = pfl.rga.fx_best
    plt.plot(best_so_far_solutions, label='RGA')
    plt.legend()
    plt.show()
    """

    #test
    X_test = {}
    for i in range(X.shape[1]):
        X_test[i] = X[test_idx,i]

    df = pd.DataFrame(X_test)
    
    X_test = df.iloc[:,0:X.shape[1]]
    X_test.columns = list(string.ascii_lowercase)[:X.shape[1]]
    df['y'] = y[test_idx]
    y_test = df.loc[:,'y']

    y_score = pfl.predict_proba(X_test)

    opt_threshold, mpc, mpcFrac, empc = pfl.score(X_test,y_test)
    y_pred_label = np.array((y_score<opt_threshold), dtype=float)
    #z_pred = 1-y_pred_label
    #from math import isclose
    #isclose( np.sum(z_pred)/len(z_pred) , mpcFrac )
            

    return y_pred_label

"""
CLV_test = torch.Tensor(CLV[test_idx])
m = ( f + gamma*(d-CLV_test) )/( gamma*(d-CLV_test) - d )
z_pred = 1-torch.relu( torch.sign(torch.Tensor(y_score) - m) )
z_opt = 1-torch.Tensor(y_test)
cm_test =metrics.confusion_matrix(z_opt.detach().numpy(), z_pred.detach().numpy(), labels=[1,0])
porcentaje_acierto = sum(z_opt == z_pred)/len(z_opt)*100
print('Confusion matrix of decisions z in test set: \n', cm_test)
print('Percentage of correct decisions in test set: ', np.round(float(porcentaje_acierto), 2), '%.')
profit_test_individuales = torch.mul(z_pred, f + torch.Tensor(y_test)*d + (1-torch.Tensor(y_test))*gamma*(d-CLV_test)).detach().numpy()
profit_optimum_test_individuales = torch.mul(z_opt, f + torch.Tensor(y_test)*d + (1-torch.Tensor(y_test))*gamma*(d-CLV_test)).detach().numpy()
profit_test_individuales = np.round(profit_test_individuales,2)
profit_test = profit_test_individuales.sum()
print('Profit in test sample: ', -round(profit_test,2), '€')
print('Would be optimum profit in test sample: ', -round( profit_optimum_test_individuales.sum(),2), '€')
print('With an incentive of d=',d, ', we obtained the ', round(profit_test/profit_optimum_test_individuales.sum()*100, 2), '% of what we could have earned.')



#Compute EMPC performance on test set
empc = pfl.score(X_test, y_test)
empc
"""