import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split

from leer_datos import read_base_CLVi#, read_base_sinCLV, split
from PNO_NNs import entrenar_y_validar
from xgboost_pno import ejecutar_xgboost
from ejecutar_proflogit import ejecutar_proflogit
from otros_modelos import ejecutar_otros_clasificadores
from codigoGonzalo import ejecutar_segmentacion
from f_aux import step_function, top10lift


exp = 'total_año'


def ejecutar_todo(entrada, exp):

    if exp == 'mes_a_mes':
        X, y, CLV, cambio_pesos, file = entrada
    if exp == 'total_año':
        datos_merged, cambio_pesos, file = entrada
        X = np.array(datos_merged.loc[:,[c for c in datos_merged.columns if 'X' in c]], dtype=float)
        y = np.array(datos_merged.loc[:,'y'], dtype=int)
        CLV = np.array(datos_merged.loc[:,'CLV'], dtype=float)

    CLV_promedio = round(float(np.mean(CLV)), 2)
    print('CLV_promedio:', CLV_promedio)
    print('CLV_std:', round(float(np.std(CLV)), 2) )

    lista_d = [round(CLV_promedio/n, 2) for n in [20, 15, 10, 5, 3]]
    # lista_d.insert(0,0) # LO QUITO PORQUE EL MODELO DE PROFLOGIT NO FUNCIONA CON d=0
    f = round(1000*cambio_pesos, 2)
    alpha, beta = 6, 14
    gamma = alpha/(alpha+beta) #gamma = np.random.beta(a=alpha, b=beta)
    """
    plt.plot(y,CLV,'o')
    plt.title('Distribución del CLV por categoría')
    plt.show()

    import seaborn as sns
    sns.boxplot(x='churn', y='CLV', data=pd.DataFrame({'CLV':CLV, 'churn':y}))
    plt.show()
    """

    for d in lista_d:

        print('\n\n----------------------------------------------- ')
        print('f, d, gamma, respectivamente: ', f, d, gamma)
        print('me interesa el CLV de todos los clientes? ', np.sum( f + gamma*(d-CLV) <0 ) == len(CLV))
        X, y, CLV = X[ f + gamma*(d-CLV) <0 ,:], y[ f + gamma*(d-CLV) <0 ], CLV[ f + gamma*(d-CLV) <0 ]
        print('dimensión final de los datos: X->', X.shape, ', y->', y.shape, ', CLV->', CLV.shape)
        print('balanceo de datos: ', round(sum(y)/len(y)*100,2), '% es no fuga')

        random.seed(1234)
        np.random.seed(1234)
        train_valid_idx, test_idx = train_test_split(np.arange(len(y)),
                                                     test_size=0.2, shuffle=True,
                                                     stratify=y) #each set contains approximately the same percentage of samples of each target class as the complete set

        ########################################################################################
        # ENTRENO CADA MODELO Y SACO PREDICCIÓN SOBRE MUESTRA TEST
        y_test, CLV_test = torch.Tensor(y[test_idx]), torch.Tensor(CLV[test_idx])
        z_opt = 1 - y_test
        profit_optimum_test_individuales = torch.mul(z_opt, f + y_test*d + (1-y_test)*gamma*(d-CLV_test)).detach().numpy()
        
        predicciones = {}
    
        #PNO
        t_i = ( f + gamma*(d-CLV_test) )/( gamma*(d-CLV_test) - d )
        #PNO_XGBOOST
        predicciones['PNO_xgboost'] = {}
        predicciones['PNO_xgboost']['y_pred'] = torch.tensor(ejecutar_xgboost(X,y,CLV, train_valid_idx, test_idx, f,d,gamma))
        
        #PNO_NN, PNO_logistic
        for ELEGIR_MODELO in ['NN', 'logistic']:
            predicciones['PNO_'+ELEGIR_MODELO] = {}
            predicciones['PNO_'+ELEGIR_MODELO]['y_pred'] = entrenar_y_validar(ELEGIR_MODELO,
                                                                              X,y,CLV,f,d,gamma,train_valid_idx,test_idx)
        for pno_model in predicciones.keys():
            predicciones[pno_model]['z_pred'] = step_function(predicciones[pno_model]['y_pred'] - t_i)
        
        #PROFLOGIT
        predicciones['proflogit'] = {}
        predicciones['proflogit']['y_pred'] = torch.tensor(ejecutar_proflogit(X,y,CLV, train_valid_idx, test_idx, f,d,gamma, CLV_promedio, case_label=0))
        predicciones['proflogit']['z_pred'] = 1-predicciones['proflogit']['y_pred']

        # otros clasificadores estándar
        y_preds = ejecutar_otros_clasificadores(X,y, train_valid_idx,test_idx)
        for k,(v,_) in y_preds.items():
            predicciones[k] = {}
            predicciones[k]['y_pred'] = torch.tensor(v)
            predicciones[k]['z_pred'] = 1-predicciones[k]['y_pred']

        #segmentación Gonzalo
        y_preds_segmentacion = ejecutar_segmentacion(y_preds, CLV_promedio, y,CLV, alpha,beta,d,f, test_idx, case_label=0,
                                                     segmentos = 2)
        for k,v in y_preds_segmentacion.items():
            predicciones[k] = {}
            predicciones[k]['y_pred'] = torch.tensor(v.astype(float))
            predicciones[k]['z_pred'] = 1-predicciones[k]['y_pred']


        ########################################################################################
        # MUESTRO RESULTADOS:
        profits = []
        accuracies = []
        AUCs = []
        top10lifts = []
        for (nombre,dict) in predicciones.items():

            print('\n --------------------------\nTESTEANDO MODELO: ', nombre)
            y_pred, z_pred = dict.values()
            
            # IMPORTANTE: LAS MEDIDAS NO SON SOBRE LAS PREDICCIONES ("NO ME IMPORTAN"), SINO SOBRE LAS DECISIONES 0/1
            cm_test = metrics.confusion_matrix(z_opt.detach().numpy(), z_pred.detach().numpy(), labels=[1,0])
            acc_test = sum(z_opt == z_pred).item()/len(z_opt)*100
            print('Confusion matrix (using decision z_pred): \n', cm_test)
            print('Accuracy (using decision z_pred): ', acc_test, '%.')
            accuracies.append( acc_test )
            
            AUC_test = metrics.roc_auc_score(y_test.detach().numpy(), y_pred.detach().numpy())
            # Passing binary class predictions (e.g., from model.predict(X_test)) would not give the correct ROC AUC score, as it would no longer be threshold-independent.
            #AUC_test = metrics.roc_auc_score(z_opt.detach().numpy(), z_pred.detach().numpy())
            AUCs.append(AUC_test)
            print("AUC(using scores y_pred): ", AUC_test)

            print('Nominal optimum profit in test sample: ', -profit_optimum_test_individuales.sum(), '€')
            profit_test_individuales = torch.mul(z_pred, f + y_test*d + (1-y_test)*gamma*(d-CLV_test)).detach().numpy()
            profit_test = -profit_test_individuales.sum()
            print('Profit in test sample: ', profit_test, '€')
            profits.append( profit_test )
            print('With an incentive of d=',d, ', we obtained the ', round(profit_test/profit_optimum_test_individuales.sum()*100, 2), '% of what we could have earned.')

            lift = top10lift(y_test.detach().numpy(), y_pred.detach().numpy())
            top10lifts.append(lift)
            print(f"Top Decile Lift: {lift:.2f}")
        

        indice_ganador_profit = [i for i in range(len(profits)) if profits[i]==max(profits)]
        indice_ganador_accuracy = [i for i in range(len(accuracies)) if accuracies[i]==max(accuracies)]
        indice_ganador_AUC = [i for i in range(len(AUCs)) if AUCs[i]==max(AUCs)]
        indice_ganador_lift = [i for i in range(len(top10lifts)) if top10lifts[i]==max(top10lifts)]
        print('CON LA BASE DE DATOS '+file+' Y EL INCENTIVO d='+str(d)+\
            ', \nEL MODELO GANADOR SEGUN PROFIT ES: '+str([list(predicciones.keys())[i] for i in indice_ganador_profit])+\
            ', \nEL MODELO GANADOR SEGUN ACCURACY ES: '+str([list(predicciones.keys())[i] for i in indice_ganador_accuracy])+\
            ', \nEL MODELO GANADOR SEGUN AUC ES: '+str([list(predicciones.keys())[i] for i in indice_ganador_AUC])+\
             ', \nEL MODELO GANADOR SEGUN top10lift ES: '+str([list(predicciones.keys())[i] for i in indice_ganador_lift])
             )


########################################################################################

#mac
#ruta_datos = r'/Users/nuriagomezvargas/Library/CloudStorage/OneDrive-UNIVERSIDADDESEVILLA/Académico/estancia Uchile/trabajo con Sebastián y Carla/churn/bases fuga/bases con clv individual/3 meses'
#windows
ruta_datos = r'C:\Users\Nuria\OneDrive - UNIVERSIDAD DE SEVILLA\Académico\estancia Uchile\trabajo con Sebastián y Carla\churn\bases fuga\bases con clv individual\3 meses'
os.chdir(ruta_datos)
#file = 'database_3m__2017-11-01.xlsx'

if exp=='total_año':
    inicio = True

for file in os.listdir():
    
    print('###########################################################')
    print('\n\n\n DATOS DEL ARCHIVO: '+file+'\n\n\n')
    X, y, CLV, cambio_pesos = read_base_CLVi(ruta_datos, file)

    if exp=='mes_a_mes':
        entrada = (X, y, CLV, cambio_pesos, file)
        ejecutar_todo(entrada, exp)

    if exp=='total_año':
        datos_i = pd.DataFrame(columns=['X_'+str(i+1) for i in range(X.shape[1])]+['y','CLV'])
        for i in range(X.shape[1]):
            datos_i['X_'+str(i+1)] = X[:,i] 
        datos_i['y'], datos_i['CLV'] = y, CLV

        if inicio:
            datos_merged = pd.DataFrame(columns=['X_'+str(i+1) for i in range(X.shape[1])]+['y','CLV'])
            inicio=False
        datos_merged = pd.concat([datos_merged, datos_i], axis=0)

if exp=='total_año':
    entrada = (datos_merged, cambio_pesos, 'todos_años')
    ejecutar_todo(entrada, exp)
