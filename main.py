import os
import torch
import numpy as np
import random
import pandas as pd
#import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split

from leer_datos import read_base_CLVi#, read_base_sinCLV, split
from funcion_principal import entrenar_y_validar
from ejecutar_proflogit import ejecutar_proflogit
from otros_modelos import ejecutar_otros_clasificadores
from codigoGonzalo import ejecutar_segmentacion

ruta_datos = r'C:\Users\Nuria\OneDrive - UNIVERSIDAD DE SEVILLA\Académico\estancia Uchile\trabajo con Sebastián y Carla\bases fuga\bases con clv individual\3 meses'
os.chdir(ruta_datos)
#file = 'database_3m__2017-11-01.xlsx'

for file in os.listdir():

    print('###########################################################')
    print('\n\n\n DATOS DEL ARCHIVO: '+file+'\n\n\n')
    X, y, CLV, cambio_pesos = read_base_CLVi(ruta_datos, file)

    CLV_promedio = round(float(np.mean(CLV)), 2)
    lista_d = [round(CLV_promedio/n, 2) for n in [20, 15, 10, 5, 3]]
    # lista_d.insert(0,0) # LO QUITO PORQUE EL MODELO DE PROFLOGIT NO FUNCIONA CON d=0
    f = round(1000*cambio_pesos, 2)
    alpha, beta = 6, 14
    gamma = alpha/(alpha+beta)
    #gamma = np.random.beta(a=alpha, b=beta)

    """
    file = 'Telecom2__Duke1.mat'
    X, y, CLV = read_base_sinCLV(file)
    f, d, gamma = 1, round(float(np.mean(CLV))/20 , 2), 0.3
    """

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
        X = X[ f + gamma*(d-CLV) <0 ,:]
        y = y[ f + gamma*(d-CLV) <0 ]
        CLV = CLV[ f + gamma*(d-CLV) <0 ]
        print('dimensión de los datos: X->', X.shape, ', y->', y.shape, ', CLV->', CLV.shape)
        print('balanceo de datos: ', round(sum(y)/len(y)*100,2), '% es no fuga')

        random.seed(1234)
        np.random.seed(1234)
        train_valid_idx, test_idx = train_test_split(np.arange(len(y)),
                                                     test_size=0.2 , shuffle=True,
                                                     stratify=y) #each set contains approximately the same percentage of samples of each target class as the complete set
    
        ########################################################################################
        # ENTRENO CADA MODELO Y SACO PREDICCIÓN SOBRE MUESTRA TEST
        X_test, y_test, CLV_test = torch.Tensor(X[test_idx,:]), torch.Tensor(y[test_idx]), torch.Tensor(CLV[test_idx])

        predicciones = {}
    
        # MI MODELO PREDICT AND OPTIMIZE PARA CHURN PREDICTION
        predicciones['PNO'] = entrenar_y_validar(X,y,CLV, f,d,gamma, train_valid_idx, test_idx)

        #PROFLOGIT
        #VER CÓMO DOY UNA DECISION EN FUNCIÓN DEL ESCORE
        predicciones['proflogit'] = 1-ejecutar_proflogit(X,y,CLV, train_valid_idx, test_idx, f,d,gamma, CLV_promedio, case_label=0)
    
        # otros clasificadores estándar
        y_preds = ejecutar_otros_clasificadores(X,y, train_valid_idx,test_idx)
        for k,(v,_) in y_preds.items():
            predicciones[k] = 1-v # LA DECISIÓN ÓPTIMA EN FUNCIÓN DE LA PREDICCIÓN

        #segmentación Gonzalo
        segmentos = 2
        y_preds_segmentacion = ejecutar_segmentacion(y_preds, CLV_promedio, y,CLV, alpha,beta,d,f, test_idx, case_label=0, segmentos = segmentos)
        for k,v in y_preds_segmentacion.items():
            predicciones[k] = np.array(1-v,dtype=float)

        # MUESTRO RESULTADOS:
        profits = []
        accuracies = []
        for (nombre,z_pred) in predicciones.items():

            print('\n --------------------------\nTESTEANDO MODELO: ', nombre)
        
            z_pred = torch.Tensor(z_pred)
            z_opt = 1 - y_test

            # IMPORTANTE: LAS MEDIDAS NO SON SOBRE LAS PREDICCIONES ("NO ME IMPORTAN"), SINO SOBRE LAS DECISIONES 0/1
            cm_test = metrics.confusion_matrix(z_opt.detach().numpy(), z_pred.detach().numpy(), labels=[1,0])
            porcentaje_acierto = sum(z_opt == z_pred)/len(z_opt)*100
            print('Confusion matrix of decisions z in test set: \n', cm_test)
            print('Percentage of correct decisions in test set: ', np.round(float(porcentaje_acierto), 2), '%.')
            accuracies.append( np.round(float(porcentaje_acierto), 2) )
            profit_test_individuales = torch.mul(z_pred, f + y_test*d + (1-y_test)*gamma*(d-CLV_test)).detach().numpy()
            profit_optimum_test_individuales = torch.mul(z_opt, f + y_test*d + (1-y_test)*gamma*(d-CLV_test)).detach().numpy()
            profit_test_individuales = np.round(profit_test_individuales,2)
            profit_test = profit_test_individuales.sum()
            print('Profit in test sample: ', -round(profit_test,2), '€')
            profits.append( -round(profit_test,2) )
            print('Would be optimum profit in test sample: ', -round( profit_optimum_test_individuales.sum(),2), '€')
            print('With an incentive of d=',d, ', we obtained the ', round(profit_test/profit_optimum_test_individuales.sum()*100, 2), '% of what we could have earned.')


        indice_ganador_profit = [i for i in range(len(profits)) if profits[i]==max(profits)]#profits.index(max(profits))
        indice_ganador_accuracy = [i for i in range(len(accuracies)) if accuracies[i]==max(accuracies)]
        print('CON LA BASE DE DATOS '+file+' Y EL INCENTIVO d='+str(d)+\
            ', \nEL MODELO GANADOR SEGUN PROFIT ES: '+str([list(predicciones.keys())[i] for i in indice_ganador_profit])+\
            '\n Y EL MODELO GANADOR SEGUN ACCURACY ES: '+str([list(predicciones.keys())[i] for i in indice_ganador_accuracy]))

        