import pandas as pd
import os
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as sio



def read_base_CLVi(ruta_datos, file):

    os.chdir(ruta_datos)
    datos = pd.read_excel(file)
    
    X = np.array(datos[['Frecuency_Aportes', 'Frecuency_Rescates', 'Monetary_Aportes',
                        'Monetary_Rescates', 'Recency_Aportes', 'Recency_Rescates',
                        'Perfil_Movimientos', 'Perfil_Cliente', 'Sexo', 'Estado_Civil',
                        'Ingeniero', 'Region_Metropolitana', 'Porcentaje_Promedio', 'Edad',
                        'Antiguedad', 'Accionario', 'Corto', 'Rentsem', 'Rentmen', 'EEUU',
                        'Nacional', 'FugaBajaS', 'FugaBajaM']])

    y = np.array(datos['FugaCategorica'])
    y[y=='F'], y[y=='N'] = 0, 1 #0 churner and 1 non-churner
    y = np.array(y, dtype = int)
    
    #el CLV está en pesos!! hay que convertir: 1€ = 732.743CLP en 2017 promedio https://es.investing.com/currencies/eur-clp-historical-data
    cambio_pesos = 1/732.743
    CLV = np.round( np.array(datos['CLV'])*cambio_pesos, 2)

    return X, y, CLV, cambio_pesos


def read_base_sinCLV(file):

    os.chdir(r'C:\Users\nuria\OneDrive - UNIVERSIDAD DE SEVILLA\Académico\estancia Uchile\trabajo con Sebastián y Carla\bases fuga\bases con clv estático')

    datos = sio.loadmat(file)

    datos.keys()
    X = datos['X']
    y = datos['Y'].flatten()
    # -1 (1 for me) non-churner and 1 (0 for me) churner
    y[y==1] = 0
    y[y==-1] = 1

    X.shape
    np.sum(y==1.)/len(y)

    np.random.seed(1)
    CLV = np.random.normal(loc = 200, scale = 1., size = len(y))

    return X, y, CLV


def split(X,y, seed, percentage_test = 0.2, percentage_valid = 0.2):

    random.seed(seed)
    np.random.seed(seed)

    train_idx, test_idx = train_test_split(np.arange(len(y)),
                                           test_size=percentage_test , shuffle=True,
                                           stratify=y) #each set contains approximately the same percentage of samples of each target class as the complete set
    train_idx, valid_idx = train_test_split(train_idx,
                                            test_size=percentage_valid, shuffle=True,
                                            stratify=y[train_idx])
    #np.sum(y)/len(y)
    #np.sum(y[train_idx])/len(train_idx)
    #np.sum(y[valid_idx])/len(valid_idx)

    return train_idx, valid_idx, test_idx

