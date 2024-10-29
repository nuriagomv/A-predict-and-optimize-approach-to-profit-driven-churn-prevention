from imblearn.over_sampling import SMOTE #https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



def ejecutar_otros_clasificadores(X,y, train_idx,test_idx):

    X_train, y_train = X[train_idx,:], y[train_idx]
    X_test, y_test = X[test_idx,:], y[test_idx]

    oversample = SMOTE()
    X_train_balanced, y_train_balanced = oversample.fit_resample(X_train, y_train)
    counter = Counter(y_train_balanced)
    print('Oversampling con SMOTE: ', counter)

    # regresion logistica
    modelos = {'knn': KNeighborsClassifier(n_neighbors=39),
               'logistic': LogisticRegression(random_state=42),
               'rf': RandomForestClassifier(n_estimators = 500, max_features = 2, random_state = 42),
               'cart': DecisionTreeClassifier(ccp_alpha=0.0045),
               'svm': SVC(C=5 , gamma=0.001, probability  = True)}
    predicciones = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train_balanced, y_train_balanced)
        predicciones[nombre] = (modelo.predict(X_test), modelo.predict_proba(X_test)[:,0]) #prob y==0 fuga
                                #par (clase, prob y==0 fuga)
    return predicciones
