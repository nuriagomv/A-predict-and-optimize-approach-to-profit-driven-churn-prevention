import torch
torch.autograd.set_detect_anomaly(True) #https://stackoverflow.com/questions/67203664/how-to-change-pytorch-sigmoid-function-to-be-steeper



class NN(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size):
        
        super(NN, self).__init__()

        # Definimos capas
        #el bias=True se lo pone a la capa anterior
        self.only_hidden = torch.nn.Linear(in_features = input_size, out_features = hidden_size,
                                           bias=True)
        self.salida = torch.nn.Linear(in_features = hidden_size, out_features = 1,
                                      bias=True)
        

    # Dentro de las clases, se pueden crear funciones que aplican a la clase definida.
    # Esta función forward la utilizamos para computar la pasada hacia adelante, 
    # que en este caso es la predicción realizada por la red.
    def forward(self, x):
        
        l = self.only_hidden(x)
        l = torch.tanh(l) #Después de cada multiplicación de matrices al pasar de una capa a otra, existe una función de activación
        
        o = self.salida(l)
        #o = 1 + torch.tanh(o)
        
        return o

    
# https://machinelearningmastery.com/building-a-logistic-regression-classifier-in-pytorch/
class NN_logistic(torch.nn.Module):
    # build the constructor
    def __init__(self, input_size):
        super(NN_logistic, self).__init__()
        self.linear = torch.nn.Linear(in_features = input_size,
                                      out_features = 1)
    # make predictions
    def forward(self, x):

        y_pred = self.linear(x)
        return y_pred