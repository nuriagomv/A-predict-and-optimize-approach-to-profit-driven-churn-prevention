import torch
import pandas as pd


def step_function(x):
    return 1 - torch.relu( torch.sign(x) ) 

def tuned_sigmoid(x, c1, c2):
    
    return 1/( 1 + torch.exp(-c1*(x-c2)) )

"""
import numpy as np
import matplotlib.pyplot as plt

input = torch.randn(1000)
CLV = 0.6
sigmoid_slope = 10

x, y2 = list(input.numpy()), list((1 - torch.relu( torch.sign(input -CLV) )).numpy())
x, y2 = [i for i, _ in sorted(zip(x,y2))], [j for _, j in sorted(zip(x,y2))]
plt.plot(x,y2, color='blue', label='$z^*(\hat{y})$', linewidth=2.5)
plt.vlines(CLV, 0,1, color='white',linewidth=3.)

output = 1- tuned_sigmoid(input, sigmoid_slope, CLV)
x,y=list(input.numpy()),list(output.numpy())
x, y = [i for i, _ in sorted(zip(x,y))], [j for _, j in sorted(zip(x,y))]
plt.plot(x,y, color='green', label='$g(\hat{y})$')
plt.vlines(CLV, 0,1, colors='red', linestyles='dashed', label='$m^i$')
#plt.hlines(0.5, -3,3)

plt.legend()
plt.savefig('L_Lsmooth.svg',bbox_inches='tight', pad_inches=0.1)

plt.show()
"""

def loss(y, y_pred, CLV, f = 1, d = 10, gamma = 0.3):

    m = ( f + gamma*(d-CLV) )/( gamma*(d-CLV) - d )

    z_pred = 1 - torch.relu( torch.sign(y_pred - m) ) #https://discuss.pytorch.org/t/binary-activation-function-with-pytorch/56674/3
    z_opt = 1 - y
    
    loss = torch.mul(torch.sub(z_pred, z_opt),
                     f + y*d + (1-y)*gamma*(d-CLV)
                     )

    return z_pred, z_opt, loss


def smooth_loss(y, y_pred, sigmoid_slope, CLV, f = 1, d = 10, gamma = 0.3):

    m = ( f + gamma*(d-CLV) )/( gamma*(d-CLV) - d )

    z_pred = 1 - tuned_sigmoid(y_pred, sigmoid_slope, m)
    z_opt = 1 - y
    
    smooth_loss = torch.mul(torch.sub(z_pred, z_opt),
                            f + y*d + (1-y)*gamma*(d-CLV)
                            )

    return z_pred, z_opt, smooth_loss


def top10lift(y_true, y_pred_prob):
    #https://chatgpt.com/share/6717fefd-9b2c-800d-9a6c-13a346cfc39d
    # Create a DataFrame to store actuals and predicted probabilities
    df = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})

    # Sort by predicted probabilities in descending order
    df = df.sort_values(by='y_pred_prob', ascending=False)

    # Calculate the number of instances to include in the top decile (top 10%)
    top_decile_count = int(0.1 * len(df))

    # Select the top 10% of instances based on predicted probabilities
    top_decile_df = df.iloc[:top_decile_count]

    #PARA MI, LA CLASE POSITIVA (SER CHURNER) ES Y=0
    # Proportion of positive outcomes in the entire dataset
    overall_positive_rate = (df['y_true']==0).mean()

    # Proportion of positive outcomes in the top decile
    top_decile_positive_rate = (top_decile_df['y_true']==0).mean()

    # Calculate the Top Decile Lift
    top_decile_lift = top_decile_positive_rate / overall_positive_rate

    return top_decile_lift
