import torch

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
