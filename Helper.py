"""Helper functions due to Aya (https://github.com/ayaabdelsalam91)"""
import numpy as np
import pandas as pd
import itertools
import torch
import time

def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

def save_intoCSV(data,file,Flip=False,col=None,index=False):
    if(Flip):
        print("Will Flip before Saving")
        data=data.reshape((data.shape[1],data.shape[0]))
    df = pd.DataFrame(data)
    if(col!=None):
        df.columns = col
    df.to_csv(file,index=index)
