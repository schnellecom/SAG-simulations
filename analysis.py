# i = 1
# with open("data/rcv1_train.binary", "rb") as f:
#     while (byte := f.read(1)) and i < 10:
#         print(byte)
#         i = i+1
        # Do stuff with byte.
import numpy as np
import pandas as pd
# np.fromfile("data/rcv1_train.binary")
data = pd.read_csv('data/phy_train.dat', sep='\t', header=None, usecols=[i for i in range(1,80)])
data = data.transpose()

print(data[2])
