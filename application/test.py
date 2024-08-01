import matplotlib.pyplot as plt

import numpy as np

x = [1,2,3,4,5]
y = [6,7,8,9,10]

x,y = np.meshgrid(x,y)

print(x,y)
