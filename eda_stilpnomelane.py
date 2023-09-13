import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

xrd = pd.read_csv(r'reference/head_qxrd.csv')
assays = pd.read_csv(r'reference/head_assays.csv')
all = xrd.merge(assays)
for i in assays.columns[3:]:
    plt.figure()
    plt.plot(all['Stilpnomelane_%'],all[i],'.')
    plt.title(i)
    plt.xlabel('Stilpnomelane_%')
    plt.ylabel(i)
plt.show()

