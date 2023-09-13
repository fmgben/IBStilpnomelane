import pandas as pd
import re
data = pd.read_csv(r'reference/BenCompXRD.csv')

cleaned = data.Client_sample_ID.map(lambda x :re.sub('[0-9\ ]{9}','',x)).str.replace('comp','Comp')

def return_match(x):
    reg = re.findall('[A-Z0-9]{8} Comp [0-9]{1,2}',x)
    if len(reg)>0:
        out = reg[0]
    else:
        out = x
    return out

dta = pd.read_csv(r'reference/DTW53_assays.csv')

x = data.Client_sample_ID.values[0]
data['Sample ID'] = cleaned.map(lambda x: return_match(x)).str.replace(' Comp ','-F-').values
data['Type'] = cleaned.map(lambda x: x.split(' ')[-1]).values
merged = pd.merge(data, dta,on=['Sample ID','Type'])
data['Type'].unique()
merged.to_csv(r'reference/compiled_xrd.csv')
merged.columns
from matplotlib import cm
for i in merged['Type'].unique():
    idx = merged['Type'] == i
    plt.scatter(merged['K-Feldspar'][idx],merged['K2O_%'][idx],label=i)
    plt.xlabel('k-spar')
    plt.ylabel('k2o')
plt.legend()
plt.show()
