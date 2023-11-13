import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data/ib_all.csv')
data.columns
plt.plot(data['Fe3O4_H-STMGN_pct'],data.Fe_pct_BEST,'.')
plt.xlim([0,100])
plt.ylim([0,120])
plt.show()




xrd = pd.read_csv('reference/BV_XRD_Assay_Merge_2.csv')
xrd = xrd.replace(np.nan, 0)
tima =pd.read_csv('reference/TIMA_AXT_Cons_and_tails.csv')
tima.rename(columns ={'FMG Training (Long List) / Sample':'SampleID'},inplace=True)
def name_getter(x):
    sx = x.split(' ')
    y = {'BHID':sx[2],'Composite':' '.join(sx[3:5])}
    return y 



tima_data = pd.concat([tima, tima.SampleID.map(name_getter).apply(pd.Series)],axis=1)
tima_data.Type = tima_data.Type.map({'Mags':'Cons','Tails':'Tails'})
merge_data = pd.merge(tima_data, xrd, left_on=['BHID', 'Composite','Type'],right_on=['BHID_x', 'Composite_x','Product'],suffixes=['_TIMA', '_XRD'])

plt.plot(merge_data.Stilpnomelane_XRD,merge_data.Stilpnomelane_TIMA,'.',label='XRD Stilpnomelane')
plt.plot(merge_data.Stilpnomelane_XRD+merge_data.Sepiolite,merge_data.Stilpnomelane_TIMA,'.',label='XRD Stilpnomelane + Sepiolite')
plt.xlabel('Stilpnomelane % XRD')
plt.ylabel('Stilpnomelane % TIMA')
plt.plot([0,10],[0,10],'k',label='1:1 Line')
plt.grid()
plt.xlim([-0.1,3])
plt.ylim([-0.1,3])
plt.legend()
plt.show()







merge_data
for i in ['Stilpnomelane_TIMA','Stilpnomelane_XRD', 'Sepiolite']:
    if i =='Sepiolite':
        name = i+'_XRD'
    else:
        name = i
    plt.plot(merge_data[i],label=name)
plt.xlabel('Sample Number')
plt.ylabel('Concentration')
plt.title('TIMA vs XRD for Cons and tails')
plt.legend()
plt.show()
import numpy as np
merge_data = merge_data.replace(np.nan, 0)
plt.plot(merge_data['Stilpnomelane_XRD'],label='Stilpnomelane XRD')
plt.plot(merge_data['Sepiolite'])
plt.plot(merge_data['Stilpnomelane_TIMA'])
plt.show()
plt.plot(merge_data['Chlorite group'])
plt.plot(merge_data['Smectite group'])
plt.plot(merge_data['Kaolinite-serpentine group'])
plt.plot(merge_data['Chlorite + Clay Minerals'],'.')
plt.show()
merge_data.columns
merge_data.colsumn
els = ['FE','SIO2','AL2O3','TIO2','K2O','LOI_371','LOI_650','LOI_1000']
els = ['Fe','SiO2','Al2O3','TiO2','K2O','LOI371','LOI650','LOI1000']

xrd.columns
idxf = xrd.Product=='Feed'

soft = ['Stilpnomelane', 'Chlorite group', 'Mica group']
med = ['Calcite group-Siderite', 'Calcite group', 'Dolomite group']
hard = ['Spinel group', 'Hematite group','Quartz','K-Feldspar','Plagioclase']
h = xrd[hard].sum(1)[idxf]
s = xrd[soft].sum(1)[idxf]
m = xrd[med].sum(1)[idxf]
minerals = [*hard, *med,*soft]
from sklearn.metrics import r2_score
pr = lr.predict(xrd[els])[idxf].reshape(-1,1)
pr[pr>1] = np.nan
for n,i in enumerate([soft, med, hard]):
    plt.subplot(1,3, n+1)
    plt.plot(xrd.loc[idxf, i].sum(1),pr,'.')
plt.show()
plt.scatter(h-m,pr ,c=pd.Categorical(xrd.BHID_x[idxf].str[0:2]).codes)
plt.show()


plt.plot(r2_score(xrd.loc[idxf,minerals].values,np.tile(pr,(1,11)),multioutput='raw_values'))
plt.show()
plt.imshow(np.corrcoef(pr,xrd.loc[idxf,minerals]))
plt.colorbar()
plt.show()
plt.imshow(r2_score(np.tile(pr,(1,11)),xrd.loc[idxf,minerals],multioutput='raw_values'))
plt.plot((h-s)/m,lr.predict(xrd[els])[idxf],'.')
plt.show()