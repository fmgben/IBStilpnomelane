from src.normative.normative import solve
from src.parser.parser import read_reference_minerals, ox_factor,clean_name
import pandas as pd
import numpy as np

minerals, mineral_matrix = read_reference_minerals(r'reference\minerals.csv')
mineral_pass_data = pd.read_csv(r'reference/selected_minerals.csv')
data = pd.read_csv(r'reference/BV_XRD_Assay_Merge.csv')
mineral_order = pd.read_csv(r'reference/solution_order.csv')

clean_assay_names = [clean_name(c,[' XRF']) for c in data.columns] 
data.columns=clean_assay_names

assay_columns = ['Fe3O4', 'Fe', 'SiO2', 'Al2O3', 'TiO2', 'Mn', 'CaO', 'P', 'S', 'MgO', 'K2O', 'Na2O', 'Zn', 'As', 'Cl', 'Cu', 'Pb', 'Ba', 'V', 'Cr', 'Ni', 'Co', 'Sn', 'Zr', 'Sr','LOI371', 'LOI650', 'LOI1000']

assays = data[assay_columns]
idx = ~assays.isna().any(axis=1)
assays = assays[idx].reset_index(drop=True)

dtr = pd.read_csv(r'data/ib_dtr.csv')
# keep only bv data
idxlab = dtr.DTR_Labcode_D == 'BV'
dtr  = dtr[idxlab].copy().reset_index()

tails = [i for i in dtr.columns if i.find('_T')>=0 and (i.find('MassRec')<0) and (i.find('Tot')<0)]
cons = [i for i in dtr.columns if i.find('_C')>=0 and (i.find('_ConsWt')<0) and (i.find('Wt_Con_g')<0)]
head = [i for i in dtr.columns if (i.find('_H')>=0) and (i.find('_ConsWt')<0)]
dtr.columns
tail_assay = dtr[tails].copy()
tail_assay.columns
clean_tail_names = [clean_name(c,[' XRF','_T']) for c in tail_assay.columns] 
tail_assay.columns = clean_tail_names
# clean the tail assay
tail_assay[tail_assay<-10] =np.nan 
tail_assay[tail_assay>100] =np.nan

tail_assay['Fe3O4'] = 0
con_assay = dtr[cons].copy()
con_assay.columns
clean_con_names = [clean_name(c,[' XRF','_C']) for c in con_assay.columns] 
con_assay.columns = clean_con_names
# clean the tail assay
con_assay[con_assay<-10] = np.nan
con_assay[con_assay>100] =np.nan
selected_con_columns = [c for c in con_assay.columns if c not in ['MassRec_pct','TotonsChk']]

con_assay = con_assay[selected_con_columns]
idxok = ~con_assay[['LOI1000','LOI371','LOI650','Fe3O4']].isna().any(axis=1)
con_assay[idxok]
tail_assay[idxok]
frac_cons = dtr['MassRec_C_pct']/100
frac_tail = 1-dtr['MassRec_C_pct']/100
head_assay = con_assay*frac_cons.values.reshape(-1,1)+tail_assay*frac_tail.values.reshape(-1,1)
head_columns = [c for c in head_assay.columns if c not in ['MassRec_pct', 'TotonsChk'] ]
head_assay = head_assay[head_columns]
minerals,limiting_reagent, intersected_elements = solve(head_assay, mineral_matrix, mineral_order)
head_assay[idxok]
plt.plot(minerals[idxok].ferristilpnomelane)
plt.show()
plt.plot(limiting_reagent.ferristilpnomelane.map(lambda x:intersected_elements[x]),limiting_reagent.ferrostilpnomelane.map(lambda x:intersected_elements[x]),'.')
plt.show()
had[idxok]
from matplotlib import pyplot as plt

['Fe','Mn','P','SiO2','CaO','MgO']
minerals.max()
minerals.columns
dtr.columns
joined = pd.concat([con_assay, dtr['MassRec_pct_BEST']],axis=1)
pd.plotting.scatter_matrix(joined)
plt.show()

plt.plot(con_assay['Fe'],head_assay['K2O'],'.')
plt.show()

for i in con_assay:
    try:
        plt.plot(con_assay[i],tail_assay[i],'.')
        plt.title(i)
        #plt.plot(con_assay[i],dtr.MassRec_C_pct,'.')
        ref =[min([con_assay[i].min(),tail_assay[i].min()]),max([con_assay[i].max(),tail_assay[i].max()])]
        plt.plot(ref,ref,'-')
        plt.xlabel(i)
        plt.ylabel('Mass Rec C')
        plt.ylabel(i)
        plt.show()

    except Exception:
        pass

1300 661 508
105 218 321 5223

delta = 1e-5
mm = mineral_matrix.values.copy()
mm[np.isnan(mm)]=0
U,D,V= np.linalg.svd(mm)
U, sdiag, VH = np.linalg.svd(mm)
S = np.zeros(mm.shape)
np.fill_diagonal(S, sdiag)
V = VH.T.conj()  # if you know you have real values only you can leave out the .conj()
plt.imshow(V)
plt.imshow(S)
plt.show()

V[D<delta]