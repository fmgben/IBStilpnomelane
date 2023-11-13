import pandas as pd
import numpy as np
import pandas as pd
import chempy
from chempy import Substance
from chempy.util import periodic
from typing import Union
from pathlib import Path
num2symbol = {p+1:n for p, n in enumerate(periodic.symbols)}
sym2num = {n:p+1 for p, n in enumerate(periodic.symbols)}

from matplotlib import pyplot as plt
assumed = pd.read_csv('data\mineral_compositions.csv')
measured =  pd.read_csv('reference\sem_assays.csv')
measured.Fe
plt.plot(measured.Fe, measured.Mg,'.')
plt.plot(assumed.Fe, assumed.Mg,'.')
for _,i in assumed.iterrows():
    plt.text(i.Fe, i.Mg, i.mineral)
plt.show()
'K5Fe48(Si63Al9)O165(OH)48(H2O)12'
tmp_form = 'Fe3Mg2Al5Si3O10(OH8)'
reagent =Substance.from_formula(tmp_form)
tmp_composition = process_composition(reagent.composition)
tc = pd.DataFrame(tmp_composition,index=[0])
tc/tc.sum(1).values.reshape(-1,1)
from tqdm import tqdm
compdf = []
compositions = []
for d in tqdm(range(5,30,5)):
    for p in range(20,100,10):
        for j in range(3,10):
            for i in range(0,60):
                tmp_form = f'K{j}Fe{60-i}Mg{i}(Si{p}Al{d})O165(OH)48(H2O)12'
                reagent =Substance.from_formula(tmp_form)
                tmp_composition = process_composition(reagent.composition)
                compositions.append(tmp_form)
                compdf.append(tmp_composition)


compdf = []
compositions = []
for z in range(10,20):
    for a in range(0,5):
        for b in range(0,10):
            for c in range(0,10):
                tmp_form = 'Fe3Mg2Al5Si3O10(OH8)'
                tmp_form = f'Fe{5-a}Mg{a}Al{c}Si{b}O{z}(OH8)'
                reagent =Substance.from_formula(tmp_form)
                tmp_composition = process_composition(reagent.composition)
                compositions.append(tmp_form)
                compdf.append(tmp_composition)


compositions = pd.DataFrame(compositions)
tt = pd.DataFrame(compdf)
ttt = tt/tt.sum(1).values.reshape(-1,1)*100
midx = measured.Mineral == 'Chlorite'
from sklearn.neighbors import KDTree
tt.columns

mm  = measured.loc[midx,measured.columns.isin(tt.columns)]
ccc = [i for i in tt.columns if i in mm.columns]
ccc=  ['O','Mg','Al','Si','Fe']
dtree = KDTree(ttt[ccc])
#mm = mm[~mm.isna().any(axis=1)]
#mm = mm[(mm.Si<24) & (mm.Al<4)]
from scipy.stats.mstats import gmean,mode
target_comp  = np.mean(mm[ccc].values,0)
_,cidx = dtree.query(target_comp.reshape(1,-1),1)
compositions.iloc[cidx.ravel(),:]
midx = measured.Mineral.isin(['Chlorite', 'Stilp'])
aidx = assumed.mineral.isin(['ferrostilpnomelane','ferristilpnomelane'])
ex = ['Fe']
ey = ['Mg']
tx = [i[0] for i in enumerate(ccc) if i[1] in ex[0]]
ty = [i[0] for i in enumerate(ccc) if i[1] in ey[0]]

plt.plot(ttt[ex],ttt[ey],'.',zorder=-20)
plt.plot(target_comp[tx],target_comp[ty],'k+',markersize=30)
plt.scatter(measured[ex][midx], measured[ey][midx],c=pd.Categorical(measured.Mineral[midx]).codes)
plt.plot(assumed[ex][aidx], assumed[ey][aidx],'.')
plt.plot(assumed[ex], assumed[ey],'.')
for c in cidx:
    plt.text(ttt.loc[c,ex].values,ttt.loc[c,ey].values,compositions.iloc[c].values[0][0])
plt.plot(ttt.loc[c,ex].values,ttt.loc[c,ey].values,'ro')
plt.xlabel(ex[0])
plt.ylabel(ey[0])


for _,i in assumed.iterrows():
    plt.text(i[ex], i[ey], i.mineral)
plt.xlabel(ex[0])
plt.ylabel(ey[0])
plt.show()


midx = measured.Mineral == 'Stilp'
col_masses = {}
for c in measured.columns:
        if c in sym2num:
            col_masses.update({c:periodic.relative_atomic_masses[sym2num[c]-1]})
measured.isna()
mean_mineral = measured.groupby('Mineral')[list(col_masses.keys())].median().reset_index()
mean_mineral.sum(1)
mean_mineral.to_csv('reference/measured_minerals.csv',index=False)
mean_mineral[col_masses.keys()].sum(1)
(mean_mineral[col_masses.keys()]/(mean_mineral[col_masses.keys()]/col_masses.values()).min(1).values.reshape(-1,1)).round()

measured_elements = measured[col_masses.keys()]
measured_elements[measured_elements.isna()] = 0
elements_mols = measured_elements/col_masses.values()
elements_ratios = elements_mols/elements_mols.min(1).values.reshape(-1,1)
er = elements_ratios.round()
tmpn = []

for n,i in er.iterrows():
    el_ok = i[~i.isna()]
    tmp_min_str = []
    for e in el_ok.index[::-1]:
        if int(el_ok[e]) == 1:
            tmp_min_str.append(e)
        else:
            tmp_min_str.append(f'{e}{int(el_ok[e])}')
    min_str = ''.join(tmp_min_str)
    mmm = measured.loc[n].Mineral
    print(min_str,mmm)


    tmp_mineral = ''.join([f'{z[1]}{int(z[0])}' for z in zip(i[~i.isna()].values, i[~i.isna()].index.to_list())])
    print(tmp_mineral)

    tmp_ = process_composition(Substance.from_formula(tmp_mineral).composition)
    tmpn.append(tmp_)
normalised = pd.DataFrame(tmpn)
normalised = normalised/(pd.DataFrame(normalised).sum(1).values.reshape(-1,1))*100

er
aidx = assumed.mineral.isin(['ferrostilpnomelane','ferristilpnomelane'])
ex = ['Fe']
ey = ['Mg']
measured
from scipy.stats import gmean
gx, gy =gmean(measured.loc[midx,[ex[0], ey[0]]])
plt.plot(measured.loc[midx,ex].median(), measured.loc[midx,ey].median(),'ro')
plt.plot(gx, gy,'r+')

plt.plot(assumed.loc[aidx,ex],assumed.loc[aidx,ey],'+',markersize=30)
#    O   F   Na   Mg   Al    Si   P   S    K  Ca  Ti   Cr   Mn     Fe  Cu  Zn  Ba  Pb
plt.scatter(measured.loc[midx,ex], measured.loc[midx,ey],c=measured.K[midx]/measured.Mg[midx])
plt.colorbar()
plt.xlabel(ex[0])
plt.ylabel(ey[0])
plt.show()

