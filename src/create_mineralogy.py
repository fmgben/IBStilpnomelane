from src.normative import normative
from src.parser.parser import read_reference_minerals, ox_factor
import pandas as pd
import numpy as np

minerals, solution_matrix = read_reference_minerals(r'C:\Users\f697768\Documents\IBStilpnomelane\reference\minerals.csv')
mineral_pass_data = pd.read_csv(r'reference/selected_minerals.csv')
assays = pd.read_csv(r'reference/BV_XRD_Assay_Merge.csv')


assays.columns =[c.replace(' ','_') for c in assays.columns]
assay_columns = ['Fe3O4', 'Fe', 'SiO2', 'Al2O3', 'TiO2', 'Mn', 'CaO', 'P_XRF', 'S_XRF', 'MgO', 'K2O', 'Na2O', 'Zn', 'As', 'Cl', 'Cu', 'Pb', 'Ba', 'V', 'Cr', 'Ni', 'Co', 'Sn_XRF', 'Zr', 'Sr']
loi_columns = ['LOI371', 'LOI650', 'LOI1000']
idx = np.all(~assays[assay_columns].isna(),1)
xrf_assays = [c for c in assay_columns if not c =='Fe3O4']
clean_assays = assays.loc[:,xrf_assays]
clean_assays.columns = [c.replace('_XRF','') for c in clean_assays.columns]
factor_names =[ox_factor(c) for c in clean_assays.columns.to_list()]
elements = {f[0]:f[1] for f in factor_names}
factors = [f[2] for f in factor_names]
elemental_assays= clean_assays*factors
elemental_assays = elemental_assays.rename(columns=elements)
elemental_assays[elemental_assays<0]=0


loi = assays[loi_columns]
loi.columns = [c.lower() for c in loi_columns]

final_assays = pd.concat([elemental_assays,loi],axis=1) 
final_assays[final_assays.isna()]=0
idx_joint= final_assays.columns.isin(solution_matrix.columns)
selected_columns = final_assays.columns[idx_joint]

idx_minerals = solution_matrix.index.isin(mineral_pass_data.mineral.values)

idx_single = (~solution_matrix.loc[idx_minerals,selected_columns].isna()).sum()==1
single_elements = selected_columns[idx_single].to_list()
solution_matrix.loc[idx_minerals, single_elements].idxmax(0)

first_pass_minerals = solution_matrix.loc[idx_minerals,single_elements].idxmax().values

# find the minerals that can consume an entire assay and leave those until last
pos_last = np.argwhere(((solution_matrix.loc[idx_minerals,selected_columns]>0).sum(1).values == 1))
last_pass_minerals = [target_minerals[i] for i in pos_last.ravel()]
target_minerals = solution_matrix[idx_minerals].index.to_list()


# extract everything into numpy arrays
x = final_assays[selected_columns]

cut = set(target_minerals).difference(first_pass_minerals)
cut = cut.difference(last_pass_minerals)
