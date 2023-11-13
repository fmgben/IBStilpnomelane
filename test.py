from src.normative import normative
from src.parser.parser import read_reference_minerals, ox_factor
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
from itertools import combinations
minerals, solution_matrix = read_reference_minerals(r'C:\Users\f697768\Documents\IBStilpnomelane\reference\minerals.csv')
mineral_pass_data = pd.read_csv(r'reference/selected_minerals.csv')
assays = pd.read_csv(r'reference/BV_XRD_Assay_Merge.csv')



assays.columns =[c.replace(' ','_') for c in assays.columns]
assay_columns = ['Fe3O4', 'Fe', 'SiO2', 'Al2O3', 'TiO2', 'Mn', 'CaO', 'P', 'S', 'MgO', 'K2O', 'Na2O', 'Zn', 'As', 'Cl', 'Cu', 'Pb', 'Ba', 'V', 'Cr', 'Ni', 'Co', 'Sn', 'Zr', 'Sr']
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

# magnetite allocation to loi, 0.14, 0.3, 0.56
cut  = [ 'ferristilpnomelane','clinochlore','siderite','calcite','fe-ankerite','mg-ankerite','siderite','sepiolite']
G = nx.Graph()
# add all the minerals as nodes
for i in cut:
    G.add_node(i,label=i)

# connect the mineral nodes
# a connected mineral node is one where an element depends on more than 1 mineral
for i in solution_matrix.loc[idx_minerals,selected_columns].columns:
    idxelement = ~solution_matrix.loc[idx_minerals,i].isna()
    nodal_minerals = idxelement.index[idxelement]
    # remove the minerals that have their concentrations limited by a single assay
    idx = nodal_minerals.isin(first_pass_minerals) 
    idx = nodal_minerals.isin(cut)

    path_minerals = nodal_minerals[idx].tolist()
    for fr, to in combinations(path_minerals, 2):
        G.add_edge(fr, to)

G.adjacency()
nx.density(G)

nx.draw(G, with_labels=True)
plt.show()


for i in G.adjacency():
    i[0],i[1]

for i in cut:
    G.degree(i)

first_pass_minerals
last_pass_minerals

for i in nx.all_simple_paths(G, 'ferristilpnomelane', 'clinochlore'):
    i


