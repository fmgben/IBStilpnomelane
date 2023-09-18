import pandas as pd
import numpy as np
import chempy
from chempy import Substance
from chempy.util import periodic
from chempy import Equilibrium
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error
num2symbol = {p+1:n for p, n in enumerate(periodic.symbols)}
def process_composition(comp):
    elemental_mass = {}
    for c in comp:
        tmp_el = num2symbol[c]
        tmp_mass = comp[c]*periodic.relative_atomic_masses[c-1]
        elemental_mass.update({tmp_el:tmp_mass})
    return elemental_mass

def loi_parser(reagent,products,gasses=['SO2','H2O','CO2']):
    split_products = products.replace(' ','').split(';')
    try:
        reac, prod = chempy.balance_stoichiometry({reagent.name,'O2'},split_products)
        massgain = True 
    except ValueError:
        # loss of water only decomposition 
        massgain = False
        reac, prod = chempy.balance_stoichiometry({reagent.name},split_products)
    solid_products = [s for s in split_products if s not in gasses]
    equation = reac
    equation.update(prod)
    forms = []
    for i in equation:
        forms.append(Substance.from_formula(i))
    mass_frame = []
    for f in forms:
        tmp_masses = {'name':f.name}
        tmp_masses.update({'mols':equation[f.name]})
        tmp_masses.update(process_composition(f.composition))
        mass_frame.append(tmp_masses)
    mass = pd.DataFrame(mass_frame)
    elcols = mass.columns[~mass.columns.isin(['name','mols'])]
    for i in elcols:
        idx = mass.loc[:,i].isna()
        mass.loc[idx,i] = 0

    mass['molmass'] = mass[elcols].sum(1)
    mass['molar_prop'] = (mass['molmass']*mass['mols'])
    mass['molar_prop']/mass['molar_prop'][0]
    # reagent mass
    # lost gass-gained gass/reactant_mass

    gass_index = mass.name.isin(gasses)
    o2_index = mass.name == 'O2'
    product_index = mass.name.isin(solid_products)
    reactant_index = mass.name.isin([reagent.name])
    lost_gass = mass[gass_index]['molar_prop'].sum()
    gained_gass = mass[o2_index]['molar_prop'].sum()
    reactant_mass =  mass[reactant_index]['molar_prop'].sum()
    product_mass = mass[product_index]['molar_prop'].sum()
    # take into gass gain (oxidation) and loss
    loi_value = (lost_gass-gained_gass)/(reactant_mass)

    return loi_value

def clean_name(x):
    y = x.strip('_%').replace(' XRF','')
    return y

def ox_factor(x):
    reagent = Substance.from_formula(x)
    comp = process_composition(reagent.composition)
    mass_fracts = chempy.mass_fractions(comp)
    key = [c for c in mass_fracts.keys() if c != 'O']
    factor = mass_fracts[key[0]]
    return x,key[0], factor

def normative(x, solution_matrix, target_minerals, factor=100):
    tmpx = x.values.copy()
    xneg = tmpx<0
    tmpx[xneg] = tmpx[xneg]*-0.5
    n_minerals = len(target_minerals)
    mineral_array = np.zeros((tmpx.shape[0],n_minerals ))
    for i, mineral in enumerate(target_minerals):
        idx = minerals.mineral == mineral
        tmpy = solution_matrix.loc[idx,:].values
        # clean the zeros
        idx_element = (tmpy!=0).ravel()
        tmp_solution = tmpx[:,idx_element]/(tmpy[:,idx_element]*factor)
        # find the minimum which is the limiting reagent
        mineral_lim = np.min(tmp_solution,1)
        mineral_lim = np.clip(mineral_lim,0,1)
        mineral_array[:,i] = mineral_lim
    return pd.DataFrame(mineral_array, columns=target_minerals)


path_reference = r'reference/minerals.csv'
minerals = pd.read_csv(path_reference,engine='python')


minerals['substance'] = ''
tmp_elements = []
tmp_loi = []
value = minerals.loc[2]
for i,value  in minerals.iterrows():
    reagent = Substance.from_formula(value.formula)
    tmp_loi_calc = {}
    for loi in ['loi371','loi650','loi1000']:
        if  not isinstance(value[loi],float):
            products = value[loi]
            tmp_calc = loi_parser(reagent, value[loi])
            tmp_loi_calc.update({loi:tmp_calc})
    tmp_loi.append(tmp_loi_calc)
    tmp_composition = process_composition(reagent.composition)
    tmp_elements.append(tmp_composition)
    minerals.loc[i,'substance'] = reagent

loi = pd.DataFrame(tmp_loi)
elemental = pd.DataFrame(tmp_elements)
proportions = elemental/elemental.sum(1).values.reshape(-1,1)
# join the data frames together as we will solve for both assay and the mineralogy
pd.concat([minerals['mineral'],proportions],axis=1)

solution_matrix = pd.concat([proportions, loi],axis=1).astype(float)

solution_matrix[solution_matrix.isna()]=0

assays = pd.read_csv(r'reference/compiled_xrd.csv')

# compiled xrd is missing 3pt xrf
from sklearn import linear_model
assay_columns = ['Fe_%', 'SiO2_%', 'Al2O3_%', 'TiO2_%', 'Mn_%', 'CaO_%', 'P XRF_%',
                'S XRF_%', 'MgO_%', 'K2O_%', 'Na2O_%', 'Zn_%', 'As_%', 'Cl_%', 'Cu_%',
                'Pb_%', 'Ba_%', 'V_%', 'Cr_%', 'Ni_%', 'Co_%', 'Sn XRF_%', 'Zr_%',
                'Sr_%', 'Fe3O4_%','LOI_ Total_%']
loi_columns = ['LOI_371_%', 'LOI_650_%', 'LOI_1000_%']
assay_ok = assays[loi_columns].isna().sum(1) == 0
loi_model = linear_model.LinearRegression().fit(assays.loc[assay_ok,assay_columns], assays.loc[assay_ok,loi_columns])
loi_hat = loi_model.predict(assays.loc[:,assay_columns])
# predict the loi into the missing
assays.loc[~assay_ok,loi_columns] = loi_hat[~assay_ok]
idxassay = assays['Type'].isin(['Feed', np.nan])
assays = assays[idxassay].reset_index(drop=True).copy()
mineral_pass_data = pd.read_csv(r'reference/selected_minerals.csv')
mineral_selection = mineral_pass_data.mineral.to_list()
xrd_to_assay = {}
for i in mineral_pass_data.xrd.unique():
    if isinstance(i, str):
        idx = mineral_pass_data.xrd == i
        xrd_to_assay.update({i:mineral_pass_data[idx].mineral.to_list()})


idx_mineral = minerals.mineral.isin(mineral_selection)

column_names = {f:clean_name(f) for f in assays.columns.to_list() if f.find('%')>=0}
clean_assays = assays[[c for c in column_names]].rename(columns=column_names)
clean_assays.sum(1)
clean_assays = clean_assays.iloc[:,0:24]


factor_names =[ox_factor(c) for c in clean_assays.columns.to_list()]
elements = {f[0]:f[1] for f in factor_names}
factors = [f[2] for f in factor_names]
elemental_assays= clean_assays*factors
elemental_assays = elemental_assays.rename(columns=elements)
elemental_assays[elemental_assays<0]=0
loi = assays[['LOI_371_%', 'LOI_650_%', 'LOI_1000_%']]
loi = loi.rename(columns={'LOI_371_%':'loi371', 'LOI_650_%':'loi650', 'LOI_1000_%':'loi1000'})
final_assays = pd.concat([elemental_assays,loi],axis=1) 
final_assays[final_assays.isna()]=0
idx_joint= final_assays.columns.isin(solution_matrix.columns)
selected_columns = final_assays.columns[idx_joint]

minerals =['Spinel_group', 'Hematite_group', 'Quartz',
            'Chlorite_group', 'Dolomite_group', 'Calcite_group-Siderite',
            'Calcite_group-Calcite', 'Pyrite_group', 'Rutile_group', 'K-Feldspar',
            'Stilpnomelane', 'Mica_group', 'Plagioclase', 'Goethite',
            'Smectite_group', 'Sepiolite', 'Kaolinite-serpentine_group', 'Pyroxene']

from sklearn.metrics import mean_squared_error
assays[minerals],final_assays


