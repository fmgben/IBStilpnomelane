import pandas as pd
import numpy as np
import chempy
from chempy import Substance
from chempy.util import periodic
from chempy import Equilibrium
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error
from typing import Union
from pathlib import Path
num2symbol = {p+1:n for p, n in enumerate(periodic.symbols)}

def process_composition(comp):
    ''' 
    process the compositions into a dictionary.

    '''
    elemental_mass = {}
    for c in comp:
        tmp_el = num2symbol[c]
        tmp_mass = comp[c]*periodic.relative_atomic_masses[c-1]
        elemental_mass.update({tmp_el:tmp_mass})
    return elemental_mass

def loi_parser(reagent,products,gasses=['SO2','H2O','CO2']):
    """
    parse the loi values from the mineralogical formula
    """
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
    """
    remove suffixes from the assay data so that the chemical parser works
    """
    y = x.strip('_%').replace(' XRF','')
    return y

def ox_factor(x):
    """
    Calculate the oxide conversion factor between the assays and the raw elements.
    """
    reagent = Substance.from_formula(x)
    comp = process_composition(reagent.composition)
    mass_fracts = chempy.mass_fractions(comp)
    key = [c for c in mass_fracts.keys() if c != 'O']
    factor = mass_fracts[key[0]]
    return x,key[0], factor

def read_reference_minerals(path_reference:Union[str, Path]):
    '''
    reads the reference minerals and converts the equations to a dense matrix of weights
    '''
    minerals = pd.read_csv(path_reference,engine='python')
    loi_decomposition = [i for i in minerals.columns.to_list() if i.startswith('loi')]

    minerals['substance'] = ''
    tmp_elements = []
    tmp_loi = []
    value = minerals.loc[2]
    for i,value  in minerals.iterrows():
        reagent = Substance.from_formula(value.formula)
        tmp_loi_calc = {}
        for loi in loi_decomposition:
            if  not isinstance(value[loi],float):
                products = value[loi]
                tmp_calc = loi_parser(reagent, value[loi])
                tmp_loi_calc.update({loi:tmp_calc})
        tmp_loi.append(tmp_loi_calc)
        tmp_composition = process_composition(reagent.composition)
        tmp_elements.append(tmp_composition)
        minerals.loc[i,'substance'] = reagent

    # calculate the loi, elements and their proportions
    loi = pd.DataFrame(tmp_loi)
    elemental = pd.DataFrame(tmp_elements)
    proportions = elemental/elemental.sum(1).values.reshape(-1,1)
    
    solution_matrix = pd.concat([proportions, loi],axis=1).astype(float)
    solution_matrix.index = minerals.mineral
    return minerals, solution_matrix

