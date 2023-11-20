from ..normative.normative import solve,apply_mineral_matrix
from ..parser.parser import read_reference_minerals, ox_factor
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Union
from tqdm import tqdm
from pyparsing.exceptions import ParseBaseException
import numpy as np
from ..v1.normative import normative_v1
# find where the file is so that we can go and find the reference data.
def ApplyV2(assay:DataFrame, 
                    solution_order:Path|str|None|DataFrame=None, 
                    minerals:Path|str|None|DataFrame=None, 
                    reference:Path|str|None|DataFrame=None, 
                    xrd_to_mineral:Path|str|None|DataFrame=None)->dict[pd.DataFrame]:
    mineral_order:DataFrame
    data:DataFrame
    minerals:DataFrame
    mineral_matrix:DataFrame
    mineral_path:Union[DataFrame,Path]
    solution_path:Path
    xrd_path:Path
    package_directory:Path = Path(__file__).parents[1]
    # deal with the various ways of getting data into the solver 
    # inputs can be either a path or a dataframe
    if isinstance(solution_order, DataFrame):
        mineral_order = solution_order.copy()
    else:
        solution_path = package_directory.joinpath(r'reference/solution_order.csv')
        mineral_order = pd.read_csv(solution_path)

    if isinstance(minerals, DataFrame):
        mineral_path = minerals.copy()
    else:
        mineral_path = package_directory.joinpath(r'reference/minerals.csv')
        mineral_path = pd.read_csv(mineral_path)

    if isinstance(reference, DataFrame):
        data = reference.copy()
    else:
        xrd_path = package_directory.joinpath(r'reference/xrd_and_assay.csv')
        data = pd.read_csv(xrd_path)
    if isinstance(xrd_to_mineral, DataFrame):
        mineral_pass_data = xrd_to_mineral.copy()
    else:
        mineral_xrd_path = package_directory.joinpath(r'reference/mineral_to_xrd.csv')
        mineral_pass_data = pd.read_csv(mineral_xrd_path)

    # split the reference data into assays and xrd
    mineral_columns = []
    assay_columns = []
    for i in data.columns:
        try:
            if i.startswith('LOI'):
                assay_columns.append(i)
            else:
                # this is only to see if the column is an assay or mineral
                # therefore we discard the values
                _ = ox_factor(i)
                assay_columns.append(i)
        except (ParseBaseException,ValueError) as e:
            mineral_columns.append(i)


    minerals, mineral_matrix = read_reference_minerals(mineral_path)
    train_xrd = data[mineral_columns].copy()
    train_assay = data[assay_columns].copy()
    idxassay = ~np.any(train_assay.isna(),1)
    train_assay = train_assay[idxassay].reset_index(drop=True)
    train_xrd = train_xrd[idxassay].reset_index(drop=True)
    inital_solution,_,_,new_matrix = solve(train_assay, train_xrd,mineral_matrix, mineral_order,mineral_pass_data)

    solution, limiting_reagent=apply_mineral_matrix(assay, mineral_order,new_matrix)

    return {'normative':solution, 'limiting_reagent':limiting_reagent, 'mineral_composition':new_matrix}

def ApplyNuer(assay:DataFrame)->pd.DataFrame:
    assay.columns = [c.lower() for c in assay.columns]
    result = normative_v1(assay)
    return {'normative':result}


def Mineralogy(assay:DataFrame, 
                    solution_order:Path|str|None|DataFrame=None, 
                    minerals:Path|str|None|DataFrame=None, 
                    reference:Path|str|None|DataFrame=None, 
                    xrd_to_mineral:Path|str|None|DataFrame=None,version:str='v2'):
    '''
    Function that calculates the mineralogy using the supplied method the default method is the latest.
    
    assay:DataFrame of assays
    solution_order: order in which the minerals are solved
    minerals: reference library of minerals
    reference: reference libary of xrd and assay used to fine tune the results
    version: which version of normative mineralogy is applied, must be either "v2","nuer","als"

    N.B the dataframe of the assays must only contain the assay columns
    assay columns must be named as chemical formulae so Fe, Fe2O3 are good 
    LOI columns must start with LOI so LOI1000,LOI371,LOI650 are good, LOI_650,loi650 are bad
    FE, FE_PCT, FE_PCT_BEST etc are not acceptable and will cause the program to fail.

    '''

    # configuration files
    # and config overwrite where applicable 

    if version == 'v2':
        result = ApplyV2(assay.copy(), solution_order, minerals, reference, xrd_to_mineral)
    elif version == 'nuer':
        result = ApplyNuer(assay.copy())
    elif version =='als':
        result = {'not coded':'na'}
    else:
        KeyError('version {version} not coded version must be one of v2, nuer or als')

    return result