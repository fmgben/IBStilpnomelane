import pandas as pd
import numpy as np
from chempy.util import periodic
from matplotlib import pyplot as plt
from src.parser.parser import ox_factor

# we extract from the chempy the mapping between periodic number and the elemental name
num2symbol = {p+1:n for p, n in enumerate(periodic.symbols)}

def normative_calc(x:pd.DataFrame, solution_matrix:pd.DataFrame, target_minerals:list[str],factor:float=100)->pd.DataFrame:
    '''
    Calculate from the assays how much of the mineral can exist given the assay, for each element that we 
    have in both the minerals and the assay we divide the assay concentration by the mineral concentration
    the minimum value for all elements is then taken as the mineral proportion as this is the limiting reagent.
    '''
    # copy the dataframe
    tmpx = x.values.copy()
    # get the number of the target minerals
    n_minerals = len(target_minerals)
    # preallocate the numpy array of the minerals that we are going to solve for 
    mineral_array = np.zeros((tmpx.shape[0],n_minerals ))
    for i, mineral in enumerate(target_minerals):
        tmpy = solution_matrix.loc[mineral,:].values
        # clean the zeros
        idx_element = (~np.isnan(tmpy)).ravel()
        tmp_solution = tmpx[:,idx_element]/(tmpy[idx_element]*factor)
        # find the minimum which is the limiting reagent
        pos_element = np.where(idx_element)[0]
        limiting_reagent = np.argmin(tmp_solution,1)

        mineral_lim = np.min(tmp_solution,1)

        mineral_lim = np.clip(mineral_lim,0,1)
        mineral_array[:,i] = mineral_lim

    return pd.DataFrame(mineral_array, columns=target_minerals),pos_element[limiting_reagent]

def solve(assays:pd.DataFrame, mineral_matrix:pd.DataFrame,mineral_order:pd.DataFrame,upper_limit:float =1)->pd.DataFrame:
    '''
    Pass the assays, weights of the minerals and the order that you would like to solve them in and the so

    '''
    # with the assays passed in we will now try and extract the columns that represent the oxides or elements and the loi
    # we also assume that the assays are only assays and not metadata etc
    # we then will ensure that we only fit elements where we have them in both the assays and the solution matrix

    # convert all the assays to elements and deal with their oxides.

    x:pd.DataFrame = assays.copy()
    analyte_columns = x.columns.to_list()
    # get the oxide factors to convert the oxides to elements
    # ignore things like Fe3O4 and LOI and give them a factor of 1
    oxide_factors =[]
    for a in analyte_columns:
        if (a =='Fe3O4') or a.startswith('LOI'):
            if a.startswith('LOI'):
                outname = a.lower()
            else:
                outname = a
            tmp = (a, outname,1.0)

        else:
            tmp = ox_factor(a) 
        oxide_factors.append(tmp) 
    # convert this list into an array so that we can multiply the factors out
    elements = x*np.stack([c[2] for c in oxide_factors])
    elements.columns = [c[1] for c in oxide_factors]

    assay_elements:list[str] = elements.columns.to_list()
    mineral_elements:list[str] = mineral_matrix.columns.to_list()

    intersect_elements:list[str] = list(set(assay_elements).intersection(mineral_elements))
    # sort the list of elements
    intersect_elements= [i for i in assay_elements if i in intersect_elements]

    initial_solution:pd.DataFrame = pd.DataFrame(np.zeros((x.shape[0],len(mineral_order))), columns=mineral_order.mineral)
    limiting_reagent:pd.DataFrame = pd.DataFrame(np.zeros((x.shape[0],len(mineral_order))), columns=mineral_order.mineral)

    running_total = elements.loc[:,intersect_elements].copy()

    for _,i in mineral_order.iterrows():
        # add some extra contraints as we go
        # 1. the upper bound cannot be more than the total
        # 2. start reducing the remaining reactants
        try:
            if i.mineral == 'magnetite':
                tmp = assays['Fe3O4']*0.01
                lim  = 0
            else:
                upper_bound = upper_limit-initial_solution.sum(1)
                if i.unconstrained:
                    tmp,lim = normative_calc(elements[intersect_elements], mineral_matrix.loc[:,intersect_elements],[i.mineral])
                else:
                    tmp,lim = normative_calc(running_total[intersect_elements], mineral_matrix.loc[:,intersect_elements],[i.mineral])
                idxub = tmp.values.ravel()>upper_bound.values.ravel()
                if any(idxub) and i.apply_upper_bound:
                    tmp[idxub] = upper_bound[idxub].values.reshape(-1,1)
        except KeyError:
            tmp = pd.DataFrame(np.zeros((assays.shape[0],1)), columns=[i.mineral])
        tmp[tmp.isna()]=0
        limiting_reagent[i.mineral]=lim
        initial_solution[i.mineral] = tmp
        idx_current_mineral = mineral_matrix.index == i.mineral
        # calculate the matrix to multiply the concentration with we also need to ensure that there are no nan values
        tmp_matrix = mineral_matrix.loc[idx_current_mineral,intersect_elements].values.copy()
        tmp_matrix[np.isnan(tmp_matrix)]=0
        running_total -= ((tmp.values.reshape(-1,1)@tmp_matrix))*100

    return initial_solution,limiting_reagent, intersect_elements