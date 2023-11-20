import pandas as pd
import numpy as np
from chempy.util import periodic
from matplotlib import pyplot as plt
from ..parser.parser import ox_factor
from scipy.optimize import least_squares
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
        idx_element = (tmpy!=0).ravel()
        tmp_solution = tmpx[:,idx_element]/(tmpy[idx_element]*factor)
        # find the minimum which is the limiting reagent
        pos_element = np.where(idx_element)[0]
        limiting_reagent = np.argmin(tmp_solution,1)

        mineral_lim = np.min(tmp_solution,1)

        mineral_lim = np.clip(mineral_lim,0,1)
        mineral_array[:,i] = mineral_lim

    return pd.DataFrame(mineral_array, columns=target_minerals),pos_element[limiting_reagent]


def solve_mineral(y, w):
    '''
    objective function to optimise the composition of the mineral against the xrd values
    '''
    tmp_solution = y/w
    # find the minimum which is the limiting reagent
    mineral_lim = np.min(tmp_solution,1)
    mineral_lim = np.clip(mineral_lim,0,1)*100

    return mineral_lim

def oxide_to_element(assay:pd.DataFrame):
    x:pd.DataFrame = assay.copy()
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
            try:
                tmp = ox_factor(a)
            except Exception:
                print(a)
        oxide_factors.append(tmp) 
    # convert this list into an array so that we can multiply the factors out
    elements = x*np.stack([c[2] for c in oxide_factors])
    elements.columns = [c[1] for c in oxide_factors]
    return elements

def apply_mineral_matrix(assays, mineral_order, mineral_matrix,upper_limit:float=1):

    # typing
    #limiting_reagent:pd.DataFrame
    #initial_solution:pd.DataFrame
    #running_total:pd.DataFrame
    #elements:pd.DataFrame

    elements = oxide_to_element(assays)

    intersect_elements = find_element_intersection(elements, mineral_matrix)
    limiting_reagent = pd.DataFrame(np.zeros((elements.shape[0],len(mineral_order))), columns=mineral_order.mineral)

    initial_solution = pd.DataFrame(np.zeros((elements.shape[0],len(mineral_order))), columns=mineral_order.mineral)
    running_total = elements.loc[:,intersect_elements].copy()
    for _,i in mineral_order.iterrows():
        # add some extra contraints as we go
        # 1. the upper bound cannot be more than the total
        # 2. start reducing the remaining reactants

        tmp_solution_matrix = mineral_matrix.loc[:,intersect_elements].copy()
        if i.mineral == 'magnetite':
            tmp = assays['Fe3O4']*0.01
            lim  = 0
        else:
            upper_bound = upper_limit-initial_solution.sum(1)
            # the unconstrained flag fits the mineral out of order i.e. no other minerals have been
            # solved before this one this means that there is no assay consumption done.
            if i.unconstrained:
                tmp,lim = normative_calc(elements[intersect_elements], tmp_solution_matrix.loc[:,intersect_elements],[i.mineral])
            else:
                tmp,lim = normative_calc(running_total[intersect_elements], tmp_solution_matrix.loc[:,intersect_elements],[i.mineral])
            idxub = tmp.values.ravel()>upper_bound.values.ravel()

            if any(idxub) and i.apply_upper_bound:
                tmp[idxub] = upper_bound[idxub].values.reshape(-1,1)


        tmp[tmp.isna()]=0
        # cheeky array conversion from list of strings to get the indexing done quickly
        limiting_reagent[i.mineral] = np.asarray(intersect_elements)[lim]
        initial_solution[i.mineral] = tmp

        idx_current_mineral = tmp_solution_matrix.index == i.mineral
        outmatrix= tmp_solution_matrix.copy()
        running_total -= ((initial_solution[i.mineral].values.reshape(-1,1)@outmatrix.loc[idx_current_mineral,intersect_elements].values))*100
        running_total = np.clip(running_total,0,100)


    return initial_solution, limiting_reagent

def find_element_intersection(assay, minerals)->list[str]:
    assay_elements:list[str] = assay.columns.to_list()
    mineral_elements:list[str] = minerals.columns.to_list()

    intersect_elements:list[str] = list(set(assay_elements).intersection(mineral_elements))
    # sort the list of elements
    intersect_elements:list[str]= [i for i in assay_elements if i in intersect_elements]
    return intersect_elements

def solve(assays:pd.DataFrame, minerals:pd.DataFrame,mineral_matrix:pd.DataFrame,mineral_order:pd.DataFrame,mineral_pass_data:pd.DataFrame,upper_limit:float =1)->pd.DataFrame:
    '''
    Pass the assays, weights of the minerals and the order that you would like to solve them in and the so

    '''
    # with the assays passed in we will now try and extract the columns that represent the oxides or elements and the loi
    # we also assume that the assays are only assays and not metadata etc
    # we then will ensure that we only fit elements where we have them in both the assays and the solution matrix

    # convert all the assays to elements and deal with their oxides.

    elements = oxide_to_element(assays)

    intersect_elements = find_element_intersection(elements, mineral_matrix)
    limiting_reagent:pd.DataFrame = pd.DataFrame(np.zeros((elements.shape[0],len(mineral_order))), columns=mineral_order.mineral)
    # remove the nan values
    mineral_matrix.replace(np.nan, 0, inplace=True)
    # run multiple passes over the reference data to get the best weights
    new_mineral_matrix = mineral_matrix.copy()

    initial_solution:pd.DataFrame
    for passes in range(0,10):
        initial_solution = pd.DataFrame(np.zeros((elements.shape[0],len(mineral_order))), columns=mineral_order.mineral)
        running_total = elements.loc[:,intersect_elements].copy()
        for _,i in mineral_order.iterrows():
            # add some extra contraints as we go
            # 1. the upper bound cannot be more than the total
            # 2. start reducing the remaining reactants
            if passes >=0:
                tmp_solution_matrix = new_mineral_matrix.loc[:,intersect_elements].copy()
            else:
                tmp_solution_matrix = mineral_matrix.loc[:,intersect_elements].copy()

            try:
                if i.mineral == 'magnetite':
                    tmp = assays['Fe3O4']*0.01
                    lim  = 0
                else:
                    # the unconstrained flag fits the mineral out of order i.e. no other minerals have been
                    # solved before this one this means that there is no assay consumption done.
                    if i.unconstrained:
                        tmp,lim = normative_calc(elements[intersect_elements], tmp_solution_matrix.loc[:,intersect_elements],[i.mineral])
                    else:
                        tmp,lim = normative_calc(running_total[intersect_elements], tmp_solution_matrix.loc[:,intersect_elements],[i.mineral])
                    upper_bound = upper_limit-initial_solution.sum(1)
                    idxub = tmp.values.ravel()>upper_bound.values.ravel()

                    if any(idxub) and i.apply_upper_bound:
                        tmp[idxub] = upper_bound[idxub].values.reshape(-1,1)

                if i.optimise_composition and (passes >-1):
                    idxmin = tmp_solution_matrix.index == i.mineral
                    idxelement = tmp_solution_matrix.loc[idxmin].values!=0
                    initial_weight = tmp_solution_matrix.loc[idxmin,idxelement.ravel()]
                    xrd_target = mineral_pass_data.loc[mineral_pass_data.mineral == i.mineral].xrd.to_list()

                    x = minerals[xrd_target].copy().values.ravel()

                    xidx = ~np.isnan(x) & (x>0)
                    if i.unconstrained:
                        Y = elements.loc[:,initial_weight.columns].values
                    else:
                        Y = running_total.loc[:,initial_weight.columns].values
                    final_idx = xidx.ravel()

                    sol = least_squares(lambda w:x[final_idx]-solve_mineral(Y[final_idx], w*100), initial_weight.values.ravel(),bounds=[0,1])
                    #w = sol.x
                    tmp_adjusted = pd.DataFrame(sol.x.reshape(1,-1), columns=initial_weight.columns)
                    new_mineral_matrix.loc[idxmin,initial_weight.columns.to_list()] = tmp_adjusted.values

            except KeyError:
                tmp = pd.DataFrame(np.zeros((assays.shape[0],1)), columns=[i.mineral])
            tmp[tmp.isna()]=0
            limiting_reagent[i.mineral] = lim
            initial_solution[i.mineral] = tmp

            idx_current_mineral = tmp_solution_matrix.index == i.mineral
            outmatrix= tmp_solution_matrix.copy()
            running_total -= ((initial_solution[i.mineral].values.reshape(-1,1)@outmatrix.loc[idx_current_mineral,intersect_elements].values))*100
            running_total = np.clip(running_total,0,100)

    return initial_solution,limiting_reagent, intersect_elements, tmp_solution_matrix