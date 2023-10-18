import pandas as pd
import numpy as np
from chempy.util import periodic

# we extract from the chempy the mapping between periodic number and the elemental name
num2symbol = {p+1:n for p, n in enumerate(periodic.symbols)}

def normative(x:pd.DataFrame, solution_matrix:pd.DataFrame, target_minerals:list[str],factor:float=100)->pd.DataFrame:
    '''
    Calculate from the assays how much of the mineral can exist given the assay, for each element that we 
    have in both the minerals and the assay we divide the assay concentration by the mineral concentration
    the minimum value for all elements is then taken as the mineral proportion as this is the limiting reagent.
    '''
    # copy the dataframe
    tmpx = x.values.copy()
    # find any assay values that are negative
    xneg = tmpx<0
    # if the value is negative replace it with the half of the value
    # which is why the *-0.5 is there 
    tmpx[xneg] = tmpx[xneg]*-0.5
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
        limiting_reagent = np.argmin(tmp_solution,1)
        mineral_lim = np.min(tmp_solution,1)

        mineral_lim = np.clip(mineral_lim,0,1)
        mineral_array[:,i] = mineral_lim
    return pd.DataFrame(mineral_array, columns=target_minerals),limiting_reagent

