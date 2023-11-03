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

def ox_factor(x):
    reagent = Substance.from_formula(x)
    comp = process_composition(reagent.composition)
    mass_fracts = chempy.mass_fractions(comp)
    key = [c for c in mass_fracts.keys() if c != 'O']
    factor = mass_fracts[key[0]]
    return x,key[0], factor

def normative(x, solution_matrix, target_minerals, minerals,factor=100):
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
        limiting_reagent = np.argmin(tmp_solution,1)
        mineral_lim = np.min(tmp_solution,1)

        mineral_lim = np.clip(mineral_lim,0,1)
        mineral_array[:,i] = mineral_lim
    return pd.DataFrame(mineral_array, columns=target_minerals),limiting_reagent


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
        if not isinstance(value[loi],float):
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

solution_matrix = pd.concat([proportions, loi],axis=1).astype(float)
solution_matrix[solution_matrix.isna()]=0
# adjust the magnetite decomposition
# magnetite allocation to loi, 0.14, 0.3, 0.56
idx_mag = minerals.mineral == 'magnetite'
solution_matrix.loc[idx_mag,['loi371','loi650','loi1000']] = solution_matrix.loc[idx_mag, ['loi371','loi650','loi1000']].sum(axis=1).ravel()*[0.14, 0.3, 0.56]
assays = pd.read_csv(r'reference/BV_XRD_Assay_Merge.csv')
assays.columns =[c.replace(' ','_') for c in assays.columns]
assays.rename(columns={'Calcite_group_-_Siderite':'Calcite_group_Siderite'},inplace=True)


# old vs new assays columns
v1 = False
if v1:
    assay_columns = ['Fe_%', 'SiO2_%', 'Al2O3_%', 'TiO2_%', 'Mn_%', 'CaO_%', 'P XRF_%',
                    'S XRF_%', 'MgO_%', 'K2O_%', 'Na2O_%', 'Zn_%', 'As_%', 'Cl_%', 'Cu_%',
                    'Pb_%', 'Ba_%', 'V_%', 'Cr_%', 'Ni_%', 'Co_%', 'Sn XRF_%', 'Zr_%',
                    'Sr_%', 'Fe3O4_%','LOI_ Total_%']
    loi_columns = ['LOI_371_%', 'LOI_650_%', 'LOI_1000_%']
else:
    assay_columns = ['Fe3O4', 'Fe', 'SiO2', 'Al2O3', 'TiO2', 'Mn', 'CaO', 'P', 'S', 'MgO', 'K2O', 'Na2O', 'Zn', 'As', 'Cl', 'Cu', 'Pb', 'Ba', 'V', 'Cr', 'Ni', 'Co', 'Sn', 'Zr', 'Sr']
    loi_columns = ['LOI371', 'LOI650', 'LOI1000']

idx = np.all(~assays[assay_columns].isna(),1)
assays= assays[idx].reset_index()



pd.concat([minerals, solution_matrix*100],axis=1).to_csv('data/mineral_compositions.csv',index=False)

if v1:
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
if v1:
    column_names = {f:clean_name(f) for f in assays.columns.to_list() if f.find('%')>=0}
    clean_assays = assays[[c for c in column_names]].rename(columns=column_names)
    clean_assays.sum(1)
    clean_assays = clean_assays.iloc[:,0:24]
else:
    xrf_assays = [c for c in assay_columns if not c =='Fe3O4']
    clean_assays = assays.loc[:,xrf_assays]
    clean_assays.columns = [c.replace('_XRF','') for c in clean_assays.columns]


factor_names =[ox_factor(c) for c in clean_assays.columns.to_list()]
elements = {f[0]:f[1] for f in factor_names}

factors = [f[2] for f in factor_names]
elemental_assays= clean_assays*factors
elemental_assays = elemental_assays.rename(columns=elements)
elemental_assays[elemental_assays<0]=0
if v1:
    loi = assays[['LOI_371_%', 'LOI_650_%', 'LOI_1000_%']]
    loi = loi.rename(columns={'LOI_371_%':'loi371', 'LOI_650_%':'loi650', 'LOI_1000_%':'loi1000'})
else:
    loi = assays[loi_columns]
    loi.columns = [c.lower() for c in loi_columns]

final_assays = pd.concat([elemental_assays,loi],axis=1) 
final_assays[final_assays.isna()]=0
idx_joint= final_assays.columns.isin(solution_matrix.columns)
selected_columns = final_assays.columns[idx_joint]

# find the minerals that give a unique solution to a 
# single element these will get solved first using normative and we will use them as constraints
# magnetite is dealt with later on

idx_single = (solution_matrix.loc[idx_mineral,selected_columns]!=0).sum()==1
single_elements = selected_columns[idx_single].to_list()
solution_matrix.loc[idx_mineral, single_elements].idxmax(0)
position_unique = solution_matrix.loc[idx_mineral,single_elements].idxmax().values
first_pass_minerals = minerals.mineral[solution_matrix.loc[position_unique,single_elements].idxmax()].to_list()
# carbonates are particularily difficult to solve for as are minerals that have end members
# so we are going to find the carbonate and end member minerals and solve their limiting reagents together i.e. 
# in the case of carbonates we will determine the simplex where they sit
group_minerals = {}
for i in minerals.group.unique():
    if isinstance(i, str):
        idxmineral_group = (minerals.group == i) & idx_mineral 
        group_minerals.update({i:minerals.mineral[idxmineral_group].to_list()})

for i in minerals.endmember.unique():
    if isinstance(i, str):
        idxmineral_group = (minerals.endmember == i) & idx_mineral 
        if idxmineral_group.any():
            group_minerals.update({i:minerals.mineral[idxmineral_group].to_list()})

# extract everything into numpy arrays
x = final_assays[selected_columns]
y = solution_matrix.loc[idx_mineral,selected_columns]


# for the grouped minerals calculate the limiting reaction individually
carbonates = ['fe-ankerite','mg-ankerite','calcite']
stilp = [ 'ferristilpnomelane','ferrostilpnomelane']
chlorite = ['clinochlore']

mineral_order = ['magnetite',*first_pass_minerals,'pyrite','orthoclase','chlorite',*stilp,'siderite',*carbonates[0:2],'sepiolite','magnesite','annite','calcite','goethite','hematite','kaolinite','quartz']
i='stilpnomelane'
from scipy.optimize import least_squares

def solve_mineral(y, w):
    tmp_solution = y/w
    # find the minimum which is the limiting reagent
    mineral_lim = np.min(tmp_solution,1)
    mineral_lim = np.clip(mineral_lim,0,1)*100
    return mineral_lim

new_solution = solution_matrix.copy()
# add some constraints on the assay value
mineral_order = pd.read_csv('reference/solution_order.csv')

for second_pass in range(0,2):
    initial_solution = pd.DataFrame(np.zeros((x.shape[0],len(mineral_order))), columns=mineral_order.mineral)
    index = []
    for i in initial_solution.columns.to_list():
        index.append(np.where(minerals.mineral ==i)[0][0])
    running_total = final_assays[selected_columns].copy()
    for _,i in mineral_order.iterrows():
        # add some extra contraints as we go
        # 1. the upper bound cannot be more than the total
        # 2. start reducing the remaining reactants
        if second_pass == 1:
            tmp_solution_matrix = new_solution.loc[:,selected_columns].copy()    
        else:
            tmp_solution_matrix = solution_matrix.loc[:,selected_columns].copy()

        if i.mineral == 'magnetite':
            tmp = assays['Fe3O4']*0.01
            lim  = 0
        else:
            if i.unconstrained:
                tmp,lim = normative(final_assays[selected_columns],tmp_solution_matrix ,[i.mineral],minerals)
            else:
                tmp,lim = normative(running_total, tmp_solution_matrix,[i.mineral],minerals)
        upper_bound = 1-initial_solution.sum(1)
        idxub = tmp.values.ravel()>upper_bound.values.ravel()
        if any(idxub) and i.apply_upper_bound:
            tmp[idxub] = upper_bound[idxub].values.reshape(-1,1)

        if i.optimise_composition and second_pass == 0:
            idxmin = minerals.mineral == i.mineral
            idxelement = tmp_solution_matrix.loc[idxmin].values>0
            initial_weight = tmp_solution_matrix.loc[idxmin,idxelement.ravel()]
            xrd_target = mineral_pass_data.loc[mineral_pass_data.mineral == i.mineral].xrd.to_list()
            x = assays[xrd_target].copy().values.ravel()

            xidx = ~np.isnan(x)
            if i.unconstrained:
                Y = final_assays.loc[:,initial_weight.columns].values
            else:
                Y = running_total.loc[:,initial_weight.columns].values

            sol = least_squares(lambda w:x[xidx]-solve_mineral(Y[xidx], w*100), initial_weight.values.ravel(),bounds=[0,1])
            print(i.mineral)
            tmp_adjusted = pd.DataFrame(sol.x.reshape(1,-1), columns=initial_weight.columns)
            new_solution.loc[idxmin,initial_weight.columns.to_list()] = tmp_adjusted.values

            plt.plot(x,solve_mineral(Y, sol.x*100),'.')
            plt.plot([0,35],[0,35])
            plt.title(i.mineral)
            plt.show()

        initial_solution[i.mineral] = tmp
        idx_current_mineral = minerals.mineral == i.mineral
        idx_col = tmp_solution_matrix.loc[idx_current_mineral,selected_columns]!=0
        outmatrix= tmp_solution_matrix.copy()

        running_total -= ((initial_solution[i.mineral].values.reshape(-1,1)@outmatrix.loc[idx_current_mineral,selected_columns].values))*100


for t in assays.BHID.unique():
    tidx = assays.BHID == t
    plt.plot(assays[['Stilpnomelane']].sum(1)[tidx],initial_solution[['stilpnomelane']].sum(1)[tidx]*100,'.',label=t)
    plt.plot()
plt.plot([0,40],[0,40])
plt.show()

for t in assays.BHID.unique():
    tidx = assays.BHID == t
    plt.plot(assays[['K-Feldspar']].sum(1)[tidx],initial_solution[['orthoclase']].sum(1)[tidx]*100,'.',label=t)
plt.plot([0,40],[0,40])
plt.show()

for t in assays.BHID.unique():
    tidx = assays.BHID == t
    plt.plot(assays[['Chlorite_group']].sum(1)[tidx],initial_solution[['chlorite2']].sum(1)[tidx]*100,'.',label=t)
    plt.plot([0,20],[0,20])
plt.legend()
plt.show()

for t in assays.BHID.unique():
    tidx = assays.BHID == t
    plt.plot(assays[['Calcite_group_Siderite']].sum(1)[tidx],initial_solution[['siderite']].sum(1)[tidx]*100,'.',label=t)
    plt.plot([0,20],[0,20])
plt.legend()
plt.show()


#initial_solution = initial_solution/initial_solution.sum(1).values.reshape(-1,1)

#initial_solution[cc] = initial_solution[cc]/initial_solution.sum(1).values.reshape(-1,1)
both = pd.concat([assays, initial_solution*100],axis=1)

mm = ['Chlorite_group', 'Quartz',
        'Calcite_group_Siderite',
       'Calcite_group', 'Dolomite_group', 'Hematite_group',
       'Stilpnomelane']

mm = ['Chlorite_group', 'Quartz',
       'Pyroxene', 'Plagioclase', 'K-Feldspar', 'Calcite_group_Siderite',
       'Calcite_group', 'Dolomite_group', 'Spinel_group', 'Hematite_group',
       'Stilpnomelane', 'Mica_group', 'Pyrite', 'Rutile_group', 'Goethite',
       'Smectite_group', 'Sepiolite', 'Kaolinite-serpentine_group']





from sklearn.linear_model import LinearRegression
both.columns
assays.columns
error_proportions = {}
for n,i in enumerate(mm):
    if isinstance(i,str):
        idxmin = mineral_pass_data.xrd == i
        tmp_minerals = mineral_pass_data[idxmin].mineral.to_list()
        midx = both.columns.isin(tmp_minerals)
        tmpy = both.loc[:,midx].sum(1)
        tmpx = both.loc[:,i]
        #plt.subplot(5, 4, n+1)
        plt.figure(figsize=(8,8))
        for t in assays.Type.unique():
            tidx = assays.Type == t
            plt.scatter(tmpx[tidx], tmpy[tidx],label=t)
        idx_na = ~(tmpx.isna() | tmpy.isna())
        rr = LinearRegression().fit(tmpx[idx_na].values.reshape(-1,1), tmpy[idx_na].values.reshape(-1,1))
        error = mean_squared_error(tmpx[idx_na].values.reshape(-1,1),tmpy[idx_na].values.reshape(-1,1),squared=False)
        bias = np.mean(tmpx[idx_na].values.reshape(-1,1)-tmpy[idx_na].values.reshape(-1,1))
        m0 = np.nanmin([np.nanmin(tmpx),np.nanmin(tmpy)])
        m1 = np.nanmax([np.nanmax(tmpx),np.nanmax(tmpy)])
        plt.plot([m0, m1], [(m0*rr.coef_[0])+rr.intercept_[0],(m1*rr.coef_[0])+rr.intercept_[0]])
        plt.plot([m0,m1],[m0,m1],'k')
        plt.xlabel('XRD')
        plt.ylabel('Solver')
        plt.legend()
        plt.title('{}\nRMSE:{:3.3}\nBIAS:{:3.3}'.format(i,error,bias))
        plt.savefig(f'report\{i}.png')
        plt.close()


# pass 0 is to get the initial bounds of the solution
# extract everything into numpy arrays
# not all minerals need to have their weights adjusted,
# magnetite can be set as fixed others are allowed to flex
# first pass minerals are fixed

x = final_assays[selected_columns].values

y = solution_matrix.loc[index,selected_columns].values

mineral_solution = initial_solution.copy()*100
mineral_solution[mineral_solution.isna()]= 0

lower_bound = (mineral_solution)*0.2
upper_bound = (mineral_solution)*1.2

lower_bound['magnetite'] = mineral_solution['magnetite']*0.99
upper_bound['magnetite'] = mineral_solution['magnetite']*1.11

for i in mineral_solution.columns.to_list():

    lower_bound[i] = np.clip(lower_bound[i],0,100)
    upper_bound[i] = np.clip(upper_bound[i],0,100)


def solver(X, y, w,complexity_penalty,docomplex=False):
    w = w.reshape(-1,1)
    yw = y*w

    # penalise the more complex minerals as the solver likes to include them as they
    # are easier to optimise for.
    # a simple total penalty is ok but not perfect as we optimise the global penalty
    # let's try a per element penalty where the idea is that we would like the optimiser to preference
    # simpler minerals per element rather than complex ones
    # ensure that total is 100
    if docomplex:
        penalty = np.sum(w*complexity_penalty,0)
    else:
        penalty = [0]

    # error for each analyte minmise the 
    # absolute value of the error
    analyte_error = np.abs(X-np.sum(yw,0))
    #np.abs(X-np.sum(yw,0))
    #analyte_error/np.abs(X)
    assay_total_error = np.abs(X.sum()-np.sum(yw))
    # ensure that weights must sum to 100
    weights_error = np.abs(100-np.sum(w))
    loss = np.concatenate([analyte_error,penalty,[weights_error,assay_total_error]])
    return loss

# reclip the factors

complexity_penalty = 0

solved = []

for i,v in enumerate(x):
    lb,ub = lower_bound.loc[i].values.copy(),upper_bound.loc[i].values.copy()
    ub[lb>=ub]+=lb[lb>=ub]+0.1
    x0 = mineral_solution.loc[i].values.copy()
    w = x0

    ww = least_squares(lambda w: solver(v,y,w,complexity_penalty, docomplex=False),x0,bounds=(lb, ub))
    solved.append(ww.x)

w = ww.x
sols = pd.DataFrame(np.stack(solved),columns=initial_solution.columns)
sols.sum(1)
initial_solution*100-sols
plt.plot(sols.sum(1))
plt.plot(initial_solution.sum(1)*100,'.')
plt.show()


both = pd.concat([assays, sols],axis=1)

mm =["Hematite_group",'Quartz','K-Feldspar','Plagioclase','Calcite_group-Siderite','Dolomite_group','Stilpnomelane','Chlorite_group','Spinel_group','Pyrite','Calcite_group']

for n,i in enumerate(mm):
    if isinstance(i,str):
        idxmin = mineral_pass_data.xrd == i
        tmp_minerals = mineral_pass_data[idxmin].mineral.to_list()
        midx = both.columns.isin(tmp_minerals)
        tmpy = both.loc[:,midx].sum(1)
        tmpx = both.loc[:,i]
        tmpyy = initial_solution.loc[:,tmp_minerals].sum(1)*100
        idx = ~(tmpyy.isna() | tmpx.isna()) 
        errraw= mean_squared_error(tmpx[idx],tmpyy[idx])
        erropt = mean_squared_error(tmpx[idx],tmpy[idx])

        print('{}\nerror opt: {:3.4} \nerror raw: {:3.4}'.format(i,erropt, errraw))

        plt.subplot(4, 3, n+1)
        plt.plot(tmpx, tmpyy,'.',c='k')
        plt.plot(tmpx, tmpy,'.',c='r')

        m0 = np.nanmin([np.nanmin(tmpx),np.nanmin(tmpy)])
        m1 = np.nanmax([np.nanmax(tmpx),np.nanmax(tmpy)])
        plt.plot([m0,m1],[m0,m1],'k')
        plt.xlabel('XRD')
        plt.ylabel('Solver')
        plt.title(i)
plt.show()

initial_solution.sum(1)
