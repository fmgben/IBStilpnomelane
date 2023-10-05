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
pd.concat([minerals.mineral,solution_matrix],axis=1)
solution_matrix[solution_matrix.isna()]=0


assays = pd.read_csv(r'reference/BV_XRD_Assay_Merge.csv')
assays.columns =[c.replace(' ','_') for c in assays.columns]

# compiled xrd is missing 3pt xrf
from sklearn import linear_model
# old vs new assays columns
v1 = False
if v1:
    assay_columns = ['Fe_%', 'SiO2_%', 'Al2O3_%', 'TiO2_%', 'Mn_%', 'CaO_%', 'P XRF_%',
                    'S XRF_%', 'MgO_%', 'K2O_%', 'Na2O_%', 'Zn_%', 'As_%', 'Cl_%', 'Cu_%',
                    'Pb_%', 'Ba_%', 'V_%', 'Cr_%', 'Ni_%', 'Co_%', 'Sn XRF_%', 'Zr_%',
                    'Sr_%', 'Fe3O4_%','LOI_ Total_%']
    loi_columns = ['LOI_371_%', 'LOI_650_%', 'LOI_1000_%']
else:
    assay_columns = ['Fe3O4', 'Fe', 'SiO2', 'Al2O3', 'TiO2', 'Mn', 'CaO', 'P_XRF', 'S_XRF', 'MgO', 'K2O', 'Na2O', 'Zn', 'As', 'Cl', 'Cu', 'Pb', 'Ba', 'V', 'Cr', 'Ni', 'Co', 'Sn_XRF', 'Zr', 'Sr']
    loi_columns = ['LOI371', 'LOI650', 'LOI1000']
idx = np.all(~assays[assay_columns].isna(),1)
assays= assays[idx].reset_index()
assay_ok = assays[loi_columns].isna().sum(1) == 0
loi_model = linear_model.LinearRegression().fit(assays.loc[assay_ok,assay_columns], assays.loc[assay_ok,loi_columns])
loi_hat = loi_model.predict(assays.loc[:,assay_columns])
# predict the loi into the missing

assays.loc[~assay_ok,loi_columns] = loi_hat[~assay_ok]
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
solution_matrix[idx_mineral]
minerals[idx_mineral]
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
# first pass at getting the bounds we are going to extract the single elements constraints and Magnetite first
solution_matrix.to_clipboard()
all_minerals,_= normative(x, solution_matrix.loc[:,selected_columns], minerals.mineral, minerals)

plt.plot(all_minerals[group_minerals['stilpnomelane']]*100,assays['Stilpnomelane'],'.' )
plt.plot([0,100],[0,100])
plt.show()

for i in mineral_pass_data.xrd.unique():
    if isinstance(i,str):
        idxmin = mineral_pass_data.xrd == i
        tmp_minerals = mineral_pass_data[idxmin].mineral.to_list()
        midx = all_minerals.columns.isin(tmp_minerals)
        tmpy = all_minerals.loc[:,midx].sum(1)*100
        tmpx = assays.loc[:,i]
        plt.figure()
        plt.plot(tmpx, tmpy,'.')
        m0 = np.nanmin([np.nanmin(tmpx),np.nanmin(tmpy)])
        m1 = np.nanmax([np.nanmax(tmpx),np.nanmax(tmpy)])
        plt.plot([m0,m1],[m0,m1],'k')
        plt.xlabel('XRD')
        plt.ylabel('Solver')
        plt.title(i)
plt.show()

# get the limiting calculatoin for the first pass minerals
first_pass_solution,_ = normative(x, solution_matrix.loc[:,selected_columns], first_pass_minerals, minerals)
# for the grouped minerals calculate the limiting reaction individually
# then calculate a 
plt.plot(first_pass_solution['rutile']*100, assays['Rutile_group'],'.')
plt.plot(first_pass_solution['sepiolite']*100, assays['Sepiolite'],'.')
plt.show()

tmp,_ = normative(x, solution_matrix.loc[:,selected_columns],['clinochlore'],minerals)
plt.plot( assays[['Chlorite_group']].sum(1),tmp.sum(1)*100,'.',label='clino')
tmp,_ = normative(x, solution_matrix.loc[:,selected_columns],['chamosite'],minerals) # apparently this is the one
plt.plot(assays[['Chlorite_group']].sum(1),tmp.sum(1)*100,'.',label='chamo')
plt.legend()
plt.plot([0,20],[0,20],'k')
plt.show()
for n,i in enumerate(loi.columns):
    plt.subplot(1, 3, n+1)
    plt.plot(assays[['Stilpnomelane']].sum(1),loi[i],'.',label=i)
plt.legend()
plt.show()

# this returns the total limited group solution
# we probaly need to relimit the values again

tmp,_ = normative(x,  solution_matrix.loc[:,selected_columns],['albite'],minerals)
plt.scatter(assays[['Plagioclase']].sum(1),tmp.sum(1)*100,c=pd.Categorical(assays['Type']).codes)
plt.plot([0,100],[0,100],'k')
plt.show()

tmp,_ = normative(x,  solution_matrix.loc[:,selected_columns],['orthoclase'], minerals)
plt.scatter(assays[['K-Feldspar']].sum(1),tmp.sum(1)*100,c=pd.Categorical(assays['Type']).codes)
plt.plot([0,100],[0,100],'k')
plt.show()

tmp,_ = normative(x, solution_matrix.loc[:,selected_columns],['hematite'],minerals)
plt.scatter(assays[['Hematite_group']].sum(1),tmp.sum(1)*100,c=pd.Categorical(assays['Type']).codes)
plt.plot([0,100],[0,100],'k')
plt.show()

tmp,_ = normative(x, solution_matrix.loc[:,selected_columns],['quartz'],minerals)
plt.scatter(assays[['Quartz']].sum(1),tmp.sum(1)*100,c=pd.Categorical(assays['Type']).codes)
plt.plot([0,100],[0,100],'k')
plt.show()

tmp,_ = normative(x,  solution_matrix.loc[:,selected_columns],group_minerals['chlorite'],minerals)
plt.scatter(assays[['Chlorite_group']].sum(1),tmp.sum(1)*100,c=pd.Categorical(assays['Type']).codes)
plt.plot([0,100],[0,100],'k')
plt.show()
tmp,_ = normative(x,  solution_matrix.loc[:,selected_columns],group_minerals['stilpnomelane'],minerals)
plt.scatter(assays[['Stilpnomelane']].sum(1),tmp.sum(1)*100,c=pd.Categorical(assays['Type']).codes)
plt.plot([0,100],[0,100],'k')
plt.show()

# solution order, magnetite, firstpass minerals, pyrite, orthoclase, albite, [stilp, chlor] unknown qtz, hem, goe
carbonates = group_minerals['carbonate']
mineral_order = ['magnetite',*group_minerals['chlorite'], *group_minerals['stilpnomelane'],*first_pass_minerals, 'pyrite', 'orthoclase',*carbonates , 'quartz','goethite','hematite']
carbonates = ['mg-ankerite', 'fe-ankerite','siderite']

fpm = ['titanite','apatite', 'chalcopyrite', 'chromite', 'zircon','rhodochrosite']

mineral_order = ['magnetite',*fpm, 'calcite','fe-ankerite','mg-ankerite','siderite','sepiolite',*['ferrostilpnomelane']]

# from original report
mineral_order = ['magnetite','chalcopyrite','siderite','pyrite','chromite','zircon', 'rhodochrosite','apatite', 'albite','ferristilpnomelane', 'orthoclase','quartz']

# what I think
siderite = ['siderite']
carbonates = ['mg-ankerite','fe-ankerite','calcite']
# slightly better order
stilp = [ 'ferrostilpnomelane','ferristilpnomelane']

stilp = [ 'ferristilpnomelane','ferrostilpnomelane']
chlorite = ['chamosite','clinochlore']
chlorite = ['clinochlore','chamosite']

mineral_order = ['magnetite',*first_pass_minerals,*siderite,*stilp,*chlorite,*carbonates,'goethite','orthoclase','quartz','hematite']

initial_solution = pd.DataFrame(np.zeros((x.shape[0],len(mineral_order))), columns=mineral_order)
index = []
for i in initial_solution.columns.to_list():
    index.append(np.where(minerals.mineral ==i)[0][0])

running_total = final_assays[selected_columns].copy()
# add some constraints on the assay value
i = 'albite'
for i in mineral_order:
    if i == 'magnetite':
        tmp = assays['Fe3O4']*0.01
        lim  = 0
    else:
        upper_bound = 1-initial_solution.sum(1)
        tmp,lim = normative(running_total, solution_matrix.loc[:,selected_columns],[i],minerals)
        idxub = tmp.values.ravel()>upper_bound.values.ravel()
        if any(idxub):
            tmp[idxub] = upper_bound[idxub].values.reshape(-1,1)
    #print(selected_columns[lim])
    initial_solution[i] = tmp
    idx_current_mineral = minerals.mineral == i
    idx_col = solution_matrix.loc[idx_current_mineral,selected_columns]!=0
    running_total -= ((initial_solution[i].values.reshape(-1,1)@solution_matrix.loc[idx_current_mineral,selected_columns].values))*100
    # add some extra contraints as we go
    # 1. the upper bound cannot be more than the total
    # 2. start reducing the remaining reactants



both = pd.concat([assays, initial_solution*100],axis=1)
i ='Spinel_group'
mm =["Hematite_group",'Quartz','K-Feldspar','Plagioclase','Calcite_group-Siderite','Dolomite_group','Stilpnomelane','Chlorite_group','Spinel_group','Sepiolite','Mica_group','Calcite_group']
#mm= ['Stilpnomelane']
for n,i in enumerate(mm):
    if isinstance(i,str):
        idxmin = mineral_pass_data.xrd == i
        tmp_minerals = mineral_pass_data[idxmin].mineral.to_list()
        midx = both.columns.isin(tmp_minerals)
        tmpy = both.loc[:,midx].sum(1)
        tmpx = both.loc[:,i]
        plt.subplot(4, 3, n+1)
        plt.plot(tmpx, tmpy,'.')
        m0 = np.nanmin([np.nanmin(tmpx),np.nanmin(tmpy)])
        m1 = np.nanmax([np.nanmax(tmpx),np.nanmax(tmpy)])
        plt.plot([m0,m1],[m0,m1],'k')
        plt.xlabel('XRD')
        plt.ylabel('Solver')
        plt.title(i)
plt.show()


# the initial solution represents the maximum value that a mineral can we can now solve for a smaller set of minerals
# we will now pass the initial solution to a solved as 
# additionally some of the solved minerals are very close to the actual result
# these minerals are magnetite, which we measure directly and the minerals that totally consume a single reagent
# we are going to fix these in the solver
fixed_minerals = ['magnetite', *first_pass_minerals]
# other minerals are quite close to being correct albite and orthoclase for example we only allow a small tolerance on the optimiser
tight_solutions = ['albite', 'orthoclase']
# grouped minerals we run an intial factoring to ensure that they start inside the bounds
for g in group_minerals:
    if g != 'ankerite':
        tmp_group = group_minerals[g]
    if any(initial_solution.columns.isin(tmp_group)):
        idx_solve1 = minerals.mineral.isin(initial_solution[tmp_group])
        estimated_assays = initial_solution.loc[:, tmp_group].values@solution_matrix.loc[idx_solve1,selected_columns]
        factor = (estimated_assays*100/final_assays.loc[:,selected_columns]).max(1)
        initial_solution.loc[:,tmp_group] = initial_solution.loc[:,tmp_group]/factor.values.reshape(-1,1)



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
    assay_total_error = np.abs(X.sum()-np.sum(yw))
    # ensure that weights must sum to 100
    weights_error = np.abs(100-np.sum(w))
    loss = np.concatenate([analyte_error,penalty,[weights_error,assay_total_error]])
    return loss



# pass 0 is to get the initial bounds of the solution
# extract everything into numpy arrays
x = final_assays[selected_columns].values

y = solution_matrix.loc[index,selected_columns].values
tmp = solution_matrix.loc[index,selected_columns]
w = np.zeros((y.shape[0],1))
upper = np.ones((y.shape[0]))*100
lower = np.zeros((y.shape[0]))

bounds = (lower, upper)

mineral_solution = initial_solution.copy()
mineral_solution[mineral_solution.isna()]= 0
lower_bound = mineral_solution*100
upper_bound = mineral_solution*100

for i in mineral_solution.columns.to_list():
    gotozero = False
    if i in fixed_minerals:
        factor = 1e-1
    elif i in tight_solutions:
        factor = 1
    else:
        factor = 2
        gotozero = True
    lower_bound[i] = np.clip(lower_bound[i]*(1-factor),0,100)
    upper_bound[i] = np.clip(upper_bound[i]*(1+factor),0,upper_bound[i])


# reclip the factors

complexity_penalty = 0
solved = []
mineral_solution.loc[i]
for i,v in enumerate(x):
    lb,ub = lower_bound.loc[i].values.copy(),upper_bound.loc[i].values.copy()
    ub[lb>=ub]+=lb[lb>=ub]+0.1

    x0 = (ub+lb)/2
    ww = least_squares(lambda w: solver(v,y,w,complexity_penalty, docomplex=False),x0,bounds=(lb, ub))
    solved.append(ww.x)

sols = pd.DataFrame(np.stack(solved),columns=initial_solution.columns)


stilp = [i for i  in sols.columns if i.find('stil')>=0]

sols[stilp]
both = pd.concat([assays, sols],axis=1)
i ='Spinel_group'
for i in mineral_pass_data.xrd.unique():
    if isinstance(i,str):
        idxmin = mineral_pass_data.xrd == i
        tmp_minerals = mineral_pass_data[idxmin].mineral.to_list()
        midx = both.columns.isin(tmp_minerals)
        tmpy = both.loc[:,midx].sum(1)
        tmpx = both.loc[:,i]
        plt.figure()
        plt.plot(tmpx, tmpy,'.')
        m0 = np.nanmin([np.nanmin(tmpx),np.nanmin(tmpy)])
        m1 = np.nanmax([np.nanmax(tmpx),np.nanmax(tmpy)])
        plt.plot([m0,m1],[m0,m1],'k')
        plt.xlabel('XRD')
        plt.ylabel('Solver')
        plt.title(i)
plt.show()



