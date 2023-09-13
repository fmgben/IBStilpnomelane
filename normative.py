


#List of gram per mole of element
Fe  = 55.845
SiO2 = 60.0843
Al2O3 = 101.9612
Mn = 54.938
CaO = 56.0774
P = 30.973
S = 32.065
MgO = 40.3044
Mg = 24.305
K2O = 94.196
Na2O = 61.97894
BaO = 153.326




# Gram per mole of each Minerals
# Difference source may showing different numbers
Magnetite = 231.5326
Apatite = 502.3012
Ankerite = 215.9194
Siderite = 122.503
Stilpnomelane = 1100.185
Chlorite = 468.2073
Kfeldspar = 278.3315
Albite = 262.223
Quartz = 60.0843
Pyrite = 119.975





#molar calculation
bm_data1['fe_mole'] = bm_data1['i_fe']/Fe
bm_data1['sio2_mole'] = bm_data1['i_si']/SiO2
bm_data1['al2o3_mole'] = bm_data1['i_al']/Al2O3
bm_data1['mn_mole'] = bm_data1['i_mn']/Mn
bm_data1['cao_mole'] = bm_data1['i_cao']/CaO
bm_data1['p_mole'] = bm_data1['i_p']/P
bm_data1['s_mole'] = bm_data1['i_s']/S
bm_data1['mgo_mole'] = bm_data1['i_mgo']/MgO
bm_data1['k2o_mole'] = bm_data1['i_k2o']/K2O
bm_data1['na2o_mole'] = bm_data1['i_na2o']/Na2O


# Stage 2 calculation where the element can be allocated to certain mineral
bm_data1['apatite_mole'] = bm_data1['p_mole']/3
bm_data1['pyrite_mole'] = bm_data1['s_mole']/2
Albite_SiO2mole = pd.DataFrame(bm_data1[['sio2_mole']]/3)
Albite_Al2O3mole = pd.DataFrame(bm_data1[['al2o3_mole']]/0.5)
Albite_Na2Omole = pd.DataFrame(bm_data1[['na2o_mole']]/0.5)

test = [Albite_SiO2mole,Albite_Al2O3mole,Albite_Na2Omole]
df=pd.concat(test,axis=1)
bm_data1['albite_mole']=df.min(axis=1)
# bm_data1['albite_mole'] = bm_data1.eval("min(sio2_mole / 3, al2o3_mole / 0.5, na2o_mole)")
bm_data1.loc[bm_data1['albite_mole'] < 0, 'albite_mole'] = 0

bm_data1['siderite_mole'] = bm_data1['mn_mole']/0.095 #based on inverse of slope between Mn mole and Siderite mole
bm_data1['ankerite_mole'] = (bm_data1['cao_mole']-5*bm_data1['apatite_mole'])/0.832 # inverse slope between CaO mole and Ankerite mole
bm_data1['kfeldspar_mole'] = bm_data1['k2o_mole']/0.5
bm_data1.loc[bm_data1['siderite_mole'] < 0, 'siderite_mole'] = 0
bm_data1.loc[bm_data1['ankerite_mole'] < 0, 'ankerite_mole'] = 0
bm_data1.loc[bm_data1['kfeldspar_mole'] < 0, 'kfeldspar_mole'] = 0


# In[9]:


# Stage 3
Chlorite_al2o3mole =  (bm_data1.al2o3_mole - bm_data1.kfeldspar_mole*0.5 - bm_data1.albite_mole*0.5)/1.104 # inverse slope Al2O3 and Chlorite mole

test2 = [Chlorite_al2o3mole]
df2=pd.concat(test2,axis=1)


bm_data1['chlorite_mole']=df2.min(axis=1)
bm_data1.loc[bm_data1['chlorite_mole'] < 0, 'chlorite_mole'] = 0

Stilp_Femole = (bm_data1.fe_mole - bm_data1.pyrite_mole - bm_data1.siderite_mole - bm_data1.ankerite_mole - bm_data1.chlorite_mole)/5
Stilp_SiO2mole = (bm_data1.sio2_mole - bm_data1.albite_mole*3 - bm_data1.kfeldspar_mole*3 - bm_data1.chlorite_mole*3)/6
Stilp_Al2O3mole = (bm_data1.al2o3_mole - bm_data1.albite_mole*0.5 - bm_data1.kfeldspar_mole*0.5)/4


test1 = [Stilp_Femole,Stilp_SiO2mole,Stilp_Al2O3mole]
df1=pd.concat(test1,axis=1)

bm_data1['stilpnomelane_mole']= df1.min(axis=1)
bm_data1.loc[bm_data1['stilpnomelane_mole'] < 0, 'stilpnomelane_mole'] = 0


# In[10]:


# Stage Quartz - from the available SiO2
bm_data1['quartz_mole'] = (bm_data1.sio2_mole - bm_data1.albite_mole*3 - bm_data1.stilpnomelane_mole*8 - bm_data1.chlorite_mole*3 - bm_data1.kfeldspar_mole*3)
bm_data1.loc[bm_data1['quartz_mole'] < 0, 'quartz_mole'] = 0
bm_data1.loc[bm_data1['quartz_mole'] > bm_data1['sio2_mole'], 'quartz_mole'] = bm_data1['sio2_mole']


# In[13]:


# Calculate mass of minerals from calculated mole
bm_data1['apatite_mass'] = bm_data1.apatite_mole*Apatite
bm_data1['pyrite_mass'] = bm_data1.pyrite_mole*Pyrite
bm_data1['albite_mass'] = bm_data1.albite_mole*Albite
bm_data1['siderite_mass'] = bm_data1.siderite_mole*Siderite
bm_data1['ankerite_mass'] = bm_data1.ankerite_mole*Ankerite
bm_data1['stilpnomelane_mass'] = bm_data1.stilpnomelane_mole*Stilpnomelane
bm_data1['chlorite_mass'] = bm_data1.chlorite_mole*Chlorite
bm_data1['kfeldspar_mass'] = bm_data1.kfeldspar_mole*Kfeldspar
bm_data1['quartz_mass'] = bm_data1.quartz_mole*Quartz


# In[14]:


# Calculate total mass - Remember that Magnetite is nor part of this calculation. Next calculation there will be magnetite factor for final percent
bm_data1['total_min_mass'] = bm_data1['apatite_mass']+ bm_data1['albite_mass']+ bm_data1['siderite_mass'] + bm_data1['ankerite_mass'] + bm_data1['stilpnomelane_mass'] + bm_data1['chlorite_mass'] + bm_data1['kfeldspar_mass'] + bm_data1['quartz_mass'] + bm_data1['pyrite_mass']


# In[15]:


# There is a fe3o4 factor as a Magnetite percent
bm_data1['mp_apatite'] = bm_data1['apatite_mass']*(100-bm_data1['i_fe3o4'])/(bm_data1['total_min_mass'])
bm_data1['mp_pyrite'] = bm_data1['pyrite_mass']*(100-bm_data1['i_fe3o4'])/(bm_data1['total_min_mass'])
bm_data1['mp_albite'] = bm_data1['albite_mass']*(100-bm_data1['i_fe3o4'])/(bm_data1['total_min_mass'])
bm_data1['mp_siderite'] = bm_data1['siderite_mass']*(100-bm_data1['i_fe3o4'])/(bm_data1['total_min_mass'])
bm_data1['mp_ankerite'] = bm_data1['ankerite_mass']*(100-bm_data1['i_fe3o4'])/(bm_data1['total_min_mass'])
bm_data1['mp_stilpnomelane'] = bm_data1['stilpnomelane_mass']*(100-bm_data1['i_fe3o4'])/(bm_data1['total_min_mass'])
bm_data1['mp_chlorite'] = bm_data1['chlorite_mass']*(100-bm_data1['i_fe3o4'])/(bm_data1['total_min_mass'])
bm_data1['mp_kfeldspar'] = bm_data1['kfeldspar_mass']*(100-bm_data1['i_fe3o4'])/(bm_data1['total_min_mass'])
bm_data1['mp_quartz'] = bm_data1['quartz_mass']*(100-bm_data1['i_fe3o4'])/(bm_data1['total_min_mass'])
bm_data1['mp_magnetite'] = bm_data1['i_fe3o4']


# In[ ]:


#bm_data1.to_csv('name of csv file - example: bm_data1.csv') # then import mineral to Vulcan Block Model based on Centroid


# In[16]:


#Fresh Cluster (DOES NOT INCLUDE KANGAROO CAVE)
bm_data_fresh = bm_data1.loc[(bm_data1['oxide'] != 1) & (bm_data1['domain'] > 1)]    # (bm_data1['domain'] != 1)]
# bm_data_fresh = bm_data_fresh[bm_data_fresh['domain'] > 1]
fresh_pred_bm = 0.984*bm_data_fresh['mp_magnetite']*(bm_data_fresh['mp_kfeldspar']+bm_data_fresh['mp_albite'])+92
bm_data_fresh['ucs_pred'] = fresh_pred_bm


# In[17]:


# Oxide Cluster
bm_data_cox = bm_data1.loc[bm_data1['oxide'] == 1]
ycox_pred_bm = 14.186*np.power((bm_data_cox['mp_albite']/bm_data_cox['mp_siderite']), -0.51)
bm_data_cox['ucs_pred'] = ycox_pred_bm


# In[18]:


# Kang Cave Cluster
bm_data_cav = bm_data1.loc[(bm_data1['oxide'] != 1) & (bm_data1['domain'] == 1)]
ycav_pred_bm = 100.96*np.power((bm_data_cav['mp_pyrite']/bm_data_cav['mp_albite']), 0.223)
bm_data_cav['ucs_pred'] = ycav_pred_bm


# In[19]:

bm_data2 = pd.concat([bm_data_cav,bm_data_cox,bm_data_fresh])


# In[ ]:


# bm_data2.to_csv('name of csv file - example: bm_data1.csv') #then import ucs to Vulcan block model based on Centroid

out_model = TableModel.from_pandas(bm_data2)
model = handle.create_model("Table", "out table")
out_model.write(model.model_path)
output_set.append_model("Output Models", model)