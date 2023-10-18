import pandas
fe = 55.845
sio2 = 60.0843
al2o3 = 101.9612
mn = 54.938
cao = 56.0774
p = 30.973
s = 32.065
mgo = 40.3044
mg = 24.305
k2o = 94.196
na2o = 61.97894
bao = 153.326

magnetite = 231.5326
apatite = 502.3012
ankerite = 215.9194
siderite = 122.503
stilpnomelane = 1100.185
chlorite = 468.2073
kfeldspar = 278.3315
albite = 262.223
quartz = 60.0843


def normative_v1(dataframe):
    '''
    Code from Nuer no changes to the control flow or the code have been made apart from wrapping the 
    code into a function and adding some very minor formatting of input data 
    '''
    df_joined = dataframe.copy()

    df_joined = [c.lower() for c in dataframe.columns]

    assay_columns = ['fe', 'sio2','al2o3','mn','cao','p','s','mgo','k2o','na2o']

    df_joined = df_joined[assay_columns].copy()

    df_joined['fe_moles'] = df_joined['fe'] / fe
    df_joined['sio2_moles'] = df_joined['sio2'] / sio2
    df_joined['al2o3_moles'] = df_joined['al2o3'] / al2o3
    df_joined['mn_moles'] = df_joined['mn'] / mn
    df_joined['cao_moles'] = df_joined['cao'] / cao
    df_joined['p_moles'] = df_joined['p'] / p
    df_joined['s_moles'] = df_joined['s'] / s
    df_joined['mgo_moles'] = df_joined['mgo'] / mgo
    df_joined['k2o_moles'] = df_joined['k2o'] / k2o
    df_joined['na2o_moles'] = df_joined['na2o'] / na2o



    df_joined['apatite_moles'] = df_joined['p_moles'] / 3
    df_joined['pyrite_moles'] = df_joined['s_moles'] / 2

    df_joined['albite_sio2_moles'] = df_joined['sio2_moles'] / 3
    df_joined['albite_al2o3_moles'] = df_joined['al2o3_moles'] / 0.5
    df_joined['albite_na2o_moles'] = df_joined['na2o_moles'] / 0.5

    df_joined['albite_moles'] = df_joined[['albite_sio2_moles', 'albite_al2o3_moles', 'albite_na2o_moles']].min(axis=1)
    df_joined.loc[df_joined['albite_moles'] < 0, 'albite_moles'] = 0

    df_joined['siderite_moles'] = df_joined['mn_moles'] / 0.095

    df_joined['ankerite_moles'] = (df_joined['cao_moles'] - 5 * df_joined['apatite_moles']) / 0.832

    df_joined['kfeldspar_moles'] = df_joined['k2o_moles'] / 0.5


    df_joined.loc[df_joined['siderite_moles'] < 0, 'siderite_moles'] = 0
    df_joined.loc[df_joined['ankerite_moles'] < 0, 'ankerite_moles'] = 0
    df_joined.loc[df_joined['kfeldspar_moles'] < 0, 'kfeldspar_moles'] = 0



    df_joined['chlorite_moles'] = (df_joined['al2o3_moles'] - df_joined['kfeldspar_moles'] / 2 - df_joined['albite_moles'] / 2) / 1.104

    df_joined.loc[df_joined['chlorite_moles'] < 0, 'chlorite_moles'] = 0

    df_joined['stilp_fe_moles'] = (df_joined['fe_moles'] - df_joined['pyrite_moles'] - df_joined['siderite_moles']  - df_joined['ankerite_moles']  - df_joined['chlorite_moles']) / 5
    df_joined['stilp_sio2_moles'] = (df_joined['sio2_moles'] - df_joined['albite_moles'] * 3 - df_joined['kfeldspar_moles'] * 3 - df_joined['chlorite_moles'] * 3) / 6
    df_joined['stilp_al2o3_moles'] = (df_joined['al2o3_moles'] - df_joined['albite_moles'] / 2 - df_joined['kfeldspar_moles'] / 2) / 4

    df_joined['stilp_moles'] = df_joined[['stilp_fe_moles', 'stilp_sio2_moles', 'stilp_al2o3_moles']].min(axis=1)
    df_joined.loc[df_joined['stilp_moles'] < 0, 'stilp_moles'] = 0

    df_joined['quartz_moles'] = (df_joined['sio2_moles'] - df_joined['albite_moles'] * 3 - df_joined['stilp_moles'] * 6 - df_joined['kfeldspar_moles'] * 3 - df_joined['chlorite_moles'] * 3)
    df_joined.loc[df_joined['quartz_moles'] < 0, 'quartz_moles'] = 0
    df_joined.loc[df_joined['quartz_moles'] > df_joined['sio2_moles'], 'quartz_moles'] = df_joined.loc[df_joined['quartz_moles'] > df_joined['sio2_moles'], 'sio2_moles']


    df_joined['magnetite_moles'] = (df_joined['fe_moles'] - df_joined['stilp_fe_moles'] * 5 - df_joined['pyrite_moles'] - df_joined['siderite_moles']  - df_joined['ankerite_moles']  - df_joined['chlorite_moles']) / 3
    df_joined.loc[df_joined['magnetite_moles'] < 0, 'magnetite_moles'] = 0


    df_joined['apatite_mass'] = df_joined['apatite_moles'] * apatite
    df_joined['ankerite_mass'] = df_joined['ankerite_moles'] * ankerite
    df_joined['siderite_mass'] = df_joined['siderite_moles'] * siderite
    df_joined['stilpnomelane_mass'] = df_joined['stilp_moles'] * stilpnomelane
    df_joined['chlorite_mass'] = df_joined['chlorite_moles'] * chlorite
    df_joined['kfeldspar_mass'] = df_joined['kfeldspar_moles'] * kfeldspar
    df_joined['albite_mass'] = df_joined['albite_moles'] * albite
    df_joined['quartz_mass'] = df_joined['quartz_moles'] * quartz
    df_joined['magnetite_mass'] = 0 # df_joined['magnetite_moles'] * magnetite

    df_joined['total_mass'] = df_joined[['apatite_mass', 'ankerite_mass', 'siderite_mass', 'stilpnomelane_mass', 'chlorite_mass', 'kfeldspar_mass', 'albite_mass', 'quartz_mass', 'magnetite_mass']].sum(axis=1)

    df_joined['apatite_mass_pct'] = df_joined['apatite_mass'] / df_joined['total_mass'] * 100
    df_joined['ankerite_mass_pct'] = df_joined['ankerite_mass'] / df_joined['total_mass'] * 100
    df_joined['siderite_mass_pct'] = df_joined['siderite_mass'] / df_joined['total_mass'] * 100
    df_joined['stilpnomelane_mass_pct'] = df_joined['stilpnomelane_mass'] / df_joined['total_mass'] * 100
    df_joined['chlorite_mass_pct'] = df_joined['chlorite_mass'] / df_joined['total_mass'] * 100
    df_joined['kfeldspar_mass_pct'] = df_joined['kfeldspar_mass'] / df_joined['total_mass'] * 100
    df_joined['albite_mass_pct'] = df_joined['albite_mass'] / df_joined['total_mass'] * 100
    df_joined['quartz_mass_pct'] = df_joined['quartz_mass'] / df_joined['total_mass'] * 100
    df_joined['magnetite_mass_pct'] = df_joined['magnetite_mass'] / df_joined['total_mass'] * 100
    outminerals = ['apatite_mass_pct', 'ankerite_mass_pct', 'siderite_mass_pct', 'stilpnomelane_mass_pct', 'chlorite_mass_pct', 'kfeldspar_mass_pct', 'albite_mass_pct', 'quartz_mass_pct', 'magnetite_mass_pct']
    output = df_joined[outminerals].copy()

    return output



