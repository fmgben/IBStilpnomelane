import pandas as pd
import pyodbc

'SELECT * FROM [acQuireMining_DATASETS].[dbo].[VW_IB_DTRAssays]'
import pyodbc
import re
import pandas as pd

drivers = pyodbc.drivers()
# get all the drivers that contain sql  server as these are the 
# ones that you will need to connect to the geodatabase

sql_drivers = [i for i in drivers if i.find('SQL Server')>=0]
reg_version = re.compile('[0-9]{1,2}?[.0-9]{1,3}')
driver = []
for s in sql_drivers:
    matches = reg_version.findall(s)
    if len(matches)>0:
        maj_version = matches[0].split('.')[0]

    else:
        maj_version = 0
    driver.append({'name':s, 'version':int(maj_version)})

driver_information = pd.DataFrame(driver)
# sort the table of drivers so that we can choose the latest.
driver_information = driver_information.sort_values('version', ascending=False).reset_index(drop=True)
# extract the latest driver
current_driver = driver_information.loc[0, 'name']

server = 'PRD-SQL-acQuire_Mining_DATASETS.FMG.local\REP' 
database = 'acQuireMining_DATASETS' 

conn_str = 'DRIVER={'+current_driver+'};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes;'
# ENCRYPT defaults to yes starting in ODBC Driver 18. It's good to always specify ENCRYPT=yes on the client side to avoid MITM attacks.

conn = pyodbc.connect(conn_str)
# read all the assay data for CB
sql_query_assay = 'SELECT * FROM [acQuireMining_DATASETS].[dbo].[VW_IB_HeadAssays]'
data = pd.read_sql(sql_query_assay,conn)


from matplotlib import pyplot as plt
import numpy as np
mwd_collars = pd.read_csv('data/mwd_collars.csv', low_memory=False)
mwd_data = pd.read_csv('data/mwd.csv', low_memory=False)
mwd_data.shape

idx_mwd = (mwd_collars.EASTING>712000) & (mwd_collars.EASTING<714500) &(mwd_collars.NORTHING<7651000) & (mwd_collars.NORTHING>7647000) & (mwd_collars.ELEVATION>1300)

u_mwd_bhid = mwd_collars[idx_mwd].HOLE_ID.unique()
idx_mwd_data= mwd_data.HOLE_ID.isin(u_mwd_bhid)

mwd_data = mwd_data[idx_mwd_data].merge(mwd_collars[['HOLE_ID','EASTING','NORTHING','ELEVATION']])
mwd_data['Z'] = (mwd_data['ELEVATION'].values.reshape(-1,1)-mwd_data[['DEPTH_FROM','DEPTH_TO']]).mean(1)
mwd_data[['EASTING','NORTHING','Z', 'PENETRATION_RATE']].to_csv('data/paraview_mwd.csv', index=False)
mwd_columns = ['PENETRATION_RATE',
       'ROTARY_REFERENCE', 'TORQUE', 'PULLDOWN_PRESSURE', 'MAIN_AIR_PRESSURE',
       'WATER_FLOW', 'HOLE_PROFILE', 'PERCUSSION_PRESSURE', 'FEEDER_PRESSURE',
       'DAMPER_PRESSURE', 'ROTATION_PRESSURE','MSE', 'IGS']
mwd_collars.columns
penrate = mwd_data.loc[:,'PENETRATION_RATE'].values
catbhid = pd.Categorical(mwd_data.HOLE_ID)

depth = mwd_data.ELEVATION.values-mwd_data.DEPTH_FROM.values
numbhid = catbhid.codes
catbhid[217]
unumbhid = np.unique(catbhid.codes)
import tqdm
size_penrate = penrate.size
idx_99 = np.floor(size_penrate*0.99).astype(int)
percentile_99 = np.sort(penrate)[idx_99]
tmp_mwd = []
for i in tqdm.tqdm(unumbhid):
    idx = numbhid == i
    zdepth = depth[idx]
    zmin = np.round(np.min(zdepth),1)
    zmax = np.round(np.max(zdepth),1)

    fpen = np.flip(penrate[idx])
    fzd  = np.flip(zdepth)
    # remove any spikes
    idx_top_1m = fzd-zmin<1
    idx_bot_1m = (zmax-fzd)<1
    # remove front spike that occurs in damaged ground remove the top metre
    # and bottom spike due to pulling being counted as pen rate
    idx_bad_pen = fpen<percentile_99
    med_value = np.nanmedian(fpen[~idx_top_1m & idx_bad_pen])

    med_ratio_5 = (fpen/med_value)>3
    idx_bad_top = ~(idx_top_1m & med_ratio_5)
    idx_bad_bot = ~(idx_bot_1m & med_ratio_5)
    idx_final = idx_bad_bot & idx_bad_top & idx_bad_pen
    newz = np.arange(zmin,zmax,0.1)
    if np.any(idx_final):
        new_depth = np.flip(newz)
        test = np.interp(new_depth,  fzd[idx_final],fpen[idx_final])
    tmp_mwd.append(pd.DataFrame({'bhid':np.tile(i, new_depth.shape[0]),'depth':new_depth.ravel(), 'penrate':test.ravel()}))
clean_pen = pd.concat(tmp_mwd)
clean_pen.reset_index(inplace=True,drop=True)
bhid_map = dict( enumerate(catbhid.categories ) )
clean_pen['HOLE_ID'] = clean_pen.bhid.map(bhid_map)
len(clean_pen.HOLE_ID.unique())

clean_pen = clean_pen.merge(mwd_collars[['HOLE_ID','EASTING','NORTHING']])
plt.plot(clean_pen.EASTING, clean_pen.NORTHING, ',')
plt.show()
clean_pen[['bhid','depth','penrate','EASTING', 'NORTHING']].to_csv('data/penrate_clean_para.csv', index=False)
plt.plot(clean_pen.penrate)
plt.plot(mwd_collars['EASTING'], mwd_collars['NORTHING'],'.')
plt.show()
mwd_collars
# remove the 0 depths they are always wrong
plt.hist(mwd_data.DEPTH_FROM,10000)
plt.plot(mwd_data.DEPTH_FROM.values[0:1000],mwd_data.PENETRATION_RATE[0:1000],'.')
plt.show()

from scipy.signal import zoom_fft
plt.plot(np.fft.fft(merge_mwd.PENETRATION_RATE[0:1028].values).real)
plt.show()
from scipy.signal import spectrogram
f, t, Sxx =spectrogram(merge_mwd.PENETRATION_RATE[0:300000])
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
from scipy.signal import medfilt
plt.plot(merge_mwd.PENETRATION_RATE)

def med_spike(x,perc=0.99,w=5):
    xnew = x.copy()
    xsmooth = medfilt(x, w)
    xsort = xnew[np.argsort(xnew)]
    pc99 = np.floor(xnew.size*perc).astype(int)
    max99 = xsort[pc99]
    yidx = x>=max99
    x[yidx] = xsmooth[yidx]
    return x

x= merge_mwd.PENETRATION_RATE.values

merge_mwd.to_csv('data/paraview.csv', index=False)
mwd_data

plt.plot(mwd_collars.EASTING[idx_mwd], mwd_collars.NORTHING[idx_mwd],'.')
plt.plot(data.Best_X[idx], data.Best_Y[idx],'.')
plt.show()

from vedo import Points
xyz= data[['Best_X','Best_Y','Best_Z']]
from matplotlib import cm
import matplotlib as mpl
cmap = mpl.colormaps['viridis']
from sklearn.preprocessing import MinMaxScaler
mfe = MinMaxScaler().fit_transform(data.Fe_pct_BEST.values.reshape(-1,1))
cc = cmap(mfe)
cm.get_cmap('viridis')
xyz['Best_Z'] = xyz['Best_Z']-data.SAMPFROM+(data.SAMPTO-data.SAMPFROM).values/2
xyz.columns= ['X','Y','Z']
pd.concat([data, xyz], axis=1).to_csv('data/paraview_assays.csv', index=False)
Points(xyz[idx]).cmap('viridis',data.Fe_pct_BEST.values[idx], name='Fe').add_scalarbar().show().close()
plt.hist(merge_mwd.MSE, bins=np.linspace(0,8e5, 100))
plt.show()
plt.hist(merge_mwd['ELEVATION'],1000)
plt.show()
Points(merge_mwd[['EASTING', 'NORTHING', 'ELEVATION']]).cmap('viridis',np.clip(merge_mwd.MSE, 0,8e5),name='MSE').add_scalarbar().show().close()
merge_mwd.MSE = np.clip(merge_mwd.MSE, 0,8e5)
merge_mwd[['EASTING', 'NORTHING', 'ELEVATION', 'MSE']].to_csv('data/paraview.csv', index=False)
