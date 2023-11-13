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
sql_query_assay = 'SELECT * FROM [acQuireMining_DATASETS].dbo.AT_IB_BestAssaysPivot'
data = pd.read_sql(sql_query_assay,conn)
data.to_csv('data/ib_all.csv')
data = pd.read_csv('data/ib_all.csv')
sql_query_assay = 'SELECT * FROM [acQuireMining_DATASETS].[dbo].[AT_IB_BestAssaysPivot_DTR]'
data = pd.read_sql(sql_query_assay,conn)
data.to_csv('data/ib_dtr.csv')

import pyodbc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

srv = "wn74261.ap-southeast-2.snowflakecomputing.com"
uid = "benjamin.chi@fmgl.com.au"
pw = ""


db = "AA_OPERATIONS_MANAGEMENT"

conn_string = f"Driver={{SnowflakeDSIIDriver}}; Server={srv}; Database={db}; schema=public; UID={uid}; PWD={pw}"

conn = pyodbc.connect(conn_string)



sql = f"select DRILL_ID, HOLE_ID, HOLE_NAME,\
       BIT_DIAMETER, DEPTH_FROM, DEPTH_TO, ELEVATION, DEPTH,\
       HOLE_TYPE, PENETRATION_RATE, ROTARY_REFERENCE, TORQUE,\
       PULLDOWN_PRESSURE, MAIN_AIR_PRESSURE, WATER_FLOW, HOLE_PROFILE,\
       PERCUSSION_PRESSURE, FEEDER_PRESSURE, DAMPER_PRESSURE,\
       ROTATION_PRESSURE, MWD_DATETIME, MSE, IGS\
 from AA_OPERATIONS_MANAGEMENT.SELFSERVICE.MWD_ASSAYS \
 where site =  'IB'"

mwd = pd.read_sql(sql, conn)

mwd.to_csv('data/mwd.csv', index=False)

sql_collar = "select *\
            from AA_OPERATIONS_MANAGEMENT.SELFSERVICE.MWD_COLLARS\
            WHERE SITE = 'IB'"
collars = pd.read_sql(sql_collar, conn)
collars.to_csv('data/mwd_collars.csv', index=False)
conn.close()

