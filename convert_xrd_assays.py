import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



from openpyxl import Workbook
wb = Workbook()
from openpyxl import load_workbook
wb = load_workbook(filename = r'data\Assay\4471 DTR Assay data.R2.xlsx')
sheets = wb.sheetnames
wb.close()

loi_map = {'LOI371-650':'LOI650',
            'LOI650-1000':'LOI1000',
            'LOI1000': 'LOI',
            'LOI110-425':'LOI371','LOI425-650':'LOI650'}

column_map = {'Unnamed: 1':'BHID','Unnamed: 2':'Composite', 'Unnamed: 3':'Size', 'Unnamed: 4':'Type'}
tmp_concat = []
for i in sheets:
    tmp_columns = pd.read_excel(r'data\Assay\4471 DTR Assay data.R2.xlsx',sheet_name=i,skiprows=1,nrows=1)
    tmp_data =pd.read_excel(r'data\Assay\4471 DTR Assay data.R2.xlsx',sheet_name=i,skiprows=2)
    ncolumns = len(tmp_columns.columns)
    new_cols = {}
    for zc in zip(tmp_data.columns[0:ncolumns],tmp_columns.columns[0:ncolumns].to_list()):
        new_cols.update({zc[0]:zc[1]})
    tmp_data.rename(columns=new_cols,inplace=True)
    tmp_data.rename(columns=column_map,inplace=True)
    tmp_data.rename(columns=loi_map,inplace=True)
    numsample = tmp_data.isna().sum(1) <6
    tmp_concat.append(tmp_data[numsample])

data = pd.concat(tmp_concat,axis=0)


len(data)
data.Sample.to_list()
import re
reg = re.compile('Tails|Cons|Feed')
data.Type = data.Type.str.upper().map(lambda x:reg.findall(x)[0])
data.Type.unique()
data.Size.unique()
data[data.Size == 'DTR'].Composite
data.to_csv(r'reference\compiled_assays.csv')
data
qxrd = pd.read_excel(r'data\XRD\XRD\QXRD_Summary_20211129 FS.xlsx',sheet_name=2)
qxrd.Mineral.map(lambda x:x.split('-')[1])
qxrd['Sample'] = qxrd.Mineral.map(lambda x:x.split('-')[1])
def clean_size(x):
    if isinstance(x,str):
        y = x.strip().replace('-','').replace('um','')
    else:
        y = ''
    return y
data.Type.unique()

def clean_type(x):
    if isinstance(x,str):
        y = reg.findall(x)[0]
    else:
        y = ''
    return y

data['CSS (µm)'] = data.Size.map(lambda x: clean_size(x))
data['CompositeID'] = data.BHID.str.strip()+' '+data.Composite.str.strip()+' '+data['CSS (µm)']+' '+data.Type.map(clean_type)
qxrd_columns ={'Unnamed: 0':'BHID', 'Unnamed: 1':'Composite','Pruduct':'Product'}
qxrd.rename(columns=qxrd_columns,inplace=True)
qxrd['CompositeID'] =qxrd.BHID.str.strip()+' '+qxrd.Composite.str.strip()+' '+qxrd['CSS (µm)'].astype(str)+' '+qxrd.Product.map(clean_type)
both = pd.merge(qxrd, data,on=['CompositeID'],how='left')
both.to_csv(r'reference/BV_XRD_Assay_Merge.csv')
