#!/egr/research-hydro/felfelan/installed_soft/anaconda/DIR/bin/python
from __future__ import division # force division to be floating point in Python
import numpy as np
#import h5py
import matplotlib.pyplot as plt
#from scipy.interpolate import griddata
import fnmatch
#import netCDF4 as nc
import os
#import datetime
import pandas as pd
from latlon2xy import smapxy
#==============================================================================
# Python Program: SMAP observation Errors compared to SCAN and CRN stations
# 				  Farshid Felfelani 
#				  03/03/2018
#==============================================================================
sDIR = '/egr/research-hydro/felfelan/SMAP/Data/SPL3SMP004/'
src = '/egr/research-hydro/felfelan/CESM1_2/MyAnalysis/KalmanFilter/src/'
fig = '/egr/research-hydro/felfelan/CESM1_2/MyAnalysis/KalmanFilter/fig/'
smap_am_3D_DIR = '/egr/research-hydro/felfelan/SMAP/Analysis/regridding/src/'


SCAN_DIR = '/egr/research-hydro/felfelan/CESM1_2/MyAnalysis/CLM_SMAP_SCAN_SM/\
scan_uscrn_snotel/whole/'

CRN_DIR = '/egr/research-hydro/felfelan/CESM1_2/MyAnalysis/CLM_SMAP_SCAN_SM/\
scan_uscrn_snotel/CRN_selected/'

dirSTATIONS = '/egr/research-hydro/felfelan/CESM1_2/MyAnalysis/\
CLM_SMAP_SCAN_SM/scan_uscrn_snotel/'


SMAPsdate = '2015-03-31'
SMAPedate = '2017-08-01'

#==============================================================================
# reading SMAP data 
#==============================================================================
smap_am_3D = np.load(smap_am_3D_DIR + 'SMAP_orig_daily_AM_20150331_20170801.npy')

ii = 0
YYYYMMDD = []
for DIR in os.listdir(sDIR):
    for SMfile in os.listdir(sDIR + DIR): 
        if fnmatch.fnmatch(SMfile, '*001.h5'):
            print(sDIR + DIR + '/' + SMfile)            
            YYYYMMDD = YYYYMMDD + [SMfile[13:21]]
            ii = ii + 1
df_smap = pd.DataFrame(columns=['Date','SMAP'])
df_smap['Date'] = pd.to_datetime(YYYYMMDD,format='%Y%m%d')
df_smap['Date'] = df_smap['Date'].dt.date # just keep the date part and remove 
                                          # time part
df_smap['Date'] = list(map(str,df_smap['Date']))  
# I used map function to make all    
# elements as string
#==============================================================================
# reading SCAN data
#==============================================================================
df_SCANstations = pd.read_csv(dirSTATIONS + 'SCAN_stations_ID_latlon.txt', \
                              sep='\t',header=0, error_bad_lines=False)

my_dict = {}
# Loop over the each file which represents a station data
# Save the SCAN data as a dataframe
# Appending each station as an item in dictionary
nSTATION = 0 
for file in sorted(os.listdir(SCAN_DIR)):
    # list only files not directories
    if os.path.isfile(os.path.join(SCAN_DIR, file)):  
        print(file)
        nSTATION = nSTATION + 1
    
        # Getting the row number of the header
        aa = open(SCAN_DIR + file).readlines()
        for ii in range(len(aa)):
            if not fnmatch.fnmatch(aa[ii],'#*'):
                               headerINDEX = ii
                               break
        # print(file, ' Header Row Number: ',ii)
        
        # Reading from the header below
        df1 = pd.read_csv(SCAN_DIR +file,sep=',', header=headerINDEX, \
                         error_bad_lines=False)
        
        my_dict[file] = df1
    
# Adding Columns from different dataframes (with matching date) stored in a \
# python dictionary 
# Initialize 'sum_df'
df_SCAN_SMAP = pd.DataFrame(columns=['Date'])

# Iterate over dataframes of dictionary
for i, tables in enumerate(my_dict):
    print('STATION: ',tables)
    
    lati = df_SCANstations[df_SCANstations['site_name'].str.\
                           contains(tables[5:9])].latitude.iloc[0]
    long = df_SCANstations[df_SCANstations['site_name'].str.\
                           contains(tables[5:9])].longitude.iloc[0]

    df_smap['SMAP'] = smap_am_3D[:,smapxy(lati,long)[1],smapxy(lati,long)[0]]
    df_smap['SMAP'] = df_smap['SMAP'].replace(to_replace = -9999.0,value = np.nan, regex=True)

    # Create dataframe
    df1 = my_dict[tables]

    # Filter rows by 'date'
    df1 = df1[(df1['Date'] >= SMAPsdate) & (df1['Date'] <= SMAPedate)] 

    # Filter for all columns starting with 
    # "Soil Moisture Percent -2in" for top 5cm 
    filter_col1 = [col for col in list(df1) if \
                   col.startswith('Soil Moisture Percent -2in')]   

    # Keep only proper cols
    df_dummy1 = df1[['Date'] + filter_col1]
    df_dummy1 = df_dummy1.rename(columns={i:'{}'.format('SCAN') for i in filter_col1})   
    df_dummy1['SCAN'] = df_dummy1['SCAN'] / 100.0 

    df_dummy1 = df_dummy1.merge(df_smap, how='outer', on=['Date']) 
    # Join new columns from dictionary to old 'sum_df' dataframe
    if i == 0:
        df_SCAN_SMAP = df_dummy1.copy()

    else:

        df_SCAN_SMAP = pd.concat([df_SCAN_SMAP,df_dummy1],axis=0)


#==============================================================================
# reading CRN data
#==============================================================================
my_dict_crn = {}
# Loop over the each file which represents a station data
# Save the SCAN data as a dataframe
# Appending each station as an item in dictionary
nSTATION_crn = 0 
for file in sorted(os.listdir(CRN_DIR)):
    if os.path.isfile(os.path.join(CRN_DIR, file)):  #list only files not directories
        print(file)
        nSTATION_crn = nSTATION_crn + 1       
        # Reading from the header below
        df2 = pd.read_csv(CRN_DIR +file,sep=',', header=0, error_bad_lines=False)
        df2[df2==-9999]=np.NaN
        df2[df2==-99]=np.NaN
        my_dict_crn[file] = df2
    
# Adding Columns from different dataframes (with matching date) stored in a python dictionary 
# Initialize 'sum_df'
df_CRN_SMAP = pd.DataFrame(columns=['Date'])

# Iterate over dataframes of dictionary
for j, tables2 in enumerate(my_dict_crn):
    print('STATION: ',tables2)
    # Create dataframe
    df2 = my_dict_crn[tables2]

    df2 = df2.rename(index=str, columns={"LST_DATE": "Date"})
    df2['Date'] = pd.to_datetime(df2.Date,format='%Y%m%d')
    df2['Date'] = df2['Date'].dt.date  # just keep the date part and remove time part
    df2['Date'] = list(map(str,df2['Date']))

    lati = df2.LATITUDE[0]
    long = df2.LONGITUDE[0]
    
    df_smap['SMAP'] = smap_am_3D[:,smapxy(lati,long)[1],smapxy(lati,long)[0]]
    df_smap['SMAP'] = df_smap['SMAP'].replace(to_replace = -9999.0,value = np.nan, regex=True)    

    # Filter rows by 'date'
    df2 = df2[(df2['Date'] >= SMAPsdate) & (df2['Date'] <= SMAPedate)] 

    # Filter for all columns starting with "Soil Moisture Percent -2in" for top 5cm 
    filter_col2 = [col for col in list(df2) if col.startswith('SOIL_MOISTURE_5_DAILY')]   

    # Keep only proper cols
    df_dummy2 = df2[['Date'] + filter_col2]
    df_dummy2 = df_dummy2.rename(columns={j:'{}'.format('CRN') for j in filter_col2})   

    df_dummy2 = df_dummy2.merge(df_smap, how='outer', on=['Date']) 
    # Join new columns from dictionary to old 'sum_df' dataframe
    if j == 0:
        df_CRN_SMAP = df_dummy2.copy()
    else:
        df_CRN_SMAP = pd.concat([df_CRN_SMAP,df_dummy2],axis=0)
										
df_CRN_SMAP = df_CRN_SMAP.dropna(how='any')
df_SCAN_SMAP = df_SCAN_SMAP.dropna(how='any')

f, ax = plt.subplots()
ax.plot(df_SCAN_SMAP.SCAN.astype(float),df_SCAN_SMAP.SMAP.astype(float) , ls="", \
        markersize= .3 , markeredgecolor='dodgerblue',marker='+', \
        markerfacecolor='none',label='SCAN')

ax.plot(df_CRN_SMAP.CRN.astype(float),df_CRN_SMAP.SMAP.astype(float) , ls="", \
        markersize= .3 , markeredgecolor= 'limegreen',marker='+', \
        markerfacecolor='none',label='USCRN')

RMSE_crn = np.sqrt(np.mean((df_CRN_SMAP.CRN.astype(float)-df_CRN_SMAP.SMAP.astype(float))**2))
RMSE_scan = np.sqrt(np.mean((df_SCAN_SMAP.SCAN.astype(float)-df_SCAN_SMAP.SMAP.astype(float))**2))

ax.text(0.03, 0.9, '$RMSE_{SCAN}: $' + str('%3.3f'%(RMSE_scan)), \
        verticalalignment='bottom', horizontalalignment='left', \
        transform=ax.transAxes, fontsize=10)
ax.text(0.03, 0.8, '$RMSE_{USCRN}: $' + str('%3.3f'%(RMSE_crn)), \
        verticalalignment='bottom', horizontalalignment='left', \
        transform=ax.transAxes, fontsize=10)

plt.ylim(-0.15,0.8)
plt.xlim(-0.15,0.8)

ax.set_ylabel('SCAN/CRN', fontsize=12, color='k')
ax.set_xlabel('SMAP', fontsize=12, color='k')

ax.plot([-0.15,0.8], [-0.15,0.8],linewidth = 0.5, color='k', linestyle = '-.')

ax.set_title('SMAP Soil Moisture Accuracy\n comparing to SCAN and USCRN Networks', fontsize=12)
lgnd = plt.legend(loc='upper right')

lgnd.legendHandles[0]._legmarker.set_markersize(6)
lgnd.legendHandles[1]._legmarker.set_markersize(6)

plt.savefig( fig + 'SMAP_Soil_Moisture_Accuracy_comparing_to_SCAN_and_USCRN_Networks' \
            + '_' + SMAPsdate + '_' + SMAPedate + '.png', bbox_inches='tight', dpi=600 )
