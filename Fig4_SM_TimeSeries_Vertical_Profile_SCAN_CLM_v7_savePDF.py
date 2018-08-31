#!~/anaconda3/DIR/bin/ipython

from __future__ import division # force division to be floating point in Python
from numpy import *
from pylab import *
from matplotlib import gridspec
import pandas as pd
import fnmatch
import os
from netCDF4 import Dataset
from scipy.io import netcdf
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.dates as mdates
from matplotlib import gridspec
from geopy.distance import vincenty

# My module to change the latlon to xy coordination in SMAP data
from latlon2xy import smapxy, cyl5minxy, cntralUSA3minxy
#%%=======================================================================
#===== Farshid Felfelani
#===== First version: 04/12/2018
#===== Vertical SM Profile from SCAN, CLM
#%%=======================================================================
CLM_ctrl_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_021_centralUSA_ICRUCLM45BGCCROPNLDAS_duplicate/'
X_024_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr/no_BiasCorrection/'
X_024_kf_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF/no_BiasCorrection/'
X_024_kf_BC_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF/grndOBS_BiasCorrection_minIRR/'
SMAP_File = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/SMAPorig_12_monthly_ave_for_20150331_20170801.npy'
dirSTATIONS = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/scan_uscrn_snotel/'
SCAN_CRN_TS_DIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/scan_uscrn_snotel/CRN_SCAN_Selected_Stations/Time_Series/'
SCAN_CRN_VP_DIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/scan_uscrn_snotel/CRN_SCAN_Selected_Stations/Vertical_Profile/Fig4_GRL_Paper/'
figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'

# Dates for the Time Series
sDate_T = '2005-01-01'
eDate_T = '2006-12-31'

sYR_T = int(sDate_T[:4])
eYR_T = int(eDate_T[:4])
sMON_T = int(sDate_T[5:7])
eMON_T = int(eDate_T[5:7])
sDY_T = int(sDate_T[8:10])
eDY_T = int(eDate_T[8:10])
 
st_names = ['SCAN_2105_TX','SCAN_2106_TX','SCAN_2107_NM','SCAN_2111_NE']
latseries = [33.55,33.63,33.54,40.37]
lonseries = [-102.37,-102.75,-103.24,-101.72]

# Dates for Vertical Profiles
sDate_V = '2015-01-01'
eDate_V = '2016-12-31'
AveMon = '08' # 2 digit Month

sYR_V = int(sDate_V[:4])
eYR_V = int(eDate_V[:4])
sMON_V = int(sDate_V[5:7])
eMON_V = int(eDate_V[5:7])
sDY_V = int(sDate_V[8:10])
eDY_V = int(eDate_V[8:10]) 

letter1 = ['a','b','c','d']
letter2 = ['e','f','g','h','i','j','k','l']

gs1 = gridspec.GridSpec(4, 1, height_ratios = [1,1,1,1], width_ratios = [1]) # the middle row is to place the colorbar axes
gs1.update(bottom=0.47, top=0.98, hspace = 0.23)
#gs1.update(bottom=0.6, top=0.98)

gs2 = gridspec.GridSpec(2, 4, height_ratios=[1,1], width_ratios=[1,1,1,1]) # the middle row is to place the colorbar axes
gs2.update(bottom=0.05, top=0.35, wspace=0.1, hspace = 0.4)

fig = plt.figure(num=1, figsize=(7.5,9.9)) #figsize: w,h letter size
ax = []
#%%======================================================================= read SCAN data
df_SCANstations = pd.read_csv(dirSTATIONS + 'SCAN_stations_ID_latlon.txt',sep='\t',header=0, error_bad_lines=False)

my_dict = {}
# Loop over the each file which represents a station data
# Save the SCAN data as a dataframe
# Appending each station as an item in dictionary
nSTATION = 0 
for file in sorted(os.listdir(SCAN_CRN_TS_DIR)):
    if os.path.isfile(os.path.join(SCAN_CRN_TS_DIR, file)):  #list only files not directories
        print(file)
        nSTATION = nSTATION + 1
        if 'SCAN' in file:
        # Getting the row number of the header
            aa = open(SCAN_CRN_TS_DIR + file).readlines()
            for ii in range(len(aa)):
                if not fnmatch.fnmatch(aa[ii],'#*'):
                                   headerINDEX = ii
                                   break
            
            # Reading from the header below
            df = pd.read_csv(SCAN_CRN_TS_DIR +file,sep=',', header=headerINDEX, error_bad_lines=False)
        else:
            
            df = pd.read_csv(SCAN_CRN_TS_DIR +file,sep=',', header=0, error_bad_lines=False)
            df[df==-9999]=NaN
            df[df==-99]=NaN
            
        my_dict[file] = df

#%%======================================================================= Time Series =======================================================================
df_SCAN_T = pd.DataFrame(columns=['date'])

# Iterate over dataframes of dictionary
for i, tables in enumerate(my_dict):
  if 'SCAN' in tables:
    # Create dataframe
    df = my_dict[tables]

    # Filter rows by 'date'
    df = df[(df['Date'] >= sDate_T) & (df['Date'] <= eDate_T)] 

    # Filter for all columns starting with "Soil Moisture Percent -2in" for top 5cm 
    filter_col = [col for col in list(df) if col.startswith('Soil Moisture Percent -2in')]

    # Keep only proper cols
    df2 = df[['Date'] + filter_col]
    df2[filter_col] = df2[filter_col]/100.0

    # Join new columns from dictionary to old 'df_SCAN_T' dataframe
    if i == 0:
        df_SCAN_T = df2.rename(columns={i:'{}'.format(tables) for i in filter_col}).copy()
    else:
        df2 = df2.rename(columns={i:'{}'.format(tables) for i in filter_col})
        df_SCAN_T = df2.merge(df_SCAN_T, how='outer', on=['Date']) #, suffixes=('_{}'.format(tables), '_y'))

  else:
      
    df = my_dict[tables]

    df = df.rename(index=str, columns={"LST_DATE": "Date"})
    df['Date'] = pd.to_datetime(df.Date,format='%Y%m%d')
    df['Date'] = df['Date'].dt.date  # just keep the date part and remove time part
    df['Date'] = list(map(str,df['Date']))
    df = df[(df['Date'] >= sDate_T) & (df['Date'] <= eDate_T)] 

    filter_col1 = [col for col in list(df) if col.startswith('SOIL_MOISTURE_5_DAILY')]
    df2 = df[['Date'] + filter_col1]
    if i == 0:
        df_SCAN_T = df2.rename(columns={i:'{}'.format(tables) for i in filter_col1}).copy()
    else:
        df2 = df2.rename(columns={i:'{}'.format(tables) for i in filter_col1})
        df_SCAN_T = df2.merge(df_SCAN_T, how='outer', on=['Date']) #, suffixes=('_{}'.format(tables), '_y'))


# Removing the leap day: Feb 29 of 2000, 2004, 2008, etc.
Leap_indx = []
for ii in range(len(df_SCAN_T)):
    if fnmatch.fnmatch(df_SCAN_T.iloc[ii,0],'*02-29'):
        # print(df_SCAN_T.iloc[ii,0])
        Leap_indx = Leap_indx + [df_SCAN_T.index[ii]]  
df_T_noLeap = df_SCAN_T.drop(Leap_indx)

#%%======================================================================= read Control CLM data
for cc in range(len(latseries)):
    clm3minxy = cntralUSA3minxy(latseries[cc],lonseries[cc])
    # print('=========== Surface SM Time Series========== Control CLM Read')
    xx = 0
    for file in sorted(os.listdir(CLM_ctrl_DIR)):
        if fnmatch.fnmatch(file, '*00000.nc') and int(file[47:51])>= sYR_T and int(file[47:51])<= eYR_T:
            IRRG_FILE = netcdf.netcdf_file(CLM_ctrl_DIR + file,'r')
            nt = len(IRRG_FILE.variables['QIRRIG'][:])
            mcdat = IRRG_FILE.variables['mcdate'][:]   
            
            IRRG_h2osoi_dat0 = IRRG_FILE.variables['H2OSOI'][:]	
            IRRG_h2osoi_dat01 = flip(IRRG_h2osoi_dat0[:,0,].reshape(nt,400,520),axis=1) 
            IRRG_h2osoi_dat02 = flip(IRRG_h2osoi_dat0[:,1,].reshape(nt,400,520),axis=1)
            
            if file[47:51] == str(sYR_T) and xx == 0:
                print('file0:',file)
                IRRG_h2osoi_dat1 = IRRG_h2osoi_dat01[:,clm3minxy[1],clm3minxy[0] ]
                IRRG_h2osoi_dat2 = IRRG_h2osoi_dat02[:,clm3minxy[1],clm3minxy[0] ]
                mcdate = mcdat  
                xx = 1
    
            else:
                print('file:',file)
                IRRG_h2osoi_dat1 = concatenate(( IRRG_h2osoi_dat1, IRRG_h2osoi_dat01[:,clm3minxy[1],clm3minxy[0]] ))
                IRRG_h2osoi_dat2 = concatenate(( IRRG_h2osoi_dat2, IRRG_h2osoi_dat02[:,clm3minxy[1],clm3minxy[0]] ))
                mcdate = concatenate((mcdate,mcdat))    	  
    
    IRRG_h2osoi_dat = IRRG_h2osoi_dat1*(0.014201/0.041649) + IRRG_h2osoi_dat2*(0.027447/0.041649) 	
    
    h2osoi_5cm = IRRG_h2osoi_dat
    
    df_ctrl_h2osoi_5cmT = pd.DataFrame(columns=['Date','ctrl_h2osoi_5cm_' + st_names[cc]])
    df_ctrl_h2osoi_5cmT['Date'] = pd.to_datetime(mcdate,format='%Y%m%d')
    df_ctrl_h2osoi_5cmT['Date'] = df_ctrl_h2osoi_5cmT['Date'].dt.date  # just keep the date part and remove time part
    df_ctrl_h2osoi_5cmT['Date'] = list(map(str,df_ctrl_h2osoi_5cmT['Date']))
    df_ctrl_h2osoi_5cmT['ctrl_h2osoi_5cm_' + st_names[cc]] = h2osoi_5cm
    df_T_noLeap = df_T_noLeap.merge(df_ctrl_h2osoi_5cmT, how='outer', on=['Date'])

  
    #%%======================================================================= read SMAP CLM data
    xx = 0
    for file in sorted(os.listdir(X_024_DIR)):
        if fnmatch.fnmatch(file, '*00000.nc') and int(file[60:64])>= sYR_T and int(file[60:64])<= eYR_T:
            SMAP_IRRG_FILE = netcdf.netcdf_file(X_024_DIR + file,'r')
            nt = len(SMAP_IRRG_FILE.variables['QIRRIG'][:])
            mcdat2 = SMAP_IRRG_FILE.variables['mcdate'][:]  
            
            SMAP_IRRG_h2osoi_dat0 = SMAP_IRRG_FILE.variables['H2OSOI'][:]	
            SMAP_IRRG_h2osoi_dat01 = flip(SMAP_IRRG_h2osoi_dat0[:,0,].reshape(nt,400,520),axis=1) 
            SMAP_IRRG_h2osoi_dat02 = flip(SMAP_IRRG_h2osoi_dat0[:,1,].reshape(nt,400,520),axis=1)

            
            if file[60:64] == str(sYR_T) and xx==0:
                print('file0:',file)
                SMAP_IRRG_h2osoi_dat1 = SMAP_IRRG_h2osoi_dat01[:,clm3minxy[1],clm3minxy[0]]
                SMAP_IRRG_h2osoi_dat2 = SMAP_IRRG_h2osoi_dat02[:,clm3minxy[1],clm3minxy[0]]

                xx = 1
                mcdate2 = mcdat2      
            else:
                print('file:',file)
                SMAP_IRRG_h2osoi_dat1 = concatenate(( SMAP_IRRG_h2osoi_dat1, SMAP_IRRG_h2osoi_dat01[:,clm3minxy[1],clm3minxy[0]] ))
                SMAP_IRRG_h2osoi_dat2 = concatenate(( SMAP_IRRG_h2osoi_dat2, SMAP_IRRG_h2osoi_dat02[:,clm3minxy[1],clm3minxy[0]] ))

                mcdate2 = concatenate((mcdate2,mcdat2))   
         
    SMAP_IRRG_h2osoi_dat = SMAP_IRRG_h2osoi_dat1*(0.014201/0.041649) + SMAP_IRRG_h2osoi_dat2*(0.027447/0.041649) 	
    
    SMAP_h2osoi_5cm = SMAP_IRRG_h2osoi_dat


    df_smap_h2osoi_5cmT = pd.DataFrame(columns=['Date','SMAP_h2osoi_5cm_' + st_names[cc]])
    df_smap_h2osoi_5cmT['Date'] = pd.to_datetime(mcdate2,format='%Y%m%d')
    df_smap_h2osoi_5cmT['Date'] = df_smap_h2osoi_5cmT['Date'].dt.date  # just keep the date part and remove time part
    df_smap_h2osoi_5cmT['Date'] = list(map(str,df_smap_h2osoi_5cmT['Date']))
    df_smap_h2osoi_5cmT['SMAP_h2osoi_5cm_' + st_names[cc]] = SMAP_h2osoi_5cm
    df_T_noLeap = df_T_noLeap.merge(df_smap_h2osoi_5cmT, how='outer', on=['Date'])

   
    #%%======================================================================= read SMAP KF BC CLM data
    xx = 0
    for file in sorted(os.listdir(X_024_kf_BC_DIR)):
        if fnmatch.fnmatch(file, '*_KF.clm2.h1*00000.nc') and int(file[63:67])>= sYR_T and int(file[63:67])<= eYR_T:
            SMAP_KF_IRRG_FILE = netcdf.netcdf_file(X_024_kf_BC_DIR + file,'r')
            nt = len(SMAP_KF_IRRG_FILE.variables['QIRRIG'][:])
            mcdat3 = SMAP_KF_IRRG_FILE.variables['mcdate'][:] 
            
            SMAP_KF_IRRG_h2osoi_dat0 = SMAP_KF_IRRG_FILE.variables['H2OSOI'][:]	
            SMAP_KF_IRRG_h2osoi_dat01 = flip(SMAP_KF_IRRG_h2osoi_dat0[:,0,].reshape(nt,400,520),axis=1) 
            SMAP_KF_IRRG_h2osoi_dat02 = flip(SMAP_KF_IRRG_h2osoi_dat0[:,1,].reshape(nt,400,520),axis=1)
            QIRRIG0 = np.flip(SMAP_KF_IRRG_FILE.variables['QIRRIG'][:].reshape(nt,400,520),axis=1)
            QFLX_RAIN_GRND = np.flip(SMAP_KF_IRRG_FILE.variables['QFLX_RAIN_GRND'][:].reshape(nt,400,520),axis=1)  # mm/s	
            
            if file[63:67] == str(sYR_T) and xx==0:
                print('file0:',file)
                SMAP_KF_IRRG_h2osoi_dat1 = SMAP_KF_IRRG_h2osoi_dat01[:,clm3minxy[1],clm3minxy[0]]
                SMAP_KF_IRRG_h2osoi_dat2 = SMAP_KF_IRRG_h2osoi_dat02[:,clm3minxy[1],clm3minxy[0]]
                QIRRIG_dat  = QIRRIG0[:,clm3minxy[1],clm3minxy[0]]
                QFLX_RAIN_GRND_dat = QFLX_RAIN_GRND[:,clm3minxy[1],clm3minxy[0]]
                
                xx = 1
                mcdate3 = mcdat3       
            else:
                print('file:',file)
                SMAP_KF_IRRG_h2osoi_dat1 = concatenate(( SMAP_KF_IRRG_h2osoi_dat1, SMAP_KF_IRRG_h2osoi_dat01[:,clm3minxy[1],clm3minxy[0]] ))
                SMAP_KF_IRRG_h2osoi_dat2 = concatenate(( SMAP_KF_IRRG_h2osoi_dat2, SMAP_KF_IRRG_h2osoi_dat02[:,clm3minxy[1],clm3minxy[0]] ))
                mcdate3 = concatenate((mcdate3,mcdat3))
                QIRRIG_dat = np.concatenate(( QIRRIG_dat, QIRRIG0[:,clm3minxy[1],clm3minxy[0]] ))
                QFLX_RAIN_GRND_dat = np.concatenate(( QFLX_RAIN_GRND_dat, QFLX_RAIN_GRND[:,clm3minxy[1],clm3minxy[0]] ))
				
    QIRRIG_dat = QIRRIG_dat * 24 * 3600    # mm/s to mm/day	    
    QFLX_RAIN_GRND_dat = QFLX_RAIN_GRND_dat * 24 * 3600    # mm/s to mm/day		
    SMAP_KF_IRRG_h2osoi_dat = SMAP_KF_IRRG_h2osoi_dat1*(0.014201/0.041649) + SMAP_KF_IRRG_h2osoi_dat2*(0.027447/0.041649) 	
    
    SMAP_KF_h2osoi_5cm = SMAP_KF_IRRG_h2osoi_dat


    df_SMAP_KF_h2osoi_5cmT = pd.DataFrame(columns=['Date','SMAP_KF_h2osoi_5cm_' + st_names[cc],'SMAP_KF_qirrig_' + st_names[cc],'SMAP_KF_qflx_rain_grnd_' + st_names[cc]])
    df_SMAP_KF_h2osoi_5cmT['Date'] = pd.to_datetime(mcdate3,format='%Y%m%d')
    df_SMAP_KF_h2osoi_5cmT['Date'] = df_SMAP_KF_h2osoi_5cmT['Date'].dt.date  # just keep the date part and remove time part
    df_SMAP_KF_h2osoi_5cmT['Date'] = list(map(str,df_SMAP_KF_h2osoi_5cmT['Date']))
    df_SMAP_KF_h2osoi_5cmT['SMAP_KF_h2osoi_5cm_' + st_names[cc]] = SMAP_KF_h2osoi_5cm
    df_SMAP_KF_h2osoi_5cmT['SMAP_KF_qirrig_' + st_names[cc]] = QIRRIG_dat
    df_SMAP_KF_h2osoi_5cmT['SMAP_KF_qflx_rain_grnd_' + st_names[cc]] = QFLX_RAIN_GRND_dat - QIRRIG_dat # QFLX_RAIN_GRND has irrigation water in itself
    df_T_noLeap = df_T_noLeap.merge(df_SMAP_KF_h2osoi_5cmT, how='outer', on=['Date'])

#%%======================================================================= Extractin JJA
    #df_T_noLeap.iloc[:,1:nSTATION + 1] = df_T_noLeap.iloc[:,1:nSTATION + 1]/100
    
    # Change the format of dates to standard datetime
#    df_T_noLeap.Date = pd.to_datetime(df_T_noLeap.Date)
    
    # Plot the mean
    
    # I added the index and mean of the five SCAN stations as new columns in the dataframe to be able to plot df.plot.scatter(x,y=label of column)
    df_T_noLeap['ind'] = df_T_noLeap.index
#    df_T_noLeap['meanSCAN'] = df_T_noLeap.iloc[:,1:nSTATION + 1].mean(1)   
#    df_T_noLeap['meanSCAN'] = df_T_noLeap[st_names[cc]]

    
    # form the first year and then concatenate
    df_JJA_index = df_T_noLeap.set_index(['Date'])
    df_JJA = df_JJA_index.loc[str(sYR_T) + '-' + '06' + '-' + str(sDY_T).zfill(2):str(sYR_T) + '-' + '08' + '-' + str(eDY_T).zfill(2)]
    
    for YYYY in range(sYR_T + 1,eYR_T + 1):
        df_JJA_dump = df_JJA_index.loc[str(YYYY) + '-' + '06' + '-' + str(sDY_T).zfill(2):str(YYYY) + '-' + '08' + '-' + str(eDY_T).zfill(2)]
        df_JJA = pd.concat([df_JJA,df_JJA_dump],axis=0)
    
    # reset the index to numbers (it was the dates)
    df_JJA.reset_index(level=0, inplace=True)
#%%======================================================================= Plotting Just the JJA
###WE can Excluding the last station for southernHPA, The last SCAN station with ID: 2108 has the lowest amount compared to others
    ax.append(fig.add_subplot(gs1[cc]))
    
    # But here we do not exclude any stations
    df_JJA['ind'] = df_JJA.index
#    df_JJA['Date'] = df_JJA['Date'].dt.strftime('%Y-%m-%d') #change the time format; removing the time 00:00:00 from Xaxis ticks labels
#    ln1 = df_JJA.plot.scatter(x='ind',y=st_names[cc] ,ax = ax[-1],s = 6 , c='black',marker="s",\
#                           xticks=df_JJA.index[0:len(df_JJA.index):30],\
#                           label='SCAN/CRN Observations')

    sc1 = ax[-1].scatter(df_JJA.index,df_JJA[st_names[cc]] ,s = 1. , c='black',marker="s",\
                           label='SCAN/USCRN Observations')


    
#    ax[-1].fill_between(df_JJA.index, df_JJA.iloc[:,1:nSTATION + 1].min(1), df_JJA.iloc[:,1:nSTATION + 1].max(1), facecolor='lightgrey',edgecolor='lightgrey', \
#                    interpolate=True,label='SCAN/CRN Range', zorder=-1)
    
#    clmctrl_col = [col for col in list(df_T_noLeap) if col.startswith('ctrl_h2osoi_5cm')]
#    clmsmap_col = [col for col in list(df_T_noLeap) if col.startswith('SMAP_h2osoi_5cm')]
#    clmsmapKF_col = [col for col in list(df_T_noLeap) if col.startswith('SMAP_KF_h2osoi_5cm')]
#    
#    ax[-1].scatter(df_JJA.index, df_JJA[clmctrl_col].copy().mean(1),s = 5 , c='tomato',label='CLM Control')
#    ax[-1].scatter(df_JJA.index, df_JJA[clmsmap_col].copy().mean(1),s = 5, c='dodgerblue',label='CLM SMAP_Raw')
#    ax[-1].scatter(df_JJA.index, df_JJA[clmsmapKF_col].copy().mean(1),s = 5, c='forestgreen',label='CLM SMAP_KF')  <



    sc2 = ax[-1].scatter(df_JJA.index, df_JJA['ctrl_h2osoi_5cm_' + st_names[cc]].copy(),s = 1.5 , c='tomato',marker="^",label='CTRL')
    sc3 = ax[-1].scatter(df_JJA.index, df_JJA['SMAP_h2osoi_5cm_' + st_names[cc]].copy(),s = 1.5, c='dodgerblue',marker="o",label='SMAP_raw')
    sc4 = ax[-1].scatter(df_JJA.index, df_JJA['SMAP_KF_h2osoi_5cm_' + st_names[cc]].copy(),s = 1.5, c='forestgreen',marker="h",label='SMAP_KF_BC')

    ln2 = ax[-1].plot(df_JJA.index, df_JJA['ctrl_h2osoi_5cm_' + st_names[cc]].copy(),linewidth=0.5, linestyle =':',c='tomato',label='CTRL')
    ln3 = ax[-1].plot(df_JJA.index, df_JJA['SMAP_h2osoi_5cm_' + st_names[cc]].copy(),linewidth=0.5, linestyle =':',c='dodgerblue',label='SMAP_raw')
    ln4 = ax[-1].plot(df_JJA.index, df_JJA['SMAP_KF_h2osoi_5cm_' + st_names[cc]].copy(),linewidth=0.5, linestyle =':',c='forestgreen',label='SMAP_KF_BC')


    ax2 = ax[-1].twinx()
    ba1 = ax2.bar(df_JJA.index,df_JJA['SMAP_KF_qirrig_' + st_names[cc]],color='forestgreen',width=1.0,alpha=0.50,label='Irrigation Water SMAP_KF_BC')
    ba2 = ax2.bar(df_JJA.index,df_JJA['SMAP_KF_qflx_rain_grnd_' + st_names[cc]],color='deepskyblue',width=1.0,bottom=df_JJA['SMAP_KF_qirrig_' + st_names[cc]],alpha=0.50,label='Precipitation')

    sc_fake = ax[-1].scatter(0,0 ,s = 0.1 , c='black',marker='*', facecolors='black',label='SMAP Satellite')  
	
    gca().set_xticklabels([''])  # remove the xtick labels
    ax[-1].set_xticks(df_JJA.index[0:len(df_JJA.index):10])
    # size of the xticks and yticks labels
    ax[-1].tick_params(axis = 'both', which = 'major', labelsize = 7)
    ax2.tick_params(axis = 'y', which = 'major', labelsize = 7)
    
    
    if cc == 3: #or cc ==2:    
        ax[-1].set_xticklabels(df_JJA.Date[0:len(df_JJA.index):10], fontsize=7, rotation=-45)
        for tick in ax[-1].xaxis.get_majorticklabels():
            tick.set_horizontalalignment("center")
        minorLocator = MultipleLocator(5)
        ax[-1].xaxis.set_minor_locator(minorLocator)
        
        # if cc ==2:
            # legend_properties = {'size': 15}     # phont size
            # lgnd = ax[-1].legend(handles=[sc1,sc2,sc3,sc4,ba1],loc='upper left',prop=legend_properties,ncol=2)
            # lgnd.legendHandles[0]._sizes = [40]  # size of the scatter in the legend
            # lgnd.legendHandles[1]._sizes = [40]
            # lgnd.legendHandles[2]._sizes = [40]
            # lgnd.legendHandles[3]._sizes = [40]
            
#        else:
#            ax[-1].legend_.remove()

    else:
#        ax[-1].legend_.remove()
        minorLocator = MultipleLocator(10)
        ax[-1].xaxis.set_minor_locator(minorLocator)
            
    ax[-1].tick_params(axis = 'x', direction='inout', length=6, width=1.2, colors='k')
    if cc == 2:
        ax[-1].set_ylabel('Soil Moisture Content ' + r'($\mathrm{mm^3/mm^3}$)', fontsize=10, color='k')
        ax[-1].yaxis.set_label_coords(-0.05, 1.0) 

    else:
        y_axis = ax[-1].axes.get_yaxis()
        y_label = y_axis.get_label()
        y_label.set_visible(False)
       
    #ax1 = plt.axes()
    x_axis = ax[-1].axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    
    ax[-1].text(0.07, 0.82, '(' + letter1[cc] + ')', verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes,fontsize=12)    
    plt.setp(ax[-1].spines.values(), color='darkgray')    
    plt.setp(ax2.spines.values(), color='darkgray') 
	
    plt.xlim(df_JJA.index.min(),df_JJA.index.max())
    # plt.ylim(0,0.5)
    plt.title('Top 5cm Soil Moisture for JJA - ' +  st_names[cc],fontsize=8)    
    
    if cc == 2:
        ax2.set_ylabel('Irrigation Water Amount ' + r'($\mathrm{mm/day}$)', fontsize=10, color='k')
        ax2.yaxis.set_label_coords(1.05, 1.0)    

axe = fig.add_axes([0.06,0.345,0.88,0.03])	
pie = plt.pie([1,1])
# plt.legend(handles=[vadose, Snow, GW, River, TOTM, PCR, PETR,P_ET_R_Trend,GRACEns,GRACE_Trend],bbox_to_anchor=(0., 1.02, 1., .102), ncol=5, mode="expand", borderaxespad=0., prop={'size':12})
lgnd = plt.legend(handles=[sc1,sc2,sc3,sc4,sc_fake,ba1, ba2],bbox_to_anchor=(0., 1.02, 1., .102),ncol=4, mode="expand", borderaxespad=0., edgecolor='grey',prop={'size':7})
lgnd.legendHandles[0]._sizes = [20]  # size of the scatter in the legend
lgnd.legendHandles[1]._sizes = [30]
lgnd.legendHandles[2]._sizes = [30]
lgnd.legendHandles[3]._sizes = [40]
lgnd.legendHandles[4]._sizes = [50]
# PETR_fake.remove()
for group in pie:
    for x in group:
        x.set_visible(False)
#%%==============================================================================================================================================================
#%%============================================================================================================================================================== 
#%%======================================================================= Vertical Profiles ==================================================================== 
#%%============================================================================================================================================================== 
#%%============================================================================================================================================================== 

nSTATION = 0 
# for file in sorted(os.listdir(SCAN_CRN_VP_DIR)):
    # if os.path.isfile(os.path.join(SCAN_CRN_VP_DIR, file)) and 'SCAN' not in file:  #list only files not directories; and only CRN files
        # print(file)
        # nSTATION = nSTATION + 1       
        #### Reading from the header below
        # df = pd.read_csv(SCAN_CRN_VP_DIR +file,sep=',', header=0, error_bad_lines=False)
        # df[df==-9999]=NaN
        # df[df==-99]=NaN
        # my_dict2[file] = df

		

my_dict2 = {}		
for file in sorted(os.listdir(SCAN_CRN_VP_DIR)):
    if os.path.isfile(os.path.join(SCAN_CRN_VP_DIR, file)):  #list only files not directories
        print(file)
        nSTATION = nSTATION + 1
        if 'SCAN' in file:
        # Getting the row number of the header
            aa = open(SCAN_CRN_VP_DIR + file).readlines()
            for ii in range(len(aa)):
                if not fnmatch.fnmatch(aa[ii],'#*'):
                                   headerINDEX = ii
                                   break
            
            # Reading from the header below
            df = pd.read_csv(SCAN_CRN_VP_DIR +file,sep=',', header=headerINDEX, error_bad_lines=False)
        else:
            
            df = pd.read_csv(SCAN_CRN_VP_DIR +file,sep=',', header=0, error_bad_lines=False)
            df[df==-9999]=NaN
            df[df==-99]=NaN
            
        my_dict2[file] = df		
		
# Adding Columns from different dataframes (with matching date) stored in a python dictionary 
# Initialize 'sum_df'

gs2counter = 0
# Iterate over dataframes of dictionary
for i, tables in enumerate(my_dict2):
	sum_df1 = pd.DataFrame(columns=['date'])
	sum_df2 = pd.DataFrame(columns=['date'])
	sum_df3 = pd.DataFrame(columns=['date'])
	sum_df4 = pd.DataFrame(columns=['date'])
	sum_df5 = pd.DataFrame(columns=['date'])
	ax.append(fig.add_subplot(gs2[gs2counter]))
	print('STATION: ',tables)
		# Create dataframe
	df = my_dict2[tables]
	if 'SCAN' in tables:
		# Filter rows by 'date'
		df = df[(df['Date'] >= sDate_V) & (df['Date'] <= eDate_V)] 
		
		lati = df_SCANstations[df_SCANstations['site_name'].str.contains(tables[5:9])].latitude.iloc[0]
		long = df_SCANstations[df_SCANstations['site_name'].str.contains(tables[5:9])].longitude.iloc[0]
		
		# Filter for all columns starting with "Soil Moisture Percent -2in" for top 5cm 
		filter_col1 = [col for col in list(df) if col.startswith('Soil Moisture Percent -2in')]
		filter_col2 = [col for col in list(df) if col.startswith('Soil Moisture Percent -4in')]    
		filter_col3 = [col for col in list(df) if col.startswith('Soil Moisture Percent -8in')]    
		filter_col4 = [col for col in list(df) if col.startswith('Soil Moisture Percent -20in')]    
		filter_col5 = [col for col in list(df) if col.startswith('Soil Moisture Percent -40in')]    

		# Keep only proper cols
		df_dummy1 = df[['Date'] + filter_col1]
		df_dummy2 = df[['Date'] + filter_col2]
		df_dummy3 = df[['Date'] + filter_col3]
		df_dummy4 = df[['Date'] + filter_col4]
		df_dummy5 = df[['Date'] + filter_col5]
		
		# removing the leap year days
		df1 = df_dummy1[~df_dummy1.Date.str.endswith('02-29')]
		df2 = df_dummy2[~df_dummy2.Date.str.endswith('02-29')]
		df3 = df_dummy3[~df_dummy3.Date.str.endswith('02-29')]
		df4 = df_dummy4[~df_dummy4.Date.str.endswith('02-29')]
		df5 = df_dummy5[~df_dummy5.Date.str.endswith('02-29')]

		# Join new columns from dictionary to old 'sum_df' dataframe
		# if i == 0:
		sum_df1 = df1.rename(columns={i:'{}'.format(tables) for i in filter_col1}).copy()
		sum_df2 = df2.rename(columns={i:'{}'.format(tables) for i in filter_col2}).copy()
		sum_df3 = df3.rename(columns={i:'{}'.format(tables) for i in filter_col3}).copy()
		sum_df4 = df4.rename(columns={i:'{}'.format(tables) for i in filter_col4}).copy()
		sum_df5 = df5.rename(columns={i:'{}'.format(tables) for i in filter_col5}).copy()

		# else:
			# df1 = df1.rename(columns={i:'{}'.format(tables) for i in filter_col1})
			# sum_df1 = df1.merge(sum_df1, how='outer', on=['Date'])
			# df2 = df2.rename(columns={i:'{}'.format(tables) for i in filter_col2})
			# sum_df2 = df2.merge(sum_df2, how='outer', on=['Date'])
			# df3 = df3.rename(columns={i:'{}'.format(tables) for i in filter_col3})
			# sum_df3 = df3.merge(sum_df3, how='outer', on=['Date'])
			# df4 = df4.rename(columns={i:'{}'.format(tables) for i in filter_col4})
			# sum_df4 = df4.merge(sum_df4, how='outer', on=['Date'])
			# df5 = df5.rename(columns={i:'{}'.format(tables) for i in filter_col5})
			# sum_df5 = df5.merge(sum_df5, how='outer', on=['Date'])

	if 'SCAN' not in tables:
		df = df.rename(index=str, columns={"LST_DATE": "Date"})
		df['Date'] = pd.to_datetime(df.Date,format='%Y%m%d')
		df['Date'] = df['Date'].dt.date  # just keep the date part and remove time part
		df['Date'] = list(map(str,df['Date']))

		lati = df.LATITUDE[0]
		long = df.LONGITUDE[0]
		# Filter rows by 'date'
		df = df[(df['Date'] >= sDate_V) & (df['Date'] <= eDate_V)] 

		# Filter for all columns starting with "Soil Moisture Percent -2in" for top 5cm 
		filter_col1 = [col for col in list(df) if col.startswith('SOIL_MOISTURE_5_DAILY')]
		filter_col2 = [col for col in list(df) if col.startswith('SOIL_MOISTURE_10_DAILY')]    
		filter_col3 = [col for col in list(df) if col.startswith('SOIL_MOISTURE_20_DAILY')]    
		filter_col4 = [col for col in list(df) if col.startswith('SOIL_MOISTURE_50_DAILY')]    
		filter_col5 = [col for col in list(df) if col.startswith('SOIL_MOISTURE_100_DAILY')]    

		# Keep only proper cols
		df_dummy1 = df[['Date'] + filter_col1]
		df_dummy2 = df[['Date'] + filter_col2]
		df_dummy3 = df[['Date'] + filter_col3]
		df_dummy4 = df[['Date'] + filter_col4]
		df_dummy5 = df[['Date'] + filter_col5]
		
		# removing the leap year days
		df1 = df_dummy1[~df_dummy1.Date.str.endswith('02-29')]
		df2 = df_dummy2[~df_dummy2.Date.str.endswith('02-29')]
		df3 = df_dummy3[~df_dummy3.Date.str.endswith('02-29')]
		df4 = df_dummy4[~df_dummy4.Date.str.endswith('02-29')]
		df5 = df_dummy5[~df_dummy5.Date.str.endswith('02-29')]

		# Join new columns from dictionary to old 'sum_df' dataframe
		# if i == 0:
		sum_df1 = df1.rename(columns={i:'{}'.format(tables) for i in filter_col1}).copy()
		sum_df2 = df2.rename(columns={i:'{}'.format(tables) for i in filter_col2}).copy()
		sum_df3 = df3.rename(columns={i:'{}'.format(tables) for i in filter_col3}).copy()
		sum_df4 = df4.rename(columns={i:'{}'.format(tables) for i in filter_col4}).copy()
		sum_df5 = df5.rename(columns={i:'{}'.format(tables) for i in filter_col5}).copy()
		# else:
			# df1 = df1.rename(columns={i:'{}'.format(tables) for i in filter_col1})
			# sum_df1 = df1.merge(sum_df1, how='outer', on=['Date']) 
			# df2 = df2.rename(columns={i:'{}'.format(tables) for i in filter_col2})
			# sum_df2 = df2.merge(sum_df2, how='outer', on=['Date'])
			# df3 = df3.rename(columns={i:'{}'.format(tables) for i in filter_col3})
			# sum_df3 = df3.merge(sum_df3, how='outer', on=['Date'])
			# df4 = df4.rename(columns={i:'{}'.format(tables) for i in filter_col4})
			# sum_df4 = df4.merge(sum_df4, how='outer', on=['Date'])
			# df5 = df5.rename(columns={i:'{}'.format(tables) for i in filter_col5})
			# sum_df5 = df5.merge(sum_df5, how='outer', on=['Date'])

	# sort based on the Date and reindex
	sum_df1 = sum_df1.sort_values(["Date"])
	sum_df1 = sum_df1.reset_index(drop=True)
	sum_df2 = sum_df2.sort_values(["Date"])
	sum_df2 = sum_df2.reset_index(drop=True)
	sum_df3 = sum_df3.sort_values(["Date"])
	sum_df3 = sum_df3.reset_index(drop=True)
	sum_df4 = sum_df4.sort_values(["Date"])
	sum_df4 = sum_df4.reset_index(drop=True)
	sum_df5 = sum_df5.sort_values(["Date"])
	sum_df5 = sum_df5.reset_index(drop=True)

	# if 'SCAN' in tables:
		# TABLE_Both = tables[0:9]
		# TABLE_Both0 = tables[:-4]
	# else:
		# TABLE_Both = tables
		# TABLE_Both0 = tables
	TABLE_Both = tables
	TABLE_Both0 = tables
	#%%======================================================================= read Control CLM data
	xx = 0
	for file in sorted(os.listdir(CLM_ctrl_DIR)):
		if fnmatch.fnmatch(file, '*00000.nc') and int(file[47:51])>= sYR_V and int(file[47:51])<= eYR_V:
			# print(file)
			IRRG_FILE = netcdf.netcdf_file(CLM_ctrl_DIR + file,'r')
			nt = len(IRRG_FILE.variables['QIRRIG'][:])
#        print(nt)

			mcdat4 = IRRG_FILE.variables['mcdate'][:]

			IRRG_QIRRIG0 = flip(IRRG_FILE.variables['QIRRIG'][:].reshape(nt,400,520),axis=1)	   
			IRRG_h2osoi_dat0 = IRRG_FILE.variables['H2OSOI'][:]	
			IRRG_h2osoi_dat01 = flip(IRRG_h2osoi_dat0[:,0,].reshape(nt,400,520),axis=1) 
			IRRG_h2osoi_dat02 = flip(IRRG_h2osoi_dat0[:,1,].reshape(nt,400,520),axis=1)
			IRRG_h2osoi_dat03 = flip(IRRG_h2osoi_dat0[:,2,].reshape(nt,400,520),axis=1)
			IRRG_h2osoi_dat04 = flip(IRRG_h2osoi_dat0[:,3,].reshape(nt,400,520),axis=1)
			IRRG_h2osoi_dat05 = flip(IRRG_h2osoi_dat0[:,4,].reshape(nt,400,520),axis=1)
			IRRG_h2osoi_dat06 = flip(IRRG_h2osoi_dat0[:,5,].reshape(nt,400,520),axis=1)
			IRRG_h2osoi_dat07 = flip(IRRG_h2osoi_dat0[:,6,].reshape(nt,400,520),axis=1)
			IRRG_h2osoi_dat08 = flip(IRRG_h2osoi_dat0[:,7,].reshape(nt,400,520),axis=1)
			# print(file[47:51],file[52:54],file[55:57])
			
			# if file[47:51] == str(sYR_V) and file[52:54] == str(sMON).zfill(2) and file[55:57] == str(sDY_T).zfill(2):
			if file[47:51] == str(sYR_V) and xx == 0:
				print('file0:',file)
				IRRG_QIRRIG_dat = IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				IRRG_h2osoi_dat1 = IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				IRRG_h2osoi_dat2 = IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				IRRG_h2osoi_dat3 = IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				IRRG_h2osoi_dat4 = IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				IRRG_h2osoi_dat5 = IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				IRRG_h2osoi_dat6 = IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				IRRG_h2osoi_dat7 = IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				IRRG_h2osoi_dat8 = IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				mcdate4 = mcdat4                
				xx = 1
				
			else:
				print('file:',file)
				IRRG_QIRRIG_dat = concatenate(( IRRG_QIRRIG_dat, IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				IRRG_h2osoi_dat1 = concatenate(( IRRG_h2osoi_dat1, IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				IRRG_h2osoi_dat2 = concatenate(( IRRG_h2osoi_dat2, IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				IRRG_h2osoi_dat3 = concatenate(( IRRG_h2osoi_dat3, IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				IRRG_h2osoi_dat4 = concatenate(( IRRG_h2osoi_dat4, IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				IRRG_h2osoi_dat5 = concatenate(( IRRG_h2osoi_dat5, IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				IRRG_h2osoi_dat6 = concatenate(( IRRG_h2osoi_dat6, IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				IRRG_h2osoi_dat7 = concatenate(( IRRG_h2osoi_dat7, IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				IRRG_h2osoi_dat8 = concatenate(( IRRG_h2osoi_dat8, IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))	 
				mcdate4 = concatenate((mcdate4,mcdat4))				
	# consider the SM of top 5 cm soil layer as the SM
	h2osoi_5cm = IRRG_h2osoi_dat1*(0.014201/0.041649) + IRRG_h2osoi_dat2*(0.027447/0.041649) 	
	IRRG_QIRRIG_dat = IRRG_QIRRIG_dat * 24 * 3600  #mm/s to mm/day	

	df_ctrl_h2osoi01 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L1'])
	df_ctrl_h2osoi01['Date'] = pd.to_datetime(mcdate4,format='%Y%m%d')
	df_ctrl_h2osoi01['Date'] = df_ctrl_h2osoi01['Date'].dt.date  # just keep the date part and remove time part
	df_ctrl_h2osoi01['Date'] = list(map(str,df_ctrl_h2osoi01['Date']))
	df_ctrl_h2osoi01[TABLE_Both + '_ctrl_L1'] = IRRG_h2osoi_dat1 
	# the Date column of sum_df is str type, 
	#this (Date column of df_ctrl_h2osoi01) should be also str to be merged in sum_df, 
	#so I used map function to make all elements as string
	sum_df1 = sum_df1.merge(df_ctrl_h2osoi01, how='outer', on=['Date'])

	df_ctrl_h2osoi02 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L2'])
	df_ctrl_h2osoi02['Date'] = df_ctrl_h2osoi01['Date']
	df_ctrl_h2osoi02[TABLE_Both + '_ctrl_L2'] = IRRG_h2osoi_dat2
	sum_df1 = sum_df1.merge(df_ctrl_h2osoi02, how='outer', on=['Date'])

	df_ctrl_h2osoi1 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L3'])
	df_ctrl_h2osoi1['Date'] = df_ctrl_h2osoi01['Date']
	df_ctrl_h2osoi1[TABLE_Both + '_ctrl_L3'] = IRRG_h2osoi_dat3
	sum_df1 = sum_df1.merge(df_ctrl_h2osoi1, how='outer', on=['Date'])	
	
	df_ctrl_h2osoi2 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L4'])
	df_ctrl_h2osoi2['Date'] = df_ctrl_h2osoi01['Date']
	df_ctrl_h2osoi2[TABLE_Both + '_ctrl_L4'] = IRRG_h2osoi_dat4
	sum_df2 = sum_df2.merge(df_ctrl_h2osoi2, how='outer', on=['Date'])	

	df_ctrl_h2osoi3 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L5'])
	df_ctrl_h2osoi3['Date'] = df_ctrl_h2osoi01['Date']
	df_ctrl_h2osoi3[TABLE_Both + '_ctrl_L5'] = IRRG_h2osoi_dat5
	sum_df3 = sum_df3.merge(df_ctrl_h2osoi3, how='outer', on=['Date'])	
	
	df_ctrl_h2osoi4 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L6'])
	df_ctrl_h2osoi4['Date'] = df_ctrl_h2osoi01['Date']
	df_ctrl_h2osoi4[TABLE_Both + '_ctrl_L6'] = IRRG_h2osoi_dat6
	sum_df4 = sum_df4.merge(df_ctrl_h2osoi4, how='outer', on=['Date'])	
	
	df_ctrl_h2osoi5 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L7'])
	df_ctrl_h2osoi5['Date'] = df_ctrl_h2osoi01['Date']
	df_ctrl_h2osoi5[TABLE_Both + '_ctrl_L7'] = IRRG_h2osoi_dat7
	sum_df5 = sum_df5.merge(df_ctrl_h2osoi5, how='outer', on=['Date'])	  

	df_ctrl_h2osoi6 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L8'])
	df_ctrl_h2osoi6['Date'] = df_ctrl_h2osoi01['Date']
	df_ctrl_h2osoi6[TABLE_Both + '_ctrl_L8'] = IRRG_h2osoi_dat8
	sum_df5 = sum_df5.merge(df_ctrl_h2osoi6, how='outer', on=['Date'])
	
#%%======================================================================= read SMAP CLM data
	xx = 0
	for file in sorted(os.listdir(X_024_DIR)):
		if fnmatch.fnmatch(file, '*00000.nc') and int(file[60:64])>= sYR_V and int(file[60:64])<= eYR_V:
			SMAP_IRRG_FILE = netcdf.netcdf_file(X_024_DIR + file,'r')
			nt = len(SMAP_IRRG_FILE.variables['QIRRIG'][:])

			mcdat5 = SMAP_IRRG_FILE.variables['mcdate'][:]
			
			SMAP_IRRG_QIRRIG0 = flip(SMAP_IRRG_FILE.variables['QIRRIG'][:].reshape(nt,400,520),axis=1)	   
			SMAP_IRRG_h2osoi_dat0 = SMAP_IRRG_FILE.variables['H2OSOI'][:]	
			SMAP_IRRG_h2osoi_dat01 = flip(SMAP_IRRG_h2osoi_dat0[:,0,].reshape(nt,400,520),axis=1) 
			SMAP_IRRG_h2osoi_dat02 = flip(SMAP_IRRG_h2osoi_dat0[:,1,].reshape(nt,400,520),axis=1)
			SMAP_IRRG_h2osoi_dat03 = flip(SMAP_IRRG_h2osoi_dat0[:,2,].reshape(nt,400,520),axis=1)
			SMAP_IRRG_h2osoi_dat04 = flip(SMAP_IRRG_h2osoi_dat0[:,3,].reshape(nt,400,520),axis=1)
			SMAP_IRRG_h2osoi_dat05 = flip(SMAP_IRRG_h2osoi_dat0[:,4,].reshape(nt,400,520),axis=1)
			SMAP_IRRG_h2osoi_dat06 = flip(SMAP_IRRG_h2osoi_dat0[:,5,].reshape(nt,400,520),axis=1)
			SMAP_IRRG_h2osoi_dat07 = flip(SMAP_IRRG_h2osoi_dat0[:,6,].reshape(nt,400,520),axis=1)
			SMAP_IRRG_h2osoi_dat08 = flip(SMAP_IRRG_h2osoi_dat0[:,7,].reshape(nt,400,520),axis=1)

			if file[60:64] == str(sYR_V) and xx==0:
				print('file0:',file)
				SMAP_IRRG_QIRRIG_dat = SMAP_IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAP_IRRG_h2osoi_dat1 = SMAP_IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAP_IRRG_h2osoi_dat2 = SMAP_IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAP_IRRG_h2osoi_dat3 = SMAP_IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAP_IRRG_h2osoi_dat4 = SMAP_IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAP_IRRG_h2osoi_dat5 = SMAP_IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAP_IRRG_h2osoi_dat6 = SMAP_IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAP_IRRG_h2osoi_dat7 = SMAP_IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAP_IRRG_h2osoi_dat8 = SMAP_IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				mcdate5 = mcdat5
				xx = 1

			else:
				print('file:',file)
				SMAP_IRRG_QIRRIG_dat = concatenate(( SMAP_IRRG_QIRRIG_dat, SMAP_IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAP_IRRG_h2osoi_dat1 = concatenate(( SMAP_IRRG_h2osoi_dat1, SMAP_IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAP_IRRG_h2osoi_dat2 = concatenate(( SMAP_IRRG_h2osoi_dat2, SMAP_IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAP_IRRG_h2osoi_dat3 = concatenate(( SMAP_IRRG_h2osoi_dat3, SMAP_IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAP_IRRG_h2osoi_dat4 = concatenate(( SMAP_IRRG_h2osoi_dat4, SMAP_IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAP_IRRG_h2osoi_dat5 = concatenate(( SMAP_IRRG_h2osoi_dat5, SMAP_IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAP_IRRG_h2osoi_dat6 = concatenate(( SMAP_IRRG_h2osoi_dat6, SMAP_IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAP_IRRG_h2osoi_dat7 = concatenate(( SMAP_IRRG_h2osoi_dat7, SMAP_IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAP_IRRG_h2osoi_dat8 = concatenate(( SMAP_IRRG_h2osoi_dat8, SMAP_IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				mcdate5 = concatenate((mcdate5,mcdat5))

	SMAP_IRRG_h2osoi_dat = SMAP_IRRG_h2osoi_dat1*(0.014201/0.041649) + SMAP_IRRG_h2osoi_dat2*(0.027447/0.041649) 	
	SMAP_IRRG_QIRRIG_dat = SMAP_IRRG_QIRRIG_dat * 24 * 3600  #mm/s to mm/day	

	SMAP_QIRRIG = SMAP_IRRG_QIRRIG_dat
	SMAP_h2osoi_5cm = SMAP_IRRG_h2osoi_dat

	df_smap_h2osoi01 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L1'])
	df_smap_h2osoi01['Date'] = pd.to_datetime(mcdate5,format='%Y%m%d')
	df_smap_h2osoi01['Date'] = df_smap_h2osoi01['Date'].dt.date  # just keep the date part and remove time part
	df_smap_h2osoi01['Date'] = list(map(str,df_smap_h2osoi01['Date']))	
	df_smap_h2osoi01[TABLE_Both + '_SMAP_L1'] = SMAP_IRRG_h2osoi_dat1
	sum_df1 = sum_df1.merge(df_smap_h2osoi01, how='outer', on=['Date'])

	df_smap_h2osoi02 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L2'])
	df_smap_h2osoi02['Date'] = df_smap_h2osoi01['Date']
	df_smap_h2osoi02[TABLE_Both + '_SMAP_L2'] = SMAP_IRRG_h2osoi_dat2
	sum_df1 = sum_df1.merge(df_smap_h2osoi02, how='outer', on=['Date'])	

	df_smap_h2osoi1 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L3'])
	df_smap_h2osoi1['Date'] = df_smap_h2osoi01['Date']
	df_smap_h2osoi1[TABLE_Both + '_SMAP_L3'] = SMAP_IRRG_h2osoi_dat3
	sum_df1 = sum_df1.merge(df_smap_h2osoi1, how='outer', on=['Date'])	
	
	df_smap_h2osoi2 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L4'])
	df_smap_h2osoi2['Date'] = df_smap_h2osoi01['Date']
	df_smap_h2osoi2[TABLE_Both + '_SMAP_L4'] = SMAP_IRRG_h2osoi_dat4
	sum_df2 = sum_df2.merge(df_smap_h2osoi2, how='outer', on=['Date'])	
	
	df_smap_h2osoi3 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L5'])
	df_smap_h2osoi3['Date'] = df_smap_h2osoi01['Date']
	df_smap_h2osoi3[TABLE_Both + '_SMAP_L5'] = SMAP_IRRG_h2osoi_dat5
	sum_df3 = sum_df3.merge(df_smap_h2osoi3, how='outer', on=['Date'])

	df_smap_h2osoi4 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L6'])
	df_smap_h2osoi4['Date'] = df_smap_h2osoi01['Date']
	df_smap_h2osoi4[TABLE_Both + '_SMAP_L6'] = SMAP_IRRG_h2osoi_dat6
	sum_df4 = sum_df4.merge(df_smap_h2osoi4, how='outer', on=['Date'])

	df_smap_h2osoi5 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L7'])
	df_smap_h2osoi5['Date'] = df_smap_h2osoi01['Date']
	df_smap_h2osoi5[TABLE_Both + '_SMAP_L7'] = SMAP_IRRG_h2osoi_dat7
	sum_df5 = sum_df5.merge(df_smap_h2osoi5, how='outer', on=['Date'])  

	df_smap_h2osoi6 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L8'])
	df_smap_h2osoi6['Date'] = df_smap_h2osoi01['Date']
	df_smap_h2osoi6[TABLE_Both + '_SMAP_L8'] = SMAP_IRRG_h2osoi_dat8
	sum_df5 = sum_df5.merge(df_smap_h2osoi6, how='outer', on=['Date'])	
	
#%%======================================================================= read SMAP CLM X_024 daily, KF_BC
	xx = 0
	for file in sorted(os.listdir(X_024_kf_BC_DIR)):
		# print(file)
		if fnmatch.fnmatch(file, '*_KF.clm2.h1*00000.nc') and int(file[63:67])>= sYR_V and int(file[63:67])<= eYR_V:
			SMAPdailyCVEX_IRRG_FILE = netcdf.netcdf_file(X_024_kf_BC_DIR + file,'r')
			nt = len(SMAPdailyCVEX_IRRG_FILE.variables['QIRRIG'][:])

			mcdat6 = SMAPdailyCVEX_IRRG_FILE.variables['mcdate'][:]
			
			SMAPdailyCVEX_IRRG_QIRRIG0 = flip(SMAPdailyCVEX_IRRG_FILE.variables['QIRRIG'][:].reshape(nt,400,520),axis=1)	   
			SMAPdailyCVEX_IRRG_h2osoi_dat0 = SMAPdailyCVEX_IRRG_FILE.variables['H2OSOI'][:]	
			SMAPdailyCVEX_IRRG_h2osoi_dat01 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,0,].reshape(nt,400,520),axis=1) 
			SMAPdailyCVEX_IRRG_h2osoi_dat02 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,1,].reshape(nt,400,520),axis=1)
			SMAPdailyCVEX_IRRG_h2osoi_dat03 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,2,].reshape(nt,400,520),axis=1)
			SMAPdailyCVEX_IRRG_h2osoi_dat04 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,3,].reshape(nt,400,520),axis=1)
			SMAPdailyCVEX_IRRG_h2osoi_dat05 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,4,].reshape(nt,400,520),axis=1)
			SMAPdailyCVEX_IRRG_h2osoi_dat06 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,5,].reshape(nt,400,520),axis=1)
			SMAPdailyCVEX_IRRG_h2osoi_dat07 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,6,].reshape(nt,400,520),axis=1)
			SMAPdailyCVEX_IRRG_h2osoi_dat08 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,7,].reshape(nt,400,520),axis=1)

			if file[63:67] == str(sYR_V) and xx==0:
				print('file0:',file)
				SMAPdailyCVEX_IRRG_QIRRIG_dat = SMAPdailyCVEX_IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAPdailyCVEX_IRRG_h2osoi_dat1 = SMAPdailyCVEX_IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAPdailyCVEX_IRRG_h2osoi_dat2 = SMAPdailyCVEX_IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAPdailyCVEX_IRRG_h2osoi_dat3 = SMAPdailyCVEX_IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAPdailyCVEX_IRRG_h2osoi_dat4 = SMAPdailyCVEX_IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAPdailyCVEX_IRRG_h2osoi_dat5 = SMAPdailyCVEX_IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAPdailyCVEX_IRRG_h2osoi_dat6 = SMAPdailyCVEX_IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAPdailyCVEX_IRRG_h2osoi_dat7 = SMAPdailyCVEX_IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				SMAPdailyCVEX_IRRG_h2osoi_dat8 = SMAPdailyCVEX_IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
				mcdate6 = mcdat6
				xx = 1

			else:
				print('file:',file)
				SMAPdailyCVEX_IRRG_QIRRIG_dat = concatenate(( SMAPdailyCVEX_IRRG_QIRRIG_dat, SMAPdailyCVEX_IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAPdailyCVEX_IRRG_h2osoi_dat1 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat1, SMAPdailyCVEX_IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAPdailyCVEX_IRRG_h2osoi_dat2 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat2, SMAPdailyCVEX_IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAPdailyCVEX_IRRG_h2osoi_dat3 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat3, SMAPdailyCVEX_IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAPdailyCVEX_IRRG_h2osoi_dat4 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat4, SMAPdailyCVEX_IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAPdailyCVEX_IRRG_h2osoi_dat5 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat5, SMAPdailyCVEX_IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAPdailyCVEX_IRRG_h2osoi_dat6 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat6, SMAPdailyCVEX_IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAPdailyCVEX_IRRG_h2osoi_dat7 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat7, SMAPdailyCVEX_IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				SMAPdailyCVEX_IRRG_h2osoi_dat8 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat8, SMAPdailyCVEX_IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
				mcdate6 = concatenate((mcdate6,mcdat6))	

	SMAPdailyCVEX_IRRG_h2osoi_dat = SMAPdailyCVEX_IRRG_h2osoi_dat1*(0.014201/0.041649) + SMAPdailyCVEX_IRRG_h2osoi_dat2*(0.027447/0.041649) 	
	SMAPdailyCVEX_IRRG_QIRRIG_dat = SMAPdailyCVEX_IRRG_QIRRIG_dat * 24 * 3600  #mm/s to mm/day	

	SMAPdailyCVEX_QIRRIG = SMAPdailyCVEX_IRRG_QIRRIG_dat
	SMAPdailyCVEX_h2osoi_5cm = SMAPdailyCVEX_IRRG_h2osoi_dat

	
	df_SMAPdailyCVEX_h2osoi01 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L1'])
	df_SMAPdailyCVEX_h2osoi01['Date'] = pd.to_datetime(mcdate6,format='%Y%m%d')
	df_SMAPdailyCVEX_h2osoi01['Date'] = df_SMAPdailyCVEX_h2osoi01['Date'].dt.date  # just keep the date part and remove time part
	df_SMAPdailyCVEX_h2osoi01['Date'] = list(map(str,df_SMAPdailyCVEX_h2osoi01['Date']))	
	df_SMAPdailyCVEX_h2osoi01[TABLE_Both + '_SMAPdailyCVEX_L1'] = SMAPdailyCVEX_IRRG_h2osoi_dat1
	sum_df1 = sum_df1.merge(df_SMAPdailyCVEX_h2osoi01, how='outer', on=['Date'])

	df_SMAPdailyCVEX_h2osoi02 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L2'])
	df_SMAPdailyCVEX_h2osoi02['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
	df_SMAPdailyCVEX_h2osoi02[TABLE_Both + '_SMAPdailyCVEX_L2'] = SMAPdailyCVEX_IRRG_h2osoi_dat2
	sum_df1 = sum_df1.merge(df_SMAPdailyCVEX_h2osoi02, how='outer', on=['Date'])	

	df_SMAPdailyCVEX_h2osoi1 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L3'])
	df_SMAPdailyCVEX_h2osoi1['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
	df_SMAPdailyCVEX_h2osoi1[TABLE_Both + '_SMAPdailyCVEX_L3'] = SMAPdailyCVEX_IRRG_h2osoi_dat3
	sum_df1 = sum_df1.merge(df_SMAPdailyCVEX_h2osoi1, how='outer', on=['Date'])	
	
	df_SMAPdailyCVEX_h2osoi2 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L4'])
	df_SMAPdailyCVEX_h2osoi2['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
	df_SMAPdailyCVEX_h2osoi2[TABLE_Both + '_SMAPdailyCVEX_L4'] = SMAPdailyCVEX_IRRG_h2osoi_dat4
	sum_df2 = sum_df2.merge(df_SMAPdailyCVEX_h2osoi2, how='outer', on=['Date'])	
	
	df_SMAPdailyCVEX_h2osoi3 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L5'])
	df_SMAPdailyCVEX_h2osoi3['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
	df_SMAPdailyCVEX_h2osoi3[TABLE_Both + '_SMAPdailyCVEX_L5'] = SMAPdailyCVEX_IRRG_h2osoi_dat5
	sum_df3 = sum_df3.merge(df_SMAPdailyCVEX_h2osoi3, how='outer', on=['Date'])

	df_SMAPdailyCVEX_h2osoi4 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L6'])
	df_SMAPdailyCVEX_h2osoi4['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
	df_SMAPdailyCVEX_h2osoi4[TABLE_Both + '_SMAPdailyCVEX_L6'] = SMAPdailyCVEX_IRRG_h2osoi_dat6
	sum_df4 = sum_df4.merge(df_SMAPdailyCVEX_h2osoi4, how='outer', on=['Date'])

	df_SMAPdailyCVEX_h2osoi5 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L7'])
	df_SMAPdailyCVEX_h2osoi5['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
	df_SMAPdailyCVEX_h2osoi5[TABLE_Both + '_SMAPdailyCVEX_L7'] = SMAPdailyCVEX_IRRG_h2osoi_dat7
	sum_df5 = sum_df5.merge(df_SMAPdailyCVEX_h2osoi5, how='outer', on=['Date']) 	

	df_SMAPdailyCVEX_h2osoi6 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L8'])
	df_SMAPdailyCVEX_h2osoi6['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
	df_SMAPdailyCVEX_h2osoi6[TABLE_Both + '_SMAPdailyCVEX_L8'] = SMAPdailyCVEX_IRRG_h2osoi_dat8
	sum_df5 = sum_df5.merge(df_SMAPdailyCVEX_h2osoi6, how='outer', on=['Date']) 	
	
#%%======================================================================= Read SMAP data
	# SMAP_monthly = np.load(SMAP_File)[int(AveMon)-1,smapxy(lati,long)[1],smapxy(lati,long)[0]]
	SMAP_monthly = np.load(SMAP_File)[5:8,smapxy(lati,long)[1],smapxy(lati,long)[0]].mean()
	
#%%======================================================================= Plotting the Whole timeseries

	if 'SCAN' in tables:
		sum_df1[tables] = sum_df1[tables]/100.00
		sum_df2[tables] = sum_df2[tables]/100.00
		sum_df3[tables] = sum_df3[tables]/100.00
		sum_df4[tables] = sum_df4[tables]/100.00
		sum_df5[tables] = sum_df5[tables]/100.00
	
	# Extarct fro June July and August month
	sum_df1_JUNE = sum_df1[sum_df1.Date.str[5:7] == '06']
	sum_df2_JUNE = sum_df2[sum_df2.Date.str[5:7] == '06']
	sum_df3_JUNE = sum_df3[sum_df3.Date.str[5:7] == '06']
	sum_df4_JUNE = sum_df4[sum_df4.Date.str[5:7] == '06']
	sum_df5_JUNE = sum_df5[sum_df5.Date.str[5:7] == '06']    

	sum_df1_JULY = sum_df1[sum_df1.Date.str[5:7] == '07']
	sum_df2_JULY = sum_df2[sum_df2.Date.str[5:7] == '07']
	sum_df3_JULY = sum_df3[sum_df3.Date.str[5:7] == '07']
	sum_df4_JULY = sum_df4[sum_df4.Date.str[5:7] == '07']
	sum_df5_JULY = sum_df5[sum_df5.Date.str[5:7] == '07']  

	sum_df1_AUG = sum_df1[sum_df1.Date.str[5:7] == '08']
	sum_df2_AUG = sum_df2[sum_df2.Date.str[5:7] == '08']
	sum_df3_AUG = sum_df3[sum_df3.Date.str[5:7] == '08']
	sum_df4_AUG = sum_df4[sum_df4.Date.str[5:7] == '08']
	sum_df5_AUG = sum_df5[sum_df5.Date.str[5:7] == '08']  


	sum_df1_Mon = pd.concat([sum_df1_JUNE,sum_df1_JULY,sum_df1_AUG])
	sum_df2_Mon = pd.concat([sum_df2_JUNE,sum_df2_JULY,sum_df2_AUG])
	sum_df3_Mon = pd.concat([sum_df3_JUNE,sum_df3_JULY,sum_df3_AUG])
	sum_df4_Mon = pd.concat([sum_df4_JUNE,sum_df4_JULY,sum_df4_AUG])
	sum_df5_Mon = pd.concat([sum_df5_JUNE,sum_df5_JULY,sum_df5_AUG])   

	# sum_df1_Mon = sum_df1.copy()
	# sum_df2_Mon = sum_df2.copy()
	# sum_df3_Mon = sum_df3.copy()
	# sum_df4_Mon = sum_df4.copy()
	# sum_df5_Mon = sum_df5.copy()   
	
	sum_df01_ind = sum_df1_Mon[['Date'] + [TABLE_Both + '_ctrl_L1'] + [TABLE_Both + '_SMAP_L1'] + [TABLE_Both + '_SMAPdailyCVEX_L1']] 	
	sum_df02_ind = sum_df1_Mon[['Date'] + [TABLE_Both + '_ctrl_L2'] + [TABLE_Both + '_SMAP_L2'] + [TABLE_Both + '_SMAPdailyCVEX_L2']] 
	
	sum_df1_ind = sum_df1_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L3'] + [TABLE_Both + '_SMAP_L3'] + [TABLE_Both + '_SMAPdailyCVEX_L3']]    
	sum_df2_ind = sum_df2_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L4'] + [TABLE_Both + '_SMAP_L4'] + [TABLE_Both + '_SMAPdailyCVEX_L4']]    
	sum_df3_ind = sum_df3_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L5'] + [TABLE_Both + '_SMAP_L5'] + [TABLE_Both + '_SMAPdailyCVEX_L5']]    
	sum_df4_ind = sum_df4_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L6'] + [TABLE_Both + '_SMAP_L6'] + [TABLE_Both + '_SMAPdailyCVEX_L6']]    
	sum_df5_ind = sum_df5_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L7'] + [TABLE_Both + '_SMAP_L7'] + [TABLE_Both + '_SMAPdailyCVEX_L7']]    
	sum_df6_ind = sum_df5_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L8'] + [TABLE_Both + '_SMAP_L8'] + [TABLE_Both + '_SMAPdailyCVEX_L8']] 
	
	array_scan = [sum_df1_ind.mean(0)[0], sum_df2_ind.mean(0)[0], sum_df3_ind.mean(0)[0], sum_df4_ind.mean(0)[0], sum_df5_ind.mean(0)[0]]

	array_ctrl = [sum_df01_ind.mean(0)[0], sum_df02_ind.mean(0)[0], sum_df1_ind.mean(0)[1], sum_df2_ind.mean(0)[1], sum_df3_ind.mean(0)[1], sum_df4_ind.mean(0)[1], sum_df5_ind.mean(0)[1], sum_df6_ind.mean(0)[1]]
	array_smap = [sum_df01_ind.mean(0)[1], sum_df02_ind.mean(0)[1], sum_df1_ind.mean(0)[2], sum_df2_ind.mean(0)[2], sum_df3_ind.mean(0)[2], sum_df4_ind.mean(0)[2], sum_df5_ind.mean(0)[2], sum_df6_ind.mean(0)[2]]

	array_smapDailyCVEX = [sum_df01_ind.mean(0)[2], sum_df02_ind.mean(0)[2], sum_df1_ind.mean(0)[3], sum_df2_ind.mean(0)[3], sum_df3_ind.mean(0)[3], sum_df4_ind.mean(0)[3], sum_df5_ind.mean(0)[3], sum_df6_ind.mean(0)[3]]

	
	ax[-1].scatter(SMAP_monthly,0.025 ,s = 20 , c='black',marker='*', facecolors='black',label='SMAP')

	if 'SCAN' in tables:	
		ax[-1].scatter(array_scan, array((0.05,0.1,0.2,0.5,1))  ,s = 3 , c='black',marker='s', facecolors='black',label='SCAN')
	else:
		ax[-1].scatter(array_scan, array((0.05,0.1,0.2,0.5,1))  ,s = 3 , c='black',marker='s', facecolors='black',label='USCRN')

	ax[-1].plot(array_scan, array((0.05,0.1,0.2,0.5,1)),c='black', linestyle =':', linewidth = 1.1)
	
	ax[-1].scatter(array_ctrl, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038))  ,s = 5 , c='tomato',marker='^', facecolors='tomato',label='CLM_ctrl')
	ax[-1].plot(array_ctrl, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038)),c='tomato', linestyle =':', linewidth = 1.)

	ax[-1].scatter(array_smap, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038))  ,s = 5 , c='dodgerblue',marker='o', facecolors='dodgerblue',label='CLM SMAP_raw')
	ax[-1].plot(array_smap, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038)),c='dodgerblue', linestyle =':', linewidth = 1.) 

	ax[-1].scatter(array_smapDailyCVEX, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038))  ,s = 5 , c='forestgreen',marker='h', facecolors='forestgreen',label='CLM SMAP_KF_BC')
	ax[-1].plot(array_smapDailyCVEX, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038)),c='forestgreen', linestyle =':', linewidth = 1.) 	

	plt.gca().invert_yaxis()
	ax[-1].set_xlim(0,max(SMAP_monthly,max(array_scan),max(array_ctrl),max(array_smap),max(array_smapDailyCVEX))+0.05)
	ax[-1].set_ylim(1.1,-0.1)
	ax[-1].tick_params(axis = 'both', which = 'major', labelsize = 7)
	# IrrPercentage = salmon_data[cyl5minxy(lati,long)[1],cyl5minxy(lati,long)[0]]/area[cyl5minxy(lati,long)[1],cyl5minxy(lati,long)[0]]*100


	if gs2counter not in [0,4]:
		ax[-1].tick_params(axis='y', labelleft='off') 
		ax[-1].tick_params(axis='y',which='both',left=False,right=False)
	
	
	plt.setp(ax[-1].spines.values(), color='darkgray')	
	ax[-1].set_title(TABLE_Both, fontsize=8)
	if gs2counter == 0:
		ax[-1].set_ylabel('Soil Depth (m)', fontsize=10, color='k')
		ax[-1].yaxis.set_label_coords(-0.25, -0.15) 
	if gs2counter == 6:			
		ax[-1].set_xlabel('Soil Moisture Content ' + r'($\mathrm{mm^3/mm^3}$)', fontsize=10, color='k')
		ax[-1].xaxis.set_label_coords(-0.12, -0.2) 
	# if gs2counter == 4:
		# legend_properties = {'size': 10,'weight':'bold'}
		# plt.legend(loc='upper right',prop=legend_properties)
	ax[-1].text(0.22, 0.1, '(' + letter2[gs2counter] + ')', verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes,fontsize=12)
	gs2counter = gs2counter + 1 
  
# savefig( figDIR + 'Fig4_SM_timeseriesLHPA_' + sDate_T + eDate_T + '_Ave_SM_vertical_X021_X024_X024kfBC_JJA' + '_' + sDate_V + '_' + eDate_V + '_v8.png', bbox_inches='tight', dpi=600 )
savefig( figDIR + 'Fig4_SM_timeseriesLPHA_' + sDate_T + eDate_T + '_Ave_SM_vertical_X021_X024_X024kfBC_JJA' + '_' + sDate_V + '_' + eDate_V + '_v8.pdf', bbox_inches='tight')



plt.close()

