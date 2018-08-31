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
CLM_SMAP_daily_convex = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr/'
# CLM_SMAP_daily_KF = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF/'
CLM_SMAP_daily_KF = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF/grndOBS_BiasCorrection_minIRR/'
SMAP_File = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/SMAPorig_12_monthly_ave_for_20150331_20170801.npy'
dirSTATIONS = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/scan_uscrn_snotel/'
SCAN_CRN_DIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/scan_uscrn_snotel/CRN_SCAN_Selected_Stations/'
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
 
#st_names = ['SCAN_2105','SCAN_2104','SCAN_2106','SCAN_2107','SCAN_2108']
#latseries = [33.55,33.62, 33.63,33.54,33.53]
#lonseries = [-102.37,-102.04, -102.75,-103.24,-103.63]


#st_names = ['SCAN_2104_TX','SCAN_2105_TX','SCAN_2106_TX','SCAN_2111_NE','USCRN_94996_NE']
#latseries = [33.62,33.55,33.63,40.37,40.69]
#lonseries = [-102.04,-102.37,-102.75,-101.72,-96.85]

st_names = ['SCAN_2105_TX','SCAN_2106_TX','SCAN_2107_NM','SCAN_2111_NE']
latseries = [33.55,33.63,33.54,40.37]
lonseries = [-102.37,-102.75,-103.24,-101.72]

gs1 = gridspec.GridSpec(2, 2, height_ratios = [1,1], width_ratios = [1,1]) # the middle row is to place the colorbar axes
gs1.update(wspace=0.10, hspace = 0.10)
#gs1.update(bottom=0.6, top=0.98)

fig = plt.figure(num=1, figsize=(25,10)) #figsize: w,h
ax = []
#%%======================================================================= read SCAN data
df_SCANstations = pd.read_csv(dirSTATIONS + 'SCAN_stations_ID_latlon.txt',sep='\t',header=0, error_bad_lines=False)

my_dict = {}
# Loop over the each file which represents a station data
# Save the SCAN data as a dataframe
# Appending each station as an item in dictionary
nSTATION = 0 
for file in sorted(os.listdir(SCAN_CRN_DIR)):
    if os.path.isfile(os.path.join(SCAN_CRN_DIR, file)):  #list only files not directories
        print(file)
        nSTATION = nSTATION + 1
        if 'SCAN' in file:
        # Getting the row number of the header
            aa = open(SCAN_CRN_DIR + file).readlines()
            for ii in range(len(aa)):
                if not fnmatch.fnmatch(aa[ii],'#*'):
                                   headerINDEX = ii
                                   break
            
            # Reading from the header below
            df = pd.read_csv(SCAN_CRN_DIR +file,sep=',', header=headerINDEX, error_bad_lines=False)
        else:
            
            df = pd.read_csv(SCAN_CRN_DIR +file,sep=',', header=0, error_bad_lines=False)
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
        print(df_SCAN_T.iloc[ii,0])
        Leap_indx = Leap_indx + [df_SCAN_T.index[ii]]  
df_T_noLeap = df_SCAN_T.drop(Leap_indx)

#%%======================================================================= read Control CLM data
for cc in range(len(latseries)):
    clm3minxy = cntralUSA3minxy(latseries[cc],lonseries[cc])
       
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
    for file in sorted(os.listdir(CLM_SMAP_daily_convex)):
        if fnmatch.fnmatch(file, '*00000.nc') and int(file[60:64])>= sYR_T and int(file[60:64])<= eYR_T:
            SMAP_IRRG_FILE = netcdf.netcdf_file(CLM_SMAP_daily_convex + file,'r')
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

   
    #%%======================================================================= read SMAP KF CLM data
    xx = 0
    for file in sorted(os.listdir(CLM_SMAP_daily_KF)):
        if fnmatch.fnmatch(file, '*00000.nc') and int(file[63:67])>= sYR_T and int(file[63:67])<= eYR_T:
            SMAP_KF_IRRG_FILE = netcdf.netcdf_file(CLM_SMAP_daily_KF + file,'r')
            nt = len(SMAP_KF_IRRG_FILE.variables['QIRRIG'][:])
            mcdat3 = SMAP_KF_IRRG_FILE.variables['mcdate'][:] 
            
            SMAP_KF_IRRG_h2osoi_dat0 = SMAP_KF_IRRG_FILE.variables['H2OSOI'][:]	
            SMAP_KF_IRRG_h2osoi_dat01 = flip(SMAP_KF_IRRG_h2osoi_dat0[:,0,].reshape(nt,400,520),axis=1) 
            SMAP_KF_IRRG_h2osoi_dat02 = flip(SMAP_KF_IRRG_h2osoi_dat0[:,1,].reshape(nt,400,520),axis=1)
            QIRRIG0 = np.flip(SMAP_KF_IRRG_FILE.variables['QIRRIG'][:].reshape(nt,400,520),axis=1)
            
            if file[63:67] == str(sYR_T) and xx==0:
                print('file0:',file)
                SMAP_KF_IRRG_h2osoi_dat1 = SMAP_KF_IRRG_h2osoi_dat01[:,clm3minxy[1],clm3minxy[0]]
                SMAP_KF_IRRG_h2osoi_dat2 = SMAP_KF_IRRG_h2osoi_dat02[:,clm3minxy[1],clm3minxy[0]]
                QIRRIG_dat  = QIRRIG0[:,clm3minxy[1],clm3minxy[0]]
                xx = 1
                mcdate3 = mcdat3       
            else:
                print('file:',file)
                SMAP_KF_IRRG_h2osoi_dat1 = concatenate(( SMAP_KF_IRRG_h2osoi_dat1, SMAP_KF_IRRG_h2osoi_dat01[:,clm3minxy[1],clm3minxy[0]] ))
                SMAP_KF_IRRG_h2osoi_dat2 = concatenate(( SMAP_KF_IRRG_h2osoi_dat2, SMAP_KF_IRRG_h2osoi_dat02[:,clm3minxy[1],clm3minxy[0]] ))
                mcdate3 = concatenate((mcdate3,mcdat3))
                QIRRIG_dat = np.concatenate(( QIRRIG_dat, QIRRIG0[:,clm3minxy[1],clm3minxy[0]] ))
    QIRRIG_dat = QIRRIG_dat * 24 * 3600    # mm/s to mm/day	                   
    SMAP_KF_IRRG_h2osoi_dat = SMAP_KF_IRRG_h2osoi_dat1*(0.014201/0.041649) + SMAP_KF_IRRG_h2osoi_dat2*(0.027447/0.041649) 	
    
    SMAP_KF_h2osoi_5cm = SMAP_KF_IRRG_h2osoi_dat


    df_SMAP_KF_h2osoi_5cmT = pd.DataFrame(columns=['Date','SMAP_KF_h2osoi_5cm_' + st_names[cc],'SMAP_KF_qirrig_' + st_names[cc]])
    df_SMAP_KF_h2osoi_5cmT['Date'] = pd.to_datetime(mcdate3,format='%Y%m%d')
    df_SMAP_KF_h2osoi_5cmT['Date'] = df_SMAP_KF_h2osoi_5cmT['Date'].dt.date  # just keep the date part and remove time part
    df_SMAP_KF_h2osoi_5cmT['Date'] = list(map(str,df_SMAP_KF_h2osoi_5cmT['Date']))
    df_SMAP_KF_h2osoi_5cmT['SMAP_KF_h2osoi_5cm_' + st_names[cc]] = SMAP_KF_h2osoi_5cm
    df_SMAP_KF_h2osoi_5cmT['SMAP_KF_qirrig_' + st_names[cc]] = QIRRIG_dat
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

    sc1 = ax[-1].scatter(df_JJA.index,df_JJA[st_names[cc]] ,s = 7 , c='black',marker="s",\
                           label='SCAN Observations')


    
#    ax[-1].fill_between(df_JJA.index, df_JJA.iloc[:,1:nSTATION + 1].min(1), df_JJA.iloc[:,1:nSTATION + 1].max(1), facecolor='lightgrey',edgecolor='lightgrey', \
#                    interpolate=True,label='SCAN/CRN Range', zorder=-1)
    
#    clmctrl_col = [col for col in list(df_T_noLeap) if col.startswith('ctrl_h2osoi_5cm')]
#    clmsmap_col = [col for col in list(df_T_noLeap) if col.startswith('SMAP_h2osoi_5cm')]
#    clmsmapKF_col = [col for col in list(df_T_noLeap) if col.startswith('SMAP_KF_h2osoi_5cm')]
#    
#    ax[-1].scatter(df_JJA.index, df_JJA[clmctrl_col].copy().mean(1),s = 5 , c='tomato',label='CLM Control')
#    ax[-1].scatter(df_JJA.index, df_JJA[clmsmap_col].copy().mean(1),s = 5, c='dodgerblue',label='CLM SMAP_Raw')
#    ax[-1].scatter(df_JJA.index, df_JJA[clmsmapKF_col].copy().mean(1),s = 5, c='forestgreen',label='CLM SMAP_KF')



    sc2 = ax[-1].scatter(df_JJA.index, df_JJA['ctrl_h2osoi_5cm_' + st_names[cc]].copy(),s = 7 , c='tomato',label='CLM Control')
    sc3 = ax[-1].scatter(df_JJA.index, df_JJA['SMAP_h2osoi_5cm_' + st_names[cc]].copy(),s = 7, c='dodgerblue',label='CLM SMAP_raw')
    sc4 = ax[-1].scatter(df_JJA.index, df_JJA['SMAP_KF_h2osoi_5cm_' + st_names[cc]].copy(),s = 7, c='forestgreen',label='CLM SMAP_kf_BC')

    ln2 = ax[-1].plot(df_JJA.index, df_JJA['ctrl_h2osoi_5cm_' + st_names[cc]].copy(),linewidth=0.5,c='tomato',label='CLM Control')
    ln3 = ax[-1].plot(df_JJA.index, df_JJA['SMAP_h2osoi_5cm_' + st_names[cc]].copy(),linewidth=0.5,c='dodgerblue',label='CLM SMAP_raw')
    ln4 = ax[-1].plot(df_JJA.index, df_JJA['SMAP_KF_h2osoi_5cm_' + st_names[cc]].copy(),linewidth=0.5,c='forestgreen',label='CLM SMAP_kf_BC')


    ax2 = ax[-1].twinx()
    ba1 = ax2.bar(df_JJA.index,df_JJA['SMAP_KF_qirrig_' + st_names[cc]],color='grey',width=1.0,alpha=0.5,label='Irrigation Water SMAP_kf_BC')

  
    gca().set_xticklabels([''])  # remove the xtick labels
    ax[-1].set_xticks(df_JJA.index[0:len(df_JJA.index):30])
    # size of the xticks and yticks labels
    ax[-1].tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax2.tick_params(axis = 'y', which = 'major', labelsize = 10)
    
    
    if cc == 3 or cc ==2:    
        ax[-1].set_xticklabels(df_JJA.Date[0:len(df_JJA.index):30], fontsize=10, rotation=-45)
        for tick in ax[-1].xaxis.get_majorticklabels():
            tick.set_horizontalalignment("left")
        minorLocator = MultipleLocator(10)
        ax[-1].xaxis.set_minor_locator(minorLocator)
        
        if cc ==2:
            legend_properties = {'size': 10}     # phont size
            lgnd = ax[-1].legend(handles=[sc1,sc2,sc3,sc4,ba1],loc='upper left',prop=legend_properties,ncol=3)
            lgnd.legendHandles[0]._sizes = [40]  # size of the scatter in the legend
            lgnd.legendHandles[1]._sizes = [40]
            lgnd.legendHandles[2]._sizes = [40]
            lgnd.legendHandles[3]._sizes = [40]
            
#        else:
#            ax[-1].legend_.remove()

    else:
#        ax[-1].legend_.remove()
        minorLocator = MultipleLocator(10)
        ax[-1].xaxis.set_minor_locator(minorLocator)
            
    ax[-1].tick_params(direction='inout', length=6, width=2, colors='k')
    if cc ==2:
        ax[-1].set_ylabel('Soil Moisture Content ' + r'($\mathrm{mm^3/mm^3}$)', fontsize=15, color='k')
        ax[-1].yaxis.set_label_coords(-0.05, 1.0) 

    else:
        y_axis = ax[-1].axes.get_yaxis()
        y_label = y_axis.get_label()
        y_label.set_visible(False)
       
    #ax1 = plt.axes()
    x_axis = ax[-1].axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    
    plt.xlim(df_JJA.index.min(),df_JJA.index.max())
    # plt.ylim(0,0.5)
    plt.title('Top 5cm Soil Moisture for JJA - ' +  st_names[cc],fontsize=15)    
    
    if cc == 3:
        ax2.set_ylabel('Irrigation Water Amount ' + r'($\mathrm{mm/day}$)', fontsize=15, color='k')
        ax2.yaxis.set_label_coords(1.05, 1)    
   
savefig( figDIR + 'Fig4_SM_timeseries' + sDate_T + eDate_T + '_v3.png', bbox_inches='tight', dpi=600 )
#savefig( figDIR + 'Fig4_SM_timeseries' + sDate_T + eDate_T + '_v3.pdf', bbox_inches='tight')

plt.close()
