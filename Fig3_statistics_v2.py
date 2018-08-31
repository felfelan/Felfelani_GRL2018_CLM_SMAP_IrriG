#!~/anaconda3/DIR/bin/ipython

from __future__ import division # force division to be floating point in Python
from numpy import *
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.gridspec import GridSpec
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
from mpl_toolkits.basemap import Basemap,shiftgrid
# My module to change the latlon to xy coordination in SMAP data
from latlon2xy import smapxy, cyl5minxy, cntralUSA3minxy, cylNorthAmerica1_8degxy
import pysal
#%%=======================================================================
#===== Farshid Felfelani
#===== First version: 08/26/2018
#===== Statistics for Irrigation Amount
#%%======================================================================= Plotting the scatters
srcDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src_v2/'

STstring  = ['NE', 'KS', 'TX']
YEARs     = ['2010','2005','2000','1995','1990','1985']
FileName1 = ['CLM_021_','CLM_024_','CLM_024_','CLM_024_']
FileName2 = ['_dlyCTRLsim_countyAggregated_irrigationWater_KM3.txt', \
             '_dlySMAPsim_countyAggregated_irrigationWater_KM3_no_BiasCorrection.txt', \
             '_dSMAPKFsim_countyAggregated_irrigationWater_KM3_no_BiasCorrection.txt', \
             '_dSMAPKFsim_countyAggregated_irrigationWater_KM3_grndOBS_BiasCorrection_minIRR.txt']
Labels    = ['CLM_021','CLM_024','CLM_024KF','CLM_024KFBC']

for YR in YEARs:
    dfusgs = pd.read_csv(srcDIR + 'USGSdat_' + YR + '_countyAggregated_irrigationWater_KM3.txt' ,sep=',', header=None, error_bad_lines=False)
    dfusgs.columns=['State','County_ID','USGS_Irrigation_KM3']
    dfmain = dfusgs.copy()

    for ff in range(len(FileName1)):
        dfdummy = pd.read_csv(srcDIR + FileName1[ff] + YR + FileName2[ff] ,sep=',', header=None, error_bad_lines=False)
        dfdummy.columns=['State','County_ID',Labels[ff]]
        dfmain = dfmain.merge(dfdummy, how='outer', on=['County_ID'])

    print('=======================================================')

    for counter in range(3):

        df = dfmain[dfmain.State.str.startswith(STstring[counter])]  
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(df.USGS_Irrigation_KM3.astype(float),df.CLM_021.astype(float))
        RMSE1 = sqrt(np.mean((df.USGS_Irrigation_KM3.astype(float)-df.CLM_021.astype(float))**2))
        nash1 = 1 - sum((df.USGS_Irrigation_KM3.astype(float)-df.CLM_021.astype(float))**2)/sum((df.USGS_Irrigation_KM3.astype(float) - np.mean(df.USGS_Irrigation_KM3.astype(float)))**2)
        MSD1 = sum((df.CLM_021.astype(float) - df.USGS_Irrigation_KM3.astype(float)))/len(df.CLM_021)  #mean signed deviation


        print('=======================================================')
        print(STstring[counter], YR)
        print('X_021 Simulation VS USGS')
        print('RMSE1,  MSD1, nash1  ',RMSE1,MSD1,nash1)
            
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df.USGS_Irrigation_KM3.astype(float),df.CLM_024.astype(float))
        RMSE2 = sqrt(np.mean((df.USGS_Irrigation_KM3.astype(float)-df.CLM_024.astype(float))**2))
        nash2 = 1 - sum((df.USGS_Irrigation_KM3.astype(float)-df.CLM_024.astype(float))**2)/sum((df.USGS_Irrigation_KM3.astype(float) - np.mean(df.USGS_Irrigation_KM3.astype(float)))**2)
        MSD2 = sum((df.CLM_024.astype(float) - df.USGS_Irrigation_KM3.astype(float)))/len(df.CLM_024)  #mean signed deviation


        print('')
        print('X_024 Simulation VS USGS')
        print('RMSE2  ,MSD2  ,nash2   ',RMSE2,MSD2,nash2)
   
    
        slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(df.USGS_Irrigation_KM3.astype(float),df.CLM_024KF.astype(float))
        RMSE3 = sqrt(np.mean((df.USGS_Irrigation_KM3.astype(float)-df.CLM_024KF.astype(float))**2))
        nash3 = 1 - sum((df.USGS_Irrigation_KM3.astype(float)-df.CLM_024KF.astype(float))**2)/sum((df.USGS_Irrigation_KM3.astype(float) - np.mean(df.USGS_Irrigation_KM3.astype(float)))**2)
        MSD3 = sum((df.CLM_024KF.astype(float) - df.USGS_Irrigation_KM3.astype(float)))/len(df.CLM_024KF)  #mean signed deviation
               
        print('')
        print('X_024_KF Simulation VS USGS')
        print('RMSE3  ,MSD3  ,nash3  ',RMSE3,MSD3,nash3)

		
        slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(df.USGS_Irrigation_KM3.astype(float),df.CLM_024KFBC.astype(float))
        RMSE4 = sqrt(np.mean((df.USGS_Irrigation_KM3.astype(float)-df.CLM_024KFBC.astype(float))**2))
        nash4 = 1 - sum((df.USGS_Irrigation_KM3.astype(float)-df.CLM_024KFBC.astype(float))**2)/sum((df.USGS_Irrigation_KM3.astype(float) - np.mean(df.USGS_Irrigation_KM3.astype(float)))**2)
        MSD4  = sum((df.CLM_024KFBC.astype(float) - df.USGS_Irrigation_KM3.astype(float)))/len(df.CLM_024KFBC)  #mean signed deviation

        print('')
        print('X_024_KFBC Simulation VS USGS')
        print('RMSE4  ,MSD4  ,nash4  ',RMSE4,MSD4,nash4)
  
        t_stat1, p_val1 = stats.ttest_rel(df.CLM_021.astype(float), df.CLM_024.astype(float),nan_policy = 'omit')
        t_stat2, p_val2 = stats.ttest_rel(df.CLM_021.astype(float), df.CLM_024KF.astype(float),nan_policy = 'omit')
        t_stat3, p_val3 = stats.ttest_rel(df.CLM_021.astype(float), df.CLM_024KFBC.astype(float),nan_policy = 'omit')  
  
        if p_val1 < 0.05:
            aa = 'X021 and X024 are significantly different'
        else:
            aa = 'NO SIGNIFICANT DIFFERENCE!'
        if p_val2 < 0.05:
            bb = 'X021 and X024KF are significantly different'
        else:
            bb = 'NO SIGNIFICANT DIFFERENCE!'
        if p_val3 < 0.05:
            cc = 'X021 and X024KFBC are significantly different'
        else:
            cc = 'NO SIGNIFICANT DIFFERENCE!'
                      
        print('Calculate the T-test on TWO RELATED samples a and b')     
        print('ttest_ind X021 VS X024 =====> t-stats, p-value',t_stat1, p_val1, '  ', aa) 
        print('ttest_ind X021 VS X024KF =====> t-stats, p-value',t_stat2, p_val2, '  ',bb) 
        print('ttest_ind X021 VS X024KFBC =====> t-stats, p-value',t_stat3, p_val3, '  ',cc) 

