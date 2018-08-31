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
#===== First version: 02/01/2018
#===== Vertical SM Profile from CRN, CLM
#%%======================================================================= Plotting spatial maps
DIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/'
figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'
outDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/'

sample3min = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/MyPythonModules/src/X_021_centralUSA_ICRUCLM45BGCCROPNLDAS.clm2.h0.2015-01.nc'
	
lats = np.flip(Dataset(sample3min)['lat'][:],axis=0)
lons = Dataset(sample3min)['lon'][:] - 360

title = ['CLM Control ',\
         'CLM SMAP_Raw ',   
         'CLM SMAP_KF ']          

for YR in ['2005','2010']:
    File_Name = ['CLM_021_cntrlSIM_countyAggregated_irrigationWater_KM3_Year_' + YR + '.npy',\
                 'CLM_024_dailySMAPSIM_countyAggregated_irrigationWater_KM3_Year_' + YR + '.npy',
                 'CLM_024_KF_Final_spatial_temporal_neighbouring_smapERROR_09_P0_2_minCTRLIRRIGandSMAPIRRIG_countyAggregated_irrigationWater_KM3_Year_' + YR + '.npy']

#    USGS_FILE = load(DIR + 'USGSirr_Gridded1to8thdeg_withdrawals_freshwater_Mgal_d_year' + YR + '.npy')
#    USGS_FILE = flip(USGS_FILE,axis=0)*1e6*0.00378541*365*1e-9 # unit conversion from Mgal/d to KM^3/y
#    ny = 400; nx = 520
#    USGS_reg = zeros((ny,nx))    
#    for i in range(ny):
#        print(i)
#        for j in range(nx):
#            USGS_reg[i,j] = USGS_FILE[cylNorthAmerica1_8degxy(lats[i],lons[j])[1],cylNorthAmerica1_8degxy(lats[i],lons[j])[0]]
#    np.save( outDIR + 'USGS_CountyLevel_Irrigation_CentralUS3min_Year_' + YR, USGS_reg)
    USGS_reg = load(outDIR + 'USGS_CountyLevel_Irrigation_CentralUS3min_Year_' + YR + '.npy').reshape(400*520,1)    
    
    
    for counter in range(3):
    
        IRRG_FILE = load(DIR + File_Name[counter]).reshape(400*520,1)
        ols = pysal.spreg.ols.OLS(USGS_reg, IRRG_FILE, name_y='USGS', name_x=[title[counter]], name_ds='USGS VS ' + title[counter] + ' ' + YR, white_test=True)
        # print(ols.summary)
        





 
#%%======================================================================= Plotting the scatters
srcDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/'

STstring = ['NE', 'KS', 'TX']


for YR in ['2005','2010']:
    dfmain = pd.read_csv(srcDIR + 'County_Irrigations_USGC_CLM_IrrAmount_km3_year' + YR + '_X21_X24dialy_convex_vs_X24KF_Final_spatial_temporal_neighbouring_smapERROR_09_P0_2_minCTRLIRRIGandSMAPIRRIG.txt',sep=',', header=None, error_bad_lines=False)
    dfmain.columns=['State_County','USGS_Irr','X_021_Irr','X_024_Irr','X_024_kf_Irr']
    for counter in range(3):
    
        df = dfmain[dfmain.State_County.str.startswith(STstring[counter])]  
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(df.USGS_Irr.astype(float),df.X_021_Irr.astype(float))
        RMSE1 = sqrt(np.mean((df.USGS_Irr.astype(float)-df.X_021_Irr.astype(float))**2))
        nash1 = 1 - sum((df.USGS_Irr.astype(float)-df.X_021_Irr.astype(float))**2)/sum((df.USGS_Irr.astype(float) - np.mean(df.USGS_Irr.astype(float)))**2)
        MSD1 = sum((df.X_021_Irr.astype(float) - df.USGS_Irr.astype(float)))/len(df.X_021_Irr)  #mean signed deviation


        print('=======================================================')
        print(STstring[counter], YR)
        print('X_021 Simulation VS USGS')
        print('RMSE1,  nash1,  MSD1  ',RMSE1,nash1,MSD1)


            
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df.USGS_Irr.astype(float),df.X_024_Irr.astype(float))
        RMSE2 = sqrt(np.mean((df.USGS_Irr.astype(float)-df.X_024_Irr.astype(float))**2))
        nash2 = 1 - sum((df.USGS_Irr.astype(float)-df.X_024_Irr.astype(float))**2)/sum((df.USGS_Irr.astype(float) - np.mean(df.USGS_Irr.astype(float)))**2)
        MSD2 = sum((df.X_024_Irr.astype(float) - df.USGS_Irr.astype(float)))/len(df.X_024_Irr)  #mean signed deviation


        print('')
        print('X_024 Simulation VS USGS')
        print('RMSE2  ,nash2,  MSD2  ',RMSE2,nash2,MSD2)
   
    
        slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(df.USGS_Irr.astype(float),df.X_024_kf_Irr.astype(float))
        RMSE3 = sqrt(np.mean((df.USGS_Irr.astype(float)-df.X_024_kf_Irr.astype(float))**2))
        nash3 = 1 - sum((df.USGS_Irr.astype(float)-df.X_024_kf_Irr.astype(float))**2)/sum((df.USGS_Irr.astype(float) - np.mean(df.USGS_Irr.astype(float)))**2)
        MSD3 = sum((df.X_024_kf_Irr.astype(float) - df.USGS_Irr.astype(float)))/len(df.X_024_kf_Irr)  #mean signed deviation

        t_stat1, p_val1 = stats.ttest_rel(df.X_021_Irr.astype(float),df.X_024_Irr.astype(float),nan_policy = 'omit')
        t_stat2, p_val2 = stats.ttest_rel(df.X_021_Irr.astype(float),df.X_024_kf_Irr.astype(float),nan_policy = 'omit')
               
        print('')
        print('X_024_KF Simulation VS USGS')
        print('RMSE3  ,nash3  ,MSD3  ',RMSE3,nash3,MSD3)
  

        if p_val1 < 0.05:
            aa = 'X021 and X024 are significantly different'
        else:
            aa = 'NO SIGNIFICANT DIFFERENCE!'
        if p_val2 < 0.05:
            bb = 'X021 and X024KF are significantly different'
        else:
            bb = 'NO SIGNIFICANT DIFFERENCE!'
            
            
        print('Calculate the T-test on TWO RELATED samples a and b')     
        print('ttest_ind X021 VS X024 =====> t-stats, p-value',t_stat1, p_val1, '  ', aa) 
        print('ttest_ind X021 VS X024KF =====> t-stats, p-value',t_stat2, p_val2, '  ',bb) 

