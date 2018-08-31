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
#%%=======================================================================
#===== Farshid Felfelani
#===== First version: 02/01/2018
#===== Vertical SM Profile from CRN, CLM
#%%======================================================================= Plotting spatial maps
DIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/'

YR = '2005'

File_Name = ['USGSirr_Gridded1to8thdeg_withdrawals_freshwater_Mgal_d_year' + YR + '.npy',\
             'CLM_021_cntrlSIM_countyAggregated_irrigationWater_KM3_Year_' + YR + '.npy',\
             # 'CLM_022_monthlySMAPSIM_countyAggregated_irrigationWater_KM3_Year_2010.npy',\
             'CLM_024_grndOBS_BIAScor_dailySMAPSIM_countyAggregated_irrigationWater_KM3_v5_Year_' + YR + '.npy',
             # 'CLM_024_kf_dailySMAPSIM_minSMclmSMsmap_smaperror09_countyAggregated_irrigationWater_KM3_Year_2010.npy']
             # 'CLM_024_kf_dailySMAPSIM_noMIN_smaperror09_countyAggregated_irrigationWater_KM3_Year_2010.npy']              
             # 'CLM_024_dailySMAPSIM_run20180318_NOkf_minSMctrlSMsmap_countyAggregated_irrigationWater_KM3_Year_2010.npy']
             # 'CLM_024_KF_run20180304_spatial_temporal_neighbouring_smapERROR_09_P0_2_countyAggregated_irrigationWater_KM3_Year_2010.npy']
             # 'CLM_024_dailySMAPSIM_run20180319kf_minIRRctrlIRRsmapKF_smapERROR_09_countyAggregated_irrigationWater_KM3_Year_2010.npy']
             'CLM_024_KF_grndOBS_BIAScor_spatial_temporal_neighbouring_smapERROR_09_P0_2_minCTRLIRRIGandSMAPIRRIG_countyIrrWat_KM3_Year_' + YR + '.npy']
title = ['USGS ',\
         'CLM Control Simulation',\
         # 'CLM SMAP Simulation (monthly) ',\
         'CLM SMAP_raw_BC ', \
         'CLM SMAP_kf_BC']      		 
         # 'CLM SMAP Spatiotemporal KF daily P0.2\n SMAPErr09 minCTRLIRRIGandSMAPIRRIG']          
figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'


# gs = gridspec.GridSpec(3, 12, height_ratios=[1,0.02,1],width_ratios=[1,1,1,1,1,1,1,1,1,1,1,1]) # the middle row is to place the colorbar axes
# gs.update(wspace=0.1) 
# gs.update(hspace=0.1)

gs1 = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,1]) # the middle row is to place the colorbar axes
gs1.update(bottom=0.35, top=0.98, wspace=0.02)

gs2 = gridspec.GridSpec(1, 3, width_ratios=[1,1,1],height_ratios=[1]) # the middle row is to place the colorbar axes
gs2.update(bottom=0.05, top=0.4, wspace=0.15)

fig = plt.figure(num=1, figsize=(13,8))
ax = []

for counter in range(4):

    IRRG_FILE = load(DIR + File_Name[counter])
    if counter == 0:
       IRRG_FILE = flip(IRRG_FILE,axis=0)*1e6*0.00378541*365*1e-9
       # unit conversion from Mgal/d to KM^3/y

    cmap = plt.cm.gist_ncar
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist = cmaplist[30:-30]
    # cmaplist[0] = (0.1,0.1,0.1,0.1)
    cmaplist[0] = (0.05,0.05,0.05,0.05)
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    #bounds = np.linspace(0,3,16)
    bounds_b = array([0,0.005, 0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0])
    bounds = ['0.0','0.005', '0.01','0.05','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.2','1.4','1.6','1.8','2.0','2.5','3.0']
   # I changed the ticks labels to string to have variable number of digits to the right of decimal point 
    norm = mpl.colors.BoundaryNorm(bounds_b, cmap.N)

    nxa=-116
    nxb=-90
    nya=50
    nyb=30
    res=1
    LT1 = 0.6
    LT2 = 0.6
    ax.append(fig.add_subplot(gs1[counter]))
    ax[-1].map=Basemap( projection ='cyl',  \
                llcrnrlon  = nxa,  \
                urcrnrlon  = nxb,  \
                llcrnrlat  = nyb,  \
                urcrnrlat  =  nya,  \
                resolution = "c")
    ax[-1].map.drawcoastlines(linewidth=LT1, color='black')
    ax[-1].map.drawcountries(linewidth=LT2, color='grey')
    ax[-1].map.drawstates(linewidth=0.25, color='grey')
    if counter == 0:
        cs = ax[-1].map.imshow(IRRG_FILE[cylNorthAmerica1_8degxy(50,-116)[1]:cylNorthAmerica1_8degxy(30,-116)[1],cylNorthAmerica1_8degxy(50,-116)[0]:cylNorthAmerica1_8degxy(50,-90)[0]]\
                            ,origin='upper',interpolation='quadric',cmap=cmap,norm=norm)
    else:
        cs = ax[-1].map.imshow(IRRG_FILE,origin='upper',interpolation='quadric',cmap=cmap,norm=norm)
    ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=0.7)
    ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_07_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.4)
    ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_08_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.4)
    ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_10_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.4)
    ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_11_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.4)
    ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_12_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.4)
    ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_13_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.4)
    ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_14_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.4)
    ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_15_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.4)
    ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_16_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.4)
    ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_17_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.4)
    plt.title(title[counter],fontsize=10) 


##ax.append(fig.add_subplot(gs[1,1]))
##fig.colorbar(cs, cax=ax[-1], cmap=cmap, norm=norm,spacing='uniform', ticks=bounds[0::2], boundaries=bounds, orientation='horizontal', extend='max')
cbar_ax = fig.add_axes([0.2, 0.5, 0.6, 0.02])    
cb = fig.colorbar(cs, cax=cbar_ax, cmap=cmap, norm=norm,spacing='uniform', ticks=bounds_b[0::1], boundaries=bounds_b, orientation='horizontal', extend='max')
cb.set_label('Irrigation Water\n' + YR + ' - $km^3/y$',fontsize=10,labelpad=-29, x=1.1)
cb.ax.set_xticklabels(bounds) # I changed the ticks labels to string to have variable number of digits to the right of decimal point    
#%%======================================================================= Plotting the scatters
srcDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/'
df1 = pd.read_csv(srcDIR + 'County_Irrigations_USGC_CLM_Irrarea_JJA_year2010_x22NEW(convex).txt',sep=',', header=None, error_bad_lines=False)
df2 = pd.read_csv(srcDIR + 'County_Irrigations_USGC_CLM_Irrarea_JJA_year2005.txt',sep=',', header=None, error_bad_lines=False)
#df3 = pd.read_csv(srcDIR + 'County_Irrigations_USGC_CLM_Irrarea_JJA_year2000.txt',sep=',', header=None, error_bad_lines=False)

# dfmain = pd.read_csv(srcDIR + 'County_Irrigations_USGC_CLM_IrrAmount_km3_year2010_x24dialy(convex)KF_minSMclmSMsmap_error09.txt',sep=',', header=None, error_bad_lines=False)
# dfmain = pd.read_csv(srcDIR + 'County_Irrigations_USGC_CLM_IrrAmount_km3_year2010_x24dialy(convex)run20180319kf_minIRRctrlIRRsmapKF_smapERROR_09.txt',sep=',', header=None, error_bad_lines=False)
# dfmain = pd.read_csv(srcDIR + 'County_Irrigations_USGC_CLM_IrrAmount_km3_year2010_X21_X24dialy_convex_vs_X24KF_spatial_temporal_neighbouring_smapERROR_09_P0_2.txt',sep=',', header=None, error_bad_lines=False)
dfmain = pd.read_csv(srcDIR + 'County_Irrigations_USGC_CLM_IrrAmount_km3_year_' + YR + '_X21_X24_vs_X24KF_grndOBS_BIAScor_spatial_temporal_neighbouring_smapERROR_09_P0_2_minCTRLIRRIGandSMAPIRRIG.txt',sep=',', header=None, error_bad_lines=False)
# dfmain = df1
# dfmain = pd.concat([df1,df2],axis = 0)
dfmain.columns=['State_County','USGS_Irr','X_021_Irr','X_024_Irr','X_024_kf_Irr']

STstring = ['NE', 'KS', 'TX']

# year = '2005 and 2010'
# removing nan
# dfmain = dfmain[dfmain.X_021_Irr.str.contains('nan') == False]


#f, ax = plt.subplots()

for counter in range(3):
    ax.append(fig.add_subplot(gs2[counter]))

    df = dfmain[dfmain.State_County.str.startswith(STstring[counter])]
    ax[-1].plot(df.USGS_Irr.astype(float),df.X_021_Irr.astype(float) , ls="", markersize= 5 , markeredgecolor='black',marker='o', markerfacecolor='none',label='CLM CONTORL')
    #ax.scatter(df.USGS_Irr.astype(float),df.X_022_Irr.astype(float) ,s = 20 , c='orangered',marker='<', facecolors='none',label='X_022_Irr')
    ax[-1].plot(df.USGS_Irr.astype(float),df.X_024_Irr.astype(float) , ls="", markersize = 5 , markeredgecolor='orangered',marker='<', markerfacecolor='none',label='SMAP_raw_BC')

    # ax[-1].plot(df.USGS_Irr.astype(float),df.X_024_Irr.astype(float) , ls="", markersize = 5 , markeredgecolor='dodgerblue',marker='s', markerfacecolor='none',label='SMAP DAYILY')
    ax[-1].plot(df.USGS_Irr.astype(float),df.X_024_kf_Irr.astype(float) , ls="", markersize = 5 , markeredgecolor='dodgerblue',marker='s', markerfacecolor='none',label='SMAP_kf_BC minIRRIG')

    ax[-1].set_yscale('log')
    ax[-1].set_xscale('log')
    
    if counter == 1:
        plt.legend(loc='upper left',fontsize=8)


    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(df.USGS_Irr.astype(float),df.X_021_Irr.astype(float))
    RMSE1 = sqrt(np.mean((df.USGS_Irr.astype(float)-df.X_021_Irr.astype(float))**2))
    nash1 = 1 - sum((df.USGS_Irr.astype(float)-df.X_021_Irr.astype(float))**2)/sum((df.USGS_Irr.astype(float) - np.mean(df.USGS_Irr.astype(float)))**2)
    MSD1 = sum((df.X_021_Irr.astype(float) - df.USGS_Irr.astype(float)))/len(df.X_021_Irr)  #mean signed deviation


    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df.USGS_Irr.astype(float),df.X_024_Irr.astype(float))
    RMSE2 = sqrt(np.mean((df.USGS_Irr.astype(float)-df.X_024_Irr.astype(float))**2))
    nash2 = 1 - sum((df.USGS_Irr.astype(float)-df.X_024_Irr.astype(float))**2)/sum((df.USGS_Irr.astype(float) - np.mean(df.USGS_Irr.astype(float)))**2)
    MSD2 = sum((df.X_024_Irr.astype(float) - df.USGS_Irr.astype(float)))/len(df.X_024_Irr)  #mean signed deviation


    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(df.USGS_Irr.astype(float),df.X_024_kf_Irr.astype(float))
    RMSE3 = sqrt(np.mean((df.USGS_Irr.astype(float)-df.X_024_kf_Irr.astype(float))**2))
    nash3 = 1 - sum((df.USGS_Irr.astype(float)-df.X_024_kf_Irr.astype(float))**2)/sum((df.USGS_Irr.astype(float) - np.mean(df.USGS_Irr.astype(float)))**2)
    MSD3 = sum((df.X_024_kf_Irr.astype(float) - df.USGS_Irr.astype(float)))/len(df.X_024_kf_Irr)  #mean signed deviation


    #ax[-1].text(0.03, 0.94, '$R_{CTRL}: $' + str('%3.3f'%(r_value1)), verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, fontsize=10)
    ax[-1].text(0.95, 0.55, '$MSD_{CTRL}: $' + str('%3.3f'%(MSD1)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
    ax[-1].text(0.95, 0.5, '$RMSE_{CTRL}: $' + str('%3.3f'%(RMSE1)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
    ax[-1].text(0.95, 0.45, '$Nash_{CTRL}: $' + str('%3.3f'%(nash1)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
    #ax[-1].text(0.03, 0.7, '$R_{SMAP_{Concave}}: $' + str('%3.3f'%(r_value2)), verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, fontsize=10)
    #ax[-1].text(0.03, 0.6, '$RMSE_{SMAP_{Concave}}: $' + str('%3.3f'%(RMSE2)), verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, fontsize=10)	

    #ax[-1].text(0.03, 0.75, '$R_{SMAP_{Monthly-Convex}}: $' + str('%3.3f'%(r_value2)), verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, fontsize=10)
    ax[-1].text(0.95, 0.35, '$MSD_{SMAP-Daily}: $' + str('%3.3f'%(MSD2)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
    ax[-1].text(0.95, 0.3, '$RMSE_{SMAP-Daily}: $' + str('%3.3f'%(RMSE2)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
    ax[-1].text(0.95, 0.25, '$Nash_{SMAP-Daily}: $' + str('%3.3f'%(nash2)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)

    #ax.text(0.34, 0.94, '$R_{SMAP_{Daily-Convex}}: $' + str('%3.3f'%(r_value3)), verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, fontsize=10)
    ax[-1].text(0.95, 0.15, '$MSD_{SMAP-KF-Daily}: $' + str('%3.3f'%(MSD3)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
    ax[-1].text(0.95, 0.1, '$RMSE_{SMAP-KF-Daily}: $' + str('%3.3f'%(RMSE3)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
    ax[-1].text(0.95, 0.05, '$Nash_{SMAP-KF-Daily}: $' + str('%3.3f'%(nash3)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)

    if counter == 0:
        ax[-1].set_ylabel('CLM Simulations ($km^3/y$)', fontsize=10, color='k')
        ax[-1].set_title('State: ' + STstring[counter], fontsize=10)
    elif counter == 1:
        ax[-1].set_xlabel('USGS Irrigation ($km^3/y$)', fontsize=10, color='k')
        ax[-1].set_title('Scatter Plots: County Scale Irrigation Water Amount for Years ' + YR + '\nState: ' + STstring[counter], fontsize=10)
    else:
        ax[-1].set_title('State: ' + STstring[counter], fontsize=10)

    plt.ylim(0.000001,max(max(df.X_021_Irr.astype(float)),max(df.X_024_Irr.astype(float)),max(df.X_024_kf_Irr.astype(float)),max(df.USGS_Irr.astype(float))) + 2)
    plt.xlim(0.000001,max(max(df.X_021_Irr.astype(float)),max(df.X_024_Irr.astype(float)),max(df.X_024_kf_Irr.astype(float)),max(df.USGS_Irr.astype(float))) + 2)

    ax[-1].plot([0.000001,max(max(df.X_021_Irr.astype(float)),max(df.X_024_Irr.astype(float)),max(df.X_024_kf_Irr.astype(float)),max(df.USGS_Irr.astype(float))) + 2], \
         [0.000001,max(max(df.X_021_Irr.astype(float)),max(df.X_024_Irr.astype(float)),max(df.X_024_kf_Irr.astype(float)),max(df.USGS_Irr.astype(float))) + 2], \
         linewidth = 0.5, color='k', linestyle = '-.')

    
plt.tight_layout(pad=0, w_pad=0, h_pad=0)

savefig( figDIR + 'Fig3_irrigation_amount_spatialMap_countyScatterPlots_grndOBS_BIAScor_spatiotemporalKF_minIRRIG_Finalallkm3_logscale_v2_' + YR + '.png', bbox_inches='tight', dpi=600 )

# savefig( figDIR + 'irrigation_amount_spatialMap_countyScatterPlots_v2' + '.png', bbox_inches='tight', dpi=600 )
#savefig( figDIR + 'irrigation_amount_spatialMap_countyScatterPlots_v2' + '.pdf', bbox_inches='tight')
plt.close()

