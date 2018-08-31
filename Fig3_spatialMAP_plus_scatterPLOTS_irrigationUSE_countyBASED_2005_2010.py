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
figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'


title = ['USGS ',\
         'CLM Control ',\
         'CLM SMAP_Raw ',   
         'CLM SMAP_KF ']          


gs1 = gridspec.GridSpec(2, 4, width_ratios=[1,1,1,1],height_ratios=[1,1]) # the middle row is to place the colorbar axes
gs1.update(bottom=0.59, top=0.98, wspace=0.02)

gs2 = gridspec.GridSpec(2, 3, width_ratios=[1,1,1],height_ratios=[1,1]) # the middle row is to place the colorbar axes
gs2.update(bottom=0.04, top=0.5, wspace=0.15)

fig = plt.figure(num=1, figsize=(13,10))
ax = []

gs1counter = 0
for YR in ['2005','2010']:
    File_Name = ['USGSirr_Gridded1to8thdeg_withdrawals_freshwater_Mgal_d_year' + YR + '.npy',\
                 'CLM_021_cntrlSIM_countyAggregated_irrigationWater_KM3_Year_' + YR + '.npy',\
                 'CLM_024_dailySMAPSIM_countyAggregated_irrigationWater_KM3_Year_' + YR + '.npy',
                 'CLM_024_KF_Final_spatial_temporal_neighbouring_smapERROR_09_P0_2_minCTRLIRRIGandSMAPIRRIG_countyAggregated_irrigationWater_KM3_Year_' + YR + '.npy']
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
        ax.append(fig.add_subplot(gs1[gs1counter]))
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
        plt.title(title[counter] + YR,fontsize=10) 
        gs1counter = gs1counter + 1

##ax.append(fig.add_subplot(gs[1,1]))
##fig.colorbar(cs, cax=ax[-1], cmap=cmap, norm=norm,spacing='uniform', ticks=bounds[0::2], boundaries=bounds, orientation='horizontal', extend='max')
cbar_ax = fig.add_axes([0.2, 0.56, 0.6, 0.02])    
cb = fig.colorbar(cs, cax=cbar_ax, cmap=cmap, norm=norm,spacing='uniform', ticks=bounds_b[0::1], boundaries=bounds_b, orientation='horizontal', extend='max')
cb.set_label('Irrigation Water\n' + '    $km^3/y$',fontsize=10,labelpad=-29, x=1.1)
cb.ax.set_xticklabels(bounds) # I changed the ticks labels to string to have variable number of digits to the right of decimal point    
#%%======================================================================= Plotting the scatters
srcDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/'

STstring = ['NE', 'KS', 'TX']
gs2counter = 0

for YR in ['2005','2010']:
    dfmain = pd.read_csv(srcDIR + 'County_Irrigations_USGC_CLM_IrrAmount_km3_year' + YR + '_X21_X24dialy_convex_vs_X24KF_Final_spatial_temporal_neighbouring_smapERROR_09_P0_2_minCTRLIRRIGandSMAPIRRIG.txt',sep=',', header=None, error_bad_lines=False)
    dfmain.columns=['State_County','USGS_Irr','X_021_Irr','X_024_Irr','X_024_kf_Irr']
    for counter in range(3):
        ax.append(fig.add_subplot(gs2[gs2counter]))
    
        df = dfmain[dfmain.State_County.str.startswith(STstring[counter])]
        ax[-1].plot(df.USGS_Irr.astype(float),df.X_021_Irr.astype(float) , ls="", markersize= 2 , markeredgecolor='tomato',marker='o', markerfacecolor='none',label='CLM CONTORL')
        ax[-1].plot(df.USGS_Irr.astype(float),df.X_024_Irr.astype(float) , ls="", markersize = 2 , markeredgecolor='dodgerblue',marker='<', markerfacecolor='none',label='CLM SMAP_Raw')
        ax[-1].plot(df.USGS_Irr.astype(float),df.X_024_kf_Irr.astype(float) , ls="", markersize = 2 , markeredgecolor='forestgreen',marker='s', markerfacecolor='none',label='CLM SMAP_KF')
    
        ax[-1].set_yscale('log')
        ax[-1].set_xscale('log')
        
        if gs2counter == 0:
            legend_properties = {'size': 10}
            lgnd = plt.legend(loc='upper left',prop=legend_properties)
            lgnd.legendHandles[0]._legmarker.set_markersize(6)  # size of the scatter in the legend
            lgnd.legendHandles[1]._legmarker.set_markersize(6)
            lgnd.legendHandles[2]._legmarker.set_markersize(6)    
    
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
    
        ax[-1].text(0.95, 0.55, '$MSD_{CTRL}: $' + str('%3.3f'%(MSD1)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
        ax[-1].text(0.95, 0.5, '$RMSE_{CTRL}: $' + str('%3.3f'%(RMSE1)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
        ax[-1].text(0.95, 0.45, '$Nash_{CTRL}: $' + str('%3.3f'%(nash1)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
    
        ax[-1].text(0.95, 0.35, '$MSD_{SMAP-Raw}: $' + str('%3.3f'%(MSD2)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
        ax[-1].text(0.95, 0.3, '$RMSE_{SMAP-Raw}: $' + str('%3.3f'%(RMSE2)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
        ax[-1].text(0.95, 0.25, '$Nash_{SMAP-Raw}: $' + str('%3.3f'%(nash2)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
    
        ax[-1].text(0.95, 0.15, '$MSD_{SMAP-KF}: $' + str('%3.3f'%(MSD3)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
        ax[-1].text(0.95, 0.1, '$RMSE_{SMAP-KF}: $' + str('%3.3f'%(RMSE3)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
        ax[-1].text(0.95, 0.05, '$Nash_{SMAP-KF}: $' + str('%3.3f'%(nash3)), verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes, fontsize=8)
 
  
        if gs2counter != 1:
            ax[-1].set_title('State: ' + STstring[counter] + ' ' + YR, fontsize=10)
        elif gs2counter == 1:
            ax[-1].set_title('Scatter Plots: County Scale Irrigation Water Amount ' + '\nState: ' + STstring[counter] + ' ' + YR, fontsize=10)

        if gs2counter == 4:
            ax[-1].set_xlabel('USGS Irrigation ($km^3/y$)', fontsize=10, color='k')
        elif gs2counter == 0:
            ax[-1].set_ylabel('CLM Simulations ($km^3/y$)', fontsize=10, color='k')  
            ax[-1].yaxis.set_label_coords(-0.16, -0.18)            
            
        plt.ylim(0.000001,max(max(df.X_021_Irr.astype(float)),max(df.X_024_Irr.astype(float)),max(df.X_024_kf_Irr.astype(float)),max(df.USGS_Irr.astype(float))) + 2)
        plt.xlim(0.000001,max(max(df.X_021_Irr.astype(float)),max(df.X_024_Irr.astype(float)),max(df.X_024_kf_Irr.astype(float)),max(df.USGS_Irr.astype(float))) + 2)
    
        ax[-1].plot([0.000001,max(max(df.X_021_Irr.astype(float)),max(df.X_024_Irr.astype(float)),max(df.X_024_kf_Irr.astype(float)),max(df.USGS_Irr.astype(float))) + 2], \
             [0.000001,max(max(df.X_021_Irr.astype(float)),max(df.X_024_Irr.astype(float)),max(df.X_024_kf_Irr.astype(float)),max(df.USGS_Irr.astype(float))) + 2], \
             linewidth = 0.5, color='k', linestyle = '-.')
        gs2counter = gs2counter + 1
    
plt.tight_layout(pad=0, w_pad=0, h_pad=0)

savefig( figDIR + 'Fig3_irrigation_amount_spatialMap_countyScatterPlots_spatiotemporalKF_minIRRIG_Finalallkm3_logscale_2005_2010' + '.png', bbox_inches='tight', dpi=600 )
#savefig( figDIR + 'Fig3_irrigation_amount_spatialMap_countyScatterPlots_spatiotemporalKF_minIRRIG_Finalallkm3_logscale_2005_2010' + '.pdf', bbox_inches='tight' )

plt.close()

