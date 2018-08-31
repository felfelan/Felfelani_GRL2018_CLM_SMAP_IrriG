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



title = ['CTRL - USGS',
         'SMAP_raw - USGS',
         # 'CLM SMAP_raw BC - USGS',
         'SMAP_kf - USGS',
         'SMAP_kf BC - USGS']          


gs1 = gridspec.GridSpec(2, 4, width_ratios=[1,1,1,1],height_ratios=[1,1]) # the middle row is to place the colorbar axes
gs1.update(bottom=0.12, top=0.98, wspace=0.03, hspace = 0.1)
letter = ['a','b','c','d','e','f','g','h']

fig = plt.figure(num=1, figsize=(7,3.0))
ax = []
rc('axes', linewidth= 0.2)
gs1counter = 0
for YR in ['2005','2010']:
    File_Name = ['CLM_021_cntrlSIM_countyAggregated_irrigationWater_KM3_Year_' + YR + '.npy',\
                 'CLM_024_dailySMAPSIM_countyAggregated_irrigationWater_KM3_Year_' + YR + '.npy',\
				 # 'CLM_024_grndOBS_BIAScor_dailySMAPSIM_countyAggregated_irrigationWater_KM3_v5_Year_' + YR + '.npy',\
                 'CLM_024_KF_Final_spatial_temporal_neighbouring_smapERROR_09_P0_2_minCTRLIRRIGandSMAPIRRIG_countyAggregated_irrigationWater_KM3_Year_' + YR + '.npy',\
				 'CLM_024_KF_grndOBS_BIAScor_spatial_temporal_neighbouring_smapERROR_09_P0_2_minCTRLIRRIGandSMAPIRRIG_countyIrrWat_KM3_Year_' + YR + '.npy']
    USGS = load(DIR + 'USGSirr_Gridded3mindeg_withdrawals_freshwater_Mgal_d_year' + YR + '.npy') * 1e6 * 0.00378541 * 365 * 1e-9
    for counter in range(4):
    

        # unit conversion from Mgal/d to KM^3/y
        IRRG_FILE = load(DIR + File_Name[counter])    

        cmap = plt.cm.RdBu
        cmaplist = [cmap(i) for i in range(cmap.N)]
#        cmaplist = cmaplist[60:-60]
        # cmaplist[0] = (0.1,0.1,0.1,0.1)
#        cmaplist[0] = (0.05,0.05,0.05,0.05)
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        #bounds = np.linspace(0,3,16)
        bounds_b = array([-3,-2.5,-2.1,-1.8,-1.5,-1.2,-0.9,-0.6,-0.3,-0.1,0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.5,3])
        bounds_tick = array([-3,-2.5,-1.8,-1.2,-0.6,-0.1,0.1,0.6,1.2,1.8,2.5,3])
        
        bounds = ['-3.0','-2.5','-2.1','-1.8','-1.5','-1.2','-0.9','-0.6','-0.3','-0.1','0.1','0.3','0.6','0.9','1.2','1.5','1.8','2.1','2.5','3.0']
        ticks = ['- 3.0','- 2.5','- 1.8','- 1.2','- 0.6','- 0.1','0.1','0.6','1.2','1.8','2.5','3.0']
       # I changed the ticks labels to string to have variable number of digits to the right of decimal point 
        norm = mpl.colors.BoundaryNorm(bounds_b, cmap.N)
    
        nxa=-116
        nxb=-92
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
        
        subtract = IRRG_FILE[:,:cntralUSA3minxy(30,-92)[0]] - USGS[:,:cntralUSA3minxy(30,-92)[0]]

        cs = ax[-1].map.imshow(subtract , origin='upper',interpolation='quadric',cmap=cmap,norm=norm)
        ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=0.3)
        # ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_07_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.2)
        # ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_08_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.2)
        # ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_10_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.2)
        # ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_11_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.2)
        # ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_12_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.2)
        # ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_13_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.2)
        # ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_14_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.2)
        # ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_15_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.2)
        # ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_16_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.2)
        # ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_17_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.2)
        ax[-1].map.plot([-115.5,-110],[45,45], color = 'forestgreen',linewidth=0.5)
        ax[-1].map.plot([-115.5,-115.5],[41,45], color = 'forestgreen', linewidth=0.5)
        ax[-1].map.plot([-115.5,-110],[41,41], color = 'forestgreen', linewidth=0.5)
        ax[-1].map.plot([-110,-110],[41,45], color = 'forestgreen', linewidth=0.5)
        plt.title(title[counter] + ' (' + YR + ')',fontsize=7)
        ax[-1].text(0.3, 0.8, '(' + letter[gs1counter] + ')', verticalalignment='bottom', horizontalalignment='right', transform=ax[-1].transAxes,fontsize=9)
        rc('axes', linewidth= 0.2) # reduce the thickness of axes frame	
        gs1counter = gs1counter + 1

cbar_ax = fig.add_axes([0.165, 0.1, 0.7, 0.02])    
cb = plt.colorbar(cs, cax=cbar_ax, cmap=cmap, norm=norm,spacing='uniform', ticks=bounds_tick, boundaries=bounds_b, orientation='horizontal',extend = 'both')
cb.set_label('Irrigation Water ($km^3/y$)',fontsize=7)
cb.ax.set_xticklabels(ticks, fontsize=7) # I changed the ticks labels to string to have variable number of digits to the right of decimal point    
#%%======================================================================= Plotting the scatters
plt.clim(-3, 3);
    
# plt.tight_layout(pad=0, w_pad=0, h_pad=0)

savefig( figDIR + 'Fig3_irrigation_amount_spatialMap_spatiotemporalKF_minIRRIG_Finalallkm_2005_2010_v3' + '.png', bbox_inches='tight', dpi=600 )
savefig( figDIR + 'Fig3_irrigation_amount_spatialMap_spatiotemporalKF_minIRRIG_Finalallkm_2005_2010_v3' + '.pdf', bbox_inches='tight' )

plt.close()

