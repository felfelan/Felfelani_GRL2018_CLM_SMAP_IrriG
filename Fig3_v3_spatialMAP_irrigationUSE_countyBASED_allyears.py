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
DIR    = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src_v2/'
figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'
YEARS  = ['2005','2010']
#YEARS  = ['1985','1990','1995','2000']
title  = ['CTRL - USGS',
          'SMAP_raw - USGS',
          'SMAP_KF - USGS',
          'SMAP_KF_BC - USGS']

gs1 = gridspec.GridSpec(len(YEARS), 4) # the middle row is to place the colorbar axes
gs1.update(bottom=0.12, top=0.98, wspace=0.03, hspace = 0.1)
#letter = ['i','j','k','l','m','n','o','p',\
#		  'q','r','s','t','u','v','w','x']
letter = ['a','b','c','d','e','f','g','h']

fig = plt.figure(num=1, figsize=(8,1.75 * len(YEARS))) # w, h
ax = []
rc('axes', linewidth= 0.2)
gs1counter = 0
for YR in YEARS:
    File_Name = ['CLM_021_' + YR + '_dlyCTRLsim_countyAggregated_irrigationWater_KM3.npy',\
                 'CLM_024_' + YR + '_dlySMAPsim_countyAggregated_irrigationWater_KM3_no_BiasCorrection.npy',\
                 'CLM_024_' + YR + '_dSMAPKFsim_countyAggregated_irrigationWater_KM3_no_BiasCorrection.npy',\
                 'CLM_024_' + YR + '_dSMAPKFsim_countyAggregated_irrigationWater_KM3_grndOBS_BiasCorrection_minIRR.npy']

    USGS = load(DIR + 'USGSirr_Gridded3mindegCentralUS_withdrawals_freshwater_KM3_year_' + YR + '.npy')

    for counter in range(4):
    

        # unit conversion from Mgal/d to KM^3/y
        IRRG_FILE = load(DIR + File_Name[counter])    

        cmap = plt.cm.RdBu
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
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
		# for pdf
        # ax[-1].map.drawstates(linewidth=0.25, color='lightgrey')
        # for png
        ax[-1].map.drawstates(linewidth=0.25, color='grey')
		
        subtract = IRRG_FILE[:,:cntralUSA3minxy(30,-92)[0]] - USGS[:,:cntralUSA3minxy(30,-92)[0]]

        cs = ax[-1].map.imshow(subtract , origin='upper',interpolation='none',cmap=cmap,norm=norm)
		#for png
        ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=0.3)
		#for pdf
        # ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=0.7, color='grey')
		
        ax[-1].map.plot([-115.5,-110],[45,45], color = 'red',linewidth=0.5)
        ax[-1].map.plot([-115.5,-115.5],[41,45], color = 'red', linewidth=0.5)
        ax[-1].map.plot([-115.5,-110],[41,41], color = 'red', linewidth=0.5)
        ax[-1].map.plot([-110,-110],[41,45], color = 'red', linewidth=0.5)

        ax[-1].map.plot([-115.7,-111],[34,34], color = 'forestgreen',linewidth=0.5)
        ax[-1].map.plot([-115.7,-115.7],[31,34], color = 'forestgreen', linewidth=0.5)
        ax[-1].map.plot([-115.7,-111],[31,31], color = 'forestgreen', linewidth=0.5)
        ax[-1].map.plot([-111,-111],[31,34], color = 'forestgreen', linewidth=0.5)

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

savefig( figDIR + 'FigS3_irrigation_amount_spatialMap_spatiotemporalKF_minIRRIG_Finalallkm_OLDyears_v4' + '.png', bbox_inches='tight', dpi=600 )
# savefig( figDIR + 'FigS3_irrigation_amount_spatialMap_spatiotemporalKF_minIRRIG_Finalallkm_OLDyears_v4' + '.pdf', bbox_inches='tight' )

plt.close()

