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
import sys
# My module to change the latlon to xy coordination in SMAP data
from latlon2xy import smapxy, cyl5minxy, cntralUSA3minxy, cylNorthAmerica1_8degxy
#%%=======================================================================
#===== Farshid Felfelani
#===== First version: 02/01/2018
#===== Vertical SM Profile from CRN, CLM
#%%======================================================================= Plotting spatial maps
DIR    = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src_v2/'
figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'
YEARS  = ['2010', '2005','2000','1995','1990','1985']
title  = ['USGS Mean',
          'USGS STD']

gs1 = gridspec.GridSpec(1, 2) # the middle row is to place the colorbar axes
gs1.update(bottom=0.12, top=0.98, wspace=0.03, hspace = 0.1)
letter = ['a','b']

xx = 0 
for YR in YEARS:

	USGS = load(DIR + 'USGSirr_Gridded3mindegCentralUS_withdrawals_freshwater_KM3_year_' + YR + '.npy')	
	if xx == 0 :
		USGSconcat = USGS.reshape(1,400,520)
		xx = 1
	else:
		USGSconcat = np.concatenate((USGSconcat,USGS.reshape(1,400,520)),axis = 0)
    
USGSmean = USGSconcat.mean(0).reshape(1,400,520)
USGSstd  = USGSconcat.std(0).reshape(1,400,520)

USGStot  = np.concatenate((USGSmean, USGSstd), axis = 0) 
#sys.exit(0)

#cmap = ['gist_ncar_r','bone_r']
cmap = ['bone_r','bone_r']
fig = plt.figure(num=1, figsize=(7,3))# w, h
ax = []
rc('axes', linewidth= 0.2)

for ii in range(2):
	nxa=-116
	nxb=-92
	nya=50
	nyb=30
	res=1
	LT1 = 0.6
	LT2 = 0.6
	ax.append(fig.add_subplot(gs1[ii]))
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
	
	
	
	cs = ax[-1].map.imshow(np.ma.masked_equal(USGStot[ii,:,:cntralUSA3minxy(30,-92)[0]],0.0), origin='upper',interpolation='none',cmap=cmap[ii], vmin =0.0, vmax = 3.0)
	#for png
	ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=0.3)
	#for pdf
	#        ax[-1].map.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=0.7, color='grey')
	
	ax[-1].map.plot([-115.5,-110],[45,45], color = 'red',linewidth=0.5)
	ax[-1].map.plot([-115.5,-115.5],[41,45], color = 'red', linewidth=0.5)
	ax[-1].map.plot([-115.5,-110],[41,41], color = 'red', linewidth=0.5)
	ax[-1].map.plot([-110,-110],[41,45], color = 'red', linewidth=0.5)
	
	ax[-1].map.plot([-115.7,-111],[34,34], color = 'forestgreen',linewidth=0.5)
	ax[-1].map.plot([-115.7,-115.7],[31,34], color = 'forestgreen', linewidth=0.5)
	ax[-1].map.plot([-115.7,-111],[31,31], color = 'forestgreen', linewidth=0.5)
	ax[-1].map.plot([-111,-111],[31,34], color = 'forestgreen', linewidth=0.5)
	
	plt.title('(' + letter[ii] + ')' + ' ' + title[ii] ,fontsize=10)
#	ax[-1].text(0.1, 0.8, '(' + letter[ii] + ')', verticalalignment='bottom', horizontalalignment='left', transform=ax[-1].transAxes,fontsize=9)
	rc('axes', linewidth= 0.2) # reduce the thickness of axes frame	

#	cbarX = [0.17,0.55]
#	cbar_ax = fig.add_axes([cbarX[ii], 0.12, 0.3, 0.02])    
#	cb = plt.colorbar(cs, cax=cbar_ax, orientation='horizontal',extend = 'max')
#	cb.set_label('$km^3/y$',fontsize=7)


cbar_ax = fig.add_axes([0.27, 0.12, 0.5, 0.02])    
cb = plt.colorbar(cs, cax=cbar_ax, orientation='horizontal',extend = 'max')
cb.set_label('$km^3/y$',fontsize=7)

savefig( figDIR + 'FigS4_USGS_irrigation_amount_spatialMap_meanPLUSstd_v2' + '.png', bbox_inches='tight', dpi=600 )
#savefig( figDIR + 'FigS4_USGS_irrigation_amount_spatialMap_meanPLUSstd_v2' + '.pdf', bbox_inches='tight' )

plt.close()

