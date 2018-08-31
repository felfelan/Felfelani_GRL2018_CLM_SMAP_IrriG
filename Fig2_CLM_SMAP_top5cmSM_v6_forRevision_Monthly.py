#!~/anaconda3/DIR/bin/ipython

from __future__ import division # force division to be floating point in Python
import numpy as np
from pylab import *
from matplotlib import gridspec
from scipy.interpolate import griddata
import time
import os
#from cf .util import *
#from cf import *
#from cf.io import *
#from netCDF4 import Dataset
import netCDF4 as nc
from scipy.io import netcdf
from numpy import savetxt
from mpl_toolkits.basemap import Basemap,shiftgrid
from latlon2xy import smapxy, cyl5minxy, cntralUSA3minxy,cyl10minxy 

#=======================================================================
#===== Farshid Felfelani
#===== First version: 05/10/2017
#===== Reading CLM hlf degree simulations and plot spatial map for irrigation
#%%=======================================================================
clm_noCRnoIR = '/mnt/scratch/felfelan/CESM_simulations/X_023_centralUSA_ICRUCLM45BGCNLDAS_noCrop_noIrr/'
ctrl_IRRG_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_021_centralUSA_ICRUCLM45BGCCROPNLDAS_duplicate/'
SMAP_IRRG_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF/no_BiasCorrection/'

src = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/'

IRRmask3min = np.load(src + 'Portmann2010_Irrigated_areas_mask_3min_centralUS.npy')
IRRmask10min = np.load(src + 'Portmann2010_Irrigated_areas_mask_10min_centralUS.npy')


SMAP_DIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/SMAP/'

figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'
#%%=======================================================================
sYEAR = 2015
eYEAR = 2015
sMONTH = 6
eMONTH = 6
#=================================== read SMAP data
#smap_sm_gridded = load(SMAP_DIR + 'SMAPregridded_10min_ave_for_JJA_' + str(sYEAR) + '_' + str(eYEAR) + '.npy')
#smap_sm_gridded = np.roll(np.ma.masked_equal(smap_sm_gridded, -9999.0),1080,axis=1)
#smap_sm_gridded_centralUS = smap_sm_gridded[cyl10minxy(50,-116)[1]:cyl10minxy(30,-116)[1],\
#                                            cyl10minxy(50,-116)[0]:cyl10minxy(50,-90)[0]]

smap3min = np.ma.masked_equal(load(SMAP_DIR + 'SMAPregridded_3minCentralUS_ave_for_' + str(sYEAR) + '_' + str(eYEAR) + '_' + str(sMONTH).zfill(2) + '_' + str(eMONTH).zfill(2) + '.npy'), -9999.0)
#=================================== read CLM data

for yy in range(sYEAR, eYEAR + 1):
	for mm in range(sMONTH, eMONTH + 1):
		clm_noIR_FILE = netcdf.netcdf_file(clm_noCRnoIR + 'X_023_centralUSA_ICRUCLM45BGCNLDAS_noCrop_noIrr.clm2.h0.' + str(yy).zfill(4) + '-' + str('%02d'%(mm) ) + '.nc','r')
		print('SMAPregridded_3minCentralUS_ave_for_' + str(sYEAR) + '_' + str(eYEAR) + '_' + str(sMONTH).zfill(2) + '_' + str(eMONTH).zfill(2) + '.npy')
		print('X_023_centralUSA_ICRUCLM45BGCNLDAS_noCrop_noIrr.clm2.h0.' + str(yy).zfill(4) + '-' + str('%02d'%(mm) ) + '.nc')
		clm_noIR_dat0 = clm_noIR_FILE.variables['H2OSOI'][0,]
		clm_noIR_dat01 = clm_noIR_dat0[0,].reshape(1,400,520)
		clm_noIR_dat02 = clm_noIR_dat0[1,].reshape(1,400,520)
		clm_noIR_dat_dummy = clm_noIR_dat01*(0.014201/0.041649) + clm_noIR_dat02*(0.027447/0.041649)
		if yy == sYEAR and mm == sMONTH:
			clm_noIR_dat = clm_noIR_dat_dummy
		else:
			clm_noIR_dat = np.ma.concatenate([clm_noIR_dat, clm_noIR_dat_dummy], axis = 0)            
clm_noIR_dat = clm_noIR_dat.mean(0)
clm_noIR_dat= flipud(clm_noIR_dat)

#%%=================================== Area scaling  
clm_noIR_zero0 = zeros ( ( 1, 400, 520 ) )		
	
for i in range(400):
    for j in range(520):
	# if AQdat_hlf[i,j] == 1:
        if clm_noIR_dat[i,j] >= 0 and clm_noIR_dat[i,j] <= 1:
            clm_noIR_zero0[0,i,j] = clm_noIR_dat[i,j]

#%%================================= Plotting
fig = plt.figure(figsize=(7, 6)) #figsize: w,h  
gs = gridspec.GridSpec(1, 1)


#for PDF
#lwHU = 0.4
#clHU = 'grey'
#clST = 'lightgrey'
#lwCOAST = 0.6
#lwCONTRS = 0.6
#lwSTS = 0.25

#for PNG
lwHU = 0.3
clHU = 'black'
clST = 'grey'
lwCOAST = 0.6
lwCONTRS = 0.6
lwSTS = 0.25

ax3 = plt.subplot(gs[0])
#cmap = plt.cm.nipy_spectral_r
#cmaplist = [cmap(i) for i in range(cmap.N)]
#cmaplist = cmaplist[10:-60]
#for ii in range(len(cmaplist)-20,len(cmaplist)):
#    cmaplist[ii] = (0.8,0.8,0.8,0.5)
#
#cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
#bounds_b = array([-0.2,-0.15,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0.0,0.1])
#bounds = ['-40','-30','-25','-20','-15','-10','-5','-1','0','0.1']
# I changed the ticks labels to string to have variable number of digits to the right of decimal point 
#norm = mpl.colors.BoundaryNorm(bounds_b, cmap.N)
nxa=-116
nxb=-92
nya=50
nyb=30
res=1
map3=Basemap( projection ='cyl',  \
			llcrnrlon  = nxa,  \
			 urcrnrlon  = nxb,  \
			 llcrnrlat  = nyb,  \
			 urcrnrlat  =  nya,  \
			 resolution = "c")
map3.drawcoastlines(linewidth=lwCOAST, color='black')
map3.drawcountries(linewidth=lwCONTRS, color='grey')
map3.drawstates(linewidth=lwSTS, color=clST)

#diff3 = clm_noIR_zero0[0,] - ctrl_h2osoi_zero0[0,]

diff3 = smap3min[:,:cntralUSA3minxy(30,-92)[0]] - clm_noIR_zero0[0,:,:cntralUSA3minxy(30,-92)[0]]

cs3 = map3.imshow(ma.masked_equal(diff3*IRRmask3min[:,:cntralUSA3minxy(30,-92)[0]],0.0),origin='upper',interpolation='nearest',cmap='seismic_r', vmin = -0.15, vmax = 0.15)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=1)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_07_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=lwHU, color=clHU)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_08_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=lwHU, color=clHU)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_10_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=lwHU, color=clHU)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_11_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=lwHU, color=clHU)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_12_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=lwHU, color=clHU)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_13_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=lwHU, color=clHU)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_14_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=lwHU, color=clHU)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_15_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=lwHU, color=clHU)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_16_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=lwHU, color=clHU)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_17_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=lwHU, color=clHU)
# cbar1 = map.colorbar(cs1,location='bottom', extend='max',pad="5%")
# cbar1.ax.tick_params(labelsize=5)
ax3.text(0.3, 0.85, '(c)', verticalalignment='bottom', horizontalalignment='right', transform=ax3.transAxes,fontsize=15)
#plt.title('NOirrig - CTRL',fontsize=15)
plt.title('SMAP_Satellite - NOirrig ' + str(sYEAR) + '-' + str(sMONTH).zfill(2),fontsize=15)

cbar_ax3 = fig.add_axes([0.15, 0.1, 0.7, 0.01])  
cb3 = fig.colorbar(cs3, cax=cbar_ax3, extend='both',orientation='horizontal')
#cb.set_label('$mm^3/mm^3$',fontsize=15,labelpad=-28, x=1.08)
cb3.set_label('$mm^3/mm^3$',fontsize=10)
cb3.ax.tick_params(labelsize=10)
plt.setp(ax3.spines.values(), color='darkgray')

savefig(figDIR + 'FigR1_CLMNOirrig_SMAP_centralUSA_5cmSM_ave_' + str(sYEAR) + '_' + str(eMONTH).zfill(2) + '.png', dpi=300, bbox_inches='tight')
#savefig(figDIR + 'Fig2_SMAP_CLMctrl_CLMSMAPKF_centralUSA_5cmSM_ave_' + str(sYEAR) + '_' + str(eYEAR) + '_v3g.pdf', bbox_inches='tight')

plt.close()


