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
eYEAR = 2016
sMONTH = 6
eMONTH = 8
#=================================== read SMAP data
smap_sm_gridded = load(SMAP_DIR + 'SMAPregridded_10min_ave_for_JJA_' + str(sYEAR) + '_' + str(eYEAR) + '.npy')
smap_sm_gridded = np.roll(np.ma.masked_equal(smap_sm_gridded, -9999.0),1080,axis=1)
smap_sm_gridded_centralUS = smap_sm_gridded[cyl10minxy(50,-116)[1]:cyl10minxy(30,-116)[1],\
                                            cyl10minxy(50,-116)[0]:cyl10minxy(50,-90)[0]]

smap3min = np.ma.masked_equal(load(SMAP_DIR + 'SMAPregridded_3minCentralUS_ave_for_JJA_2015_2016.npy'), -9999.0)
#=================================== read CLM data

for yy in range(sYEAR, eYEAR + 1):
    for mm in range(sMONTH, eMONTH + 1):
        ctrl_IRRG_FILE = netcdf.netcdf_file(ctrl_IRRG_DIR + 'X_021_centralUSA_ICRUCLM45BGCCROPNLDAS.clm2.h0.' + str(yy).zfill(4) + '-' + str('%02d'%(mm) ) + '.nc','r')
        ctrl_h2osoi_dat0 = ctrl_IRRG_FILE.variables['H2OSOI'][0,]
        ctrl_h2osoi_dat01 = ctrl_h2osoi_dat0[0,].reshape(1,400,520)
        ctrl_h2osoi_dat02 = ctrl_h2osoi_dat0[1,].reshape(1,400,520)
        ctrl_h2osoi_dat_dummy = ctrl_h2osoi_dat01*(0.014201/0.041649) + ctrl_h2osoi_dat02*(0.027447/0.041649)

        if yy == sYEAR and mm == sMONTH:
            ctrl_h2osoi_dat = ctrl_h2osoi_dat_dummy
        else:
            ctrl_h2osoi_dat = np.ma.concatenate([ctrl_h2osoi_dat, ctrl_h2osoi_dat_dummy], axis = 0)            
ctrl_h2osoi_dat = ctrl_h2osoi_dat.mean(0)
ctrl_h2osoi_dat= flipud(ctrl_h2osoi_dat)


for yy in range(sYEAR, eYEAR + 1):
    for mm in range(sMONTH, eMONTH + 1):
        clm_noIR_FILE = netcdf.netcdf_file(clm_noCRnoIR + 'X_023_centralUSA_ICRUCLM45BGCNLDAS_noCrop_noIrr.clm2.h0.' + str(yy).zfill(4) + '-' + str('%02d'%(mm) ) + '.nc','r')
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



for yy in range(sYEAR, eYEAR + 1):
    for mm in range(sMONTH, eMONTH + 1):
        smap_IRRG_FILE = netcdf.netcdf_file(SMAP_IRRG_DIR + 'X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF.clm2.h0.' + str(yy).zfill(4) + '-' + str('%02d'%(mm) ) + '.nc','r')
        smap_h2osoi_dat0 = smap_IRRG_FILE.variables['H2OSOI'][0,]
        smap_h2osoi_dat01 = smap_h2osoi_dat0[0,].reshape(1,400,520)
        smap_h2osoi_dat02 = smap_h2osoi_dat0[1,].reshape(1,400,520)
        smap_h2osoi_dat_dummy = smap_h2osoi_dat01*(0.014201/0.041649) + smap_h2osoi_dat02*(0.027447/0.041649)

        if yy == sYEAR and mm == sMONTH:
            smap_h2osoi_dat = smap_h2osoi_dat_dummy
        else:
            smap_h2osoi_dat = np.ma.concatenate([smap_h2osoi_dat, smap_h2osoi_dat_dummy], axis = 0)            
smap_h2osoi_dat = smap_h2osoi_dat.mean(0)
smap_h2osoi_dat = flipud(smap_h2osoi_dat)

#%%=================================== Area scaling  
clm_noIR_zero0 = zeros ( ( 1, 400, 520 ) )		
ctrl_h2osoi_zero0 = zeros ( ( 1, 400, 520 ) )		
smap_h2osoi_zero0 = zeros ( ( 1, 400, 520 ) )	
	
for i in range(400):
    for j in range(520):
	# if AQdat_hlf[i,j] == 1:
        if ctrl_h2osoi_dat[i,j] >= 0 and ctrl_h2osoi_dat[i,j] <= 1:
            clm_noIR_zero0[0,i,j] = clm_noIR_dat[i,j]
            ctrl_h2osoi_zero0[0,i,j] = ctrl_h2osoi_dat[i,j]
            smap_h2osoi_zero0[0,i,j] = smap_h2osoi_dat[i,j]

fig = plt.figure(figsize=(12, 11))  
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
gs.update(wspace=0.02,hspace = 0.2) 

cmap = plt.cm.jet_r
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[:-70]
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bounds_b = array([0,0.02, 0.04,0.06,0.08,0.1,0.13,0.16,0.19,0.22,0.26,0.3])
bounds = ['0.0','0.02', '0.04','0.06','0.08','0.1','0.13','0.16','0.19','0.22','0.26','0.3']
# I changed the ticks labels to string to have variable number of digits to the right of decimal point 
norm = mpl.colors.BoundaryNorm(bounds_b, cmap.N)

ax0 = plt.subplot(gs[0])
nxa=-116
nxb=-90
nya=50
nyb=30
res=1
LT1 = 0.6
LT2 = 0.6
map0=Basemap( projection ='cyl',  \
			llcrnrlon  = nxa,  \
			 urcrnrlon  = nxb,  \
			 llcrnrlat  = nyb,  \
			 urcrnrlat  =  nya,  \
			 resolution = "c")
map0.drawcoastlines(linewidth=LT1, color='black')
map0.drawcountries(linewidth=LT2, color='grey')
map0.drawstates(linewidth=0.25, color='grey')
cs0 = map0.imshow(ma.masked_equal(smap_sm_gridded_centralUS*IRRmask10min,0.0),origin='upper',interpolation='nearest',cmap=cmap,norm=norm, vmin=0.0, vmax=0.3)
#cs0 = map0.imshow(ma.masked_equal(smap3min*IRRmask3min,0.0),origin='upper',interpolation='nearest',cmap=cmap,norm=norm, vmin=0.0, vmax=0.3)


map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=1.0)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_07_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_08_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_10_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_11_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_12_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_13_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_14_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_15_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_16_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_17_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
# cbar0 = map1.colorbar(cs0,location='bottom', extend='max',pad="5%")
# cbar0.ax.tick_params(labelsize=5)
ax0.text(0.3, 0.85, '(a)', verticalalignment='bottom', horizontalalignment='right', transform=ax0.transAxes,fontsize=20)
plt.title('SMAP_Satellite',fontsize=15)
#================================================================================
ax1 = plt.subplot(gs[1])
nxa=-116
nxb=-90
nya=50
nyb=30
res=1
LT1 = 0.6
LT2 = 0.6
map1=Basemap( projection ='cyl',  \
			llcrnrlon  = nxa,  \
			 urcrnrlon  = nxb,  \
			 llcrnrlat  = nyb,  \
			 urcrnrlat  =  nya,  \
			 resolution = "c")
map1.drawcoastlines(linewidth=LT1, color='black')
map1.drawcountries(linewidth=LT2, color='grey')
map1.drawstates(linewidth=0.25, color='grey')
cs1 = map1.imshow(ma.masked_equal(ctrl_h2osoi_zero0[0,]*IRRmask3min,0.0),origin='upper',interpolation='nearest',cmap=cmap,norm=norm, vmin=0.0, vmax=0.3)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=1)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_07_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_08_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_10_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_11_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_12_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_13_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_14_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_15_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_16_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_17_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
# cbar1 = map.colorbar(cs1,location='bottom', extend='max',pad="5%")
# cbar1.ax.tick_params(labelsize=5)
ax1.text(0.3, 0.85, '(b)', verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes,fontsize=20)
plt.title('CTRL',fontsize=15)

#plt.title('Top 5cm SM (JJA Mean of ' + str(sYEAR) + '_' + str(eYEAR) + ')\n' + 'CTRL',fontsize=15)
cbar_ax = fig.add_axes([0.2, 0.51, 0.6, 0.01])    
cb = fig.colorbar(cs0, cax=cbar_ax, cmap=cmap, norm=norm,spacing='uniform', ticks=bounds_b,orientation='horizontal', extend='max')
cb.set_label('$mm^3/mm^3$',fontsize=10,labelpad=-28, x=1.08)
#cb.set_label('$mm^3/mm^3$',fontsize=15)
cb.ax.tick_params(labelsize=10)
#================================================================================
ax2 = plt.subplot(gs[3])
cmap = plt.cm.nipy_spectral_r
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[10:-60]
for ii in range(len(cmaplist)-20,len(cmaplist)):
    cmaplist[ii] = (0.1,0.1,0.1,0.05)

cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bounds_b = array([-50,-40,-30,-25,-20,-15,-10,-5,-1,0,20])
#bounds = ['-40','-30','-25','-20','-15','-10','-5','-1','0','0.1']
# I changed the ticks labels to string to have variable number of digits to the right of decimal point 
norm = mpl.colors.BoundaryNorm(bounds_b, cmap.N)
nxa=-116
nxb=-90
nya=50
nyb=30
res=1
LT1 = 0.6
LT2 = 0.6
map2=Basemap( projection ='cyl',  \
			llcrnrlon  = nxa,  \
			 urcrnrlon  = nxb,  \
			 llcrnrlat  = nyb,  \
			 urcrnrlat  =  nya,  \
			 resolution = "c")
map2.drawcoastlines(linewidth=LT1, color='black')
map2.drawcountries(linewidth=LT2, color='grey')
map2.drawstates(linewidth=0.25, color='grey')
diff = smap_h2osoi_zero0[0,] - ctrl_h2osoi_zero0[0,]
fraction = (diff*IRRmask3min)/ma.masked_equal(ctrl_h2osoi_zero0*IRRmask3min,0.0)

cs2 = map2.imshow(ma.masked_equal(fraction[0]*100,0.0),origin='upper',interpolation='nearest',cmap=cmap,norm=norm)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=1)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_07_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_08_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_10_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_11_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_12_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_13_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_14_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_15_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_16_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_17_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
#plt.title(r'$\frac{SMAP_{kf} - CTRL}{CTRL}$' ,fontsize=15)
plt.title('% Change in SMAP_kf SM from CTRL' ,fontsize=15)
cbar_ax2 = fig.add_axes([0.53, 0.1, 0.35, 0.01])  
cb2 = fig.colorbar(cs2, cax=cbar_ax2, cmap=cmap, norm=norm, extend='both',spacing='uniform', ticks=bounds_b, boundaries=bounds_b,orientation='horizontal')
#cb2.set_label('$mm^3/mm^3$',fontsize=10,labelpad=-28, x=1.05)
cb2.ax.tick_params(labelsize=10)
cb2.set_label('% Change',fontsize=10)
#cbar2 = map2.colorbar(cs2,location='bottom', extend='max',pad="5%")
#cbar2.ax.tick_params(labelsize=5)
ax2.text(0.3, 0.85, '(d)', verticalalignment='bottom', horizontalalignment='right', transform=ax2.transAxes,fontsize=20)
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
#================================================================================
ax3 = plt.subplot(gs[2])
cmap = plt.cm.nipy_spectral_r
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[10:-60]
for ii in range(len(cmaplist)-20,len(cmaplist)):
    cmaplist[ii] = (0.1,0.1,0.1,0.05)

cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bounds_b = array([-0.2,-0.15,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0.0,0.1])
#bounds = ['-40','-30','-25','-20','-15','-10','-5','-1','0','0.1']
# I changed the ticks labels to string to have variable number of digits to the right of decimal point 
norm = mpl.colors.BoundaryNorm(bounds_b, cmap.N)
nxa=-116
nxb=-90
nya=50
nyb=30
res=1
LT1 = 0.6
LT2 = 0.6
map3=Basemap( projection ='cyl',  \
			llcrnrlon  = nxa,  \
			 urcrnrlon  = nxb,  \
			 llcrnrlat  = nyb,  \
			 urcrnrlat  =  nya,  \
			 resolution = "c")
map3.drawcoastlines(linewidth=LT1, color='black')
map3.drawcountries(linewidth=LT2, color='grey')
map3.drawstates(linewidth=0.25, color='grey')

#diff3 = clm_noIR_zero0[0,] - ctrl_h2osoi_zero0[0,]

diff3 = smap3min - clm_noIR_zero0[0,]

cs3 = map3.imshow(ma.masked_equal(diff3*IRRmask3min,0.0),origin='upper',interpolation='nearest',cmap=cmap,norm=norm,vmax = 0.0)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=1)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_07_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_08_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_10_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_11_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_12_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_13_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_14_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_15_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_16_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
map3.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_17_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5)
# cbar1 = map.colorbar(cs1,location='bottom', extend='max',pad="5%")
# cbar1.ax.tick_params(labelsize=5)
ax3.text(0.3, 0.85, '(c)', verticalalignment='bottom', horizontalalignment='right', transform=ax3.transAxes,fontsize=20)
#plt.title('NOirrig - CTRL',fontsize=15)
plt.title('SMAP_Satellite - NOirrig',fontsize=15)

cbar_ax3 = fig.add_axes([0.15, 0.1, 0.35, 0.01])  
cb3 = fig.colorbar(cs3, cax=cbar_ax3, cmap=cmap, norm=norm, extend='both',spacing='uniform', ticks=bounds_b, boundaries=bounds_b,orientation='horizontal')
#cb.set_label('$mm^3/mm^3$',fontsize=15,labelpad=-28, x=1.08)
cb3.set_label('$mm^3/mm^3$',fontsize=10)
cb3.ax.tick_params(labelsize=10)


savefig(figDIR + 'Fig2_SMAP_CLMctrl_CLMSMAPKF_centralUSA_5cmSM_ave_' + str(sYEAR) + '_' + str(eYEAR) + '_v3e.png', dpi=600, bbox_inches='tight')
savefig(figDIR + 'Fig2_SMAP_CLMctrl_CLMSMAPKF_centralUSA_5cmSM_ave_' + str(sYEAR) + '_' + str(eYEAR) + '_v3e.pdf', dpi=600, bbox_inches='tight')
#
plt.close()


