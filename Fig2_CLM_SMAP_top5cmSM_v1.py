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
#ctrl_IRRG_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_021_centralUSA_ICRUCLM45BGCCROPNLDAS_duplicate/'
ctrl_IRRG_DIR = '/mnt/scratch/felfelan/CESM_Archiving/X_021_centralUSA_ICRUCLM45BGCCROPNLDAS_duplicate/lnd/hist/'
#SMAP_IRRG_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_022_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPirr_SMextrapolationModified/'
#SMAP_IRRG_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr/'
SMAP_IRRG_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF/'

sample = '/mnt/home/felfelan/CESM_DIRs/InData_WFDEI05d/lnd/clm2/\
rawdata/mksrf_topo.10min.c080912.nc'

SMAP_DIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/SMAP/'


figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'
#%%=======================================================================
YEAR = 2016
MONTH = 6

#==============================================================================
# get the 2-D lat and lon from a standard CLM input data (CRUNCEPT forcing)
#==============================================================================
sa_data = nc.Dataset(sample,'r')
LONG = sa_data.variables['LONGXY'][:]
LATI = sa_data.variables['LATIXY'][:]

#=================================== read SMAP data
# encoding='latin1' is for solving incompatibility of .npy filed created with python 2.7 with
# while reading it with python 3
smap_sm = np.load(SMAP_DIR + 'SMAP_' + str(YEAR) + str('%02d'%(MONTH) ),encoding='latin1')
smap_sm = np.ma.masked_equal(smap_sm,-9999.0)
#==============================================================================
# Extract latitude and longitude for a given row and column form NASA 
# Geolocation files

# Read binary files and reshape to correct size
# The number of rows and columns are in the file name
#==============================================================================
#lats = np.fromfile(SMAP_DIR + 'EASE2_M36km.lats.964x406x1.double', 
#                      dtype=np.float64).reshape((406,964))
#lons = np.fromfile(SMAP_DIR + 'EASE2_M36km.lons.964x406x1.double', 
#                      dtype=np.float64).reshape((406,964))
#lat_val = []
#lon_val = []
#for ii in range(406):
#    print(ii, '====== out of 406')
#    for jj in range(964):
#        lat_val = lat_val + [lats[ii, jj]]
#        if lons[ii, jj] <= 0:
#            lon_val = lon_val + [lons[ii, jj] + 360]   # shift 180 degree across the longitude
#        else:
#            lon_val = lon_val + [lons[ii, jj]]
#points = np.zeros((len(lat_val),2))        
#points[:,0] = np.array(lat_val)
#points[:,1] = np.array(lon_val)  
#==============================================================================
# regridding using scipy.interpolate.griddata
#==============================================================================
## LATI in the original CRUNCEPT file is upside down, so we have to flip it
#grid_latlon = np.zeros((len(smap_sm),np.shape(LATI)[0],np.shape(LATI)[1]))

#for kk in range(len(smap_sm)):
#    grid_latlon[kk,] = griddata(points, smap_sm[kk,].reshape(406*964), (np.flipud(LATI), \
#                                                LONG), method='nearest')
#save(SMAP_DIR + 'SMAPregridded_10min_daily_for_' + str(YEAR) + str(MONTH).zfill(2),grid_latlon)
smap_sm_gridded = load(SMAP_DIR + 'SMAPregridded_10min_daily_for_' + str(YEAR) + str(MONTH).zfill(2) + '.npy')

smap_sm_gridded = np.roll(np.ma.masked_equal(smap_sm_gridded, -9999.0),1080,axis=2)

smap_sm_gridded_centralUS = smap_sm_gridded[:,cyl10minxy(50,-116)[1]:cyl10minxy(30,-116)[1],\
                                              cyl10minxy(50,-116)[0]:cyl10minxy(50,-90)[0]].mean(0)
#=================================== read CLM data

for YR in range(YEAR, YEAR+1):
  print('Year: ', YR)
  for MON in range(MONTH, MONTH+1):
    
    ctrl_IRRG_FILE = netcdf.netcdf_file(ctrl_IRRG_DIR + 'X_021_centralUSA_ICRUCLM45BGCCROPNLDAS.clm2.h0.' + str(YR).zfill(4) + '-' + str('%02d'%(MON) ) + '.nc','r')
    ctrl_h2osoi_dat0 = ctrl_IRRG_FILE.variables['H2OSOI'][0,]	
    ctrl_h2osoi_dat01 = ctrl_h2osoi_dat0[0,].reshape(1,400,520) 
    ctrl_h2osoi_dat02 = ctrl_h2osoi_dat0[1,].reshape(1,400,520)
  
# consider the SM of top 5 cm soil layer as the SM
ctrl_h2osoi_dat = ctrl_h2osoi_dat01*(0.014201/0.041649) + ctrl_h2osoi_dat02*(0.027447/0.041649)

# IRRG_QIRRIG_dat = flipud(ctrl_IRRG_QIRRIG0[0])
ctrl_h2osoi_dat[0] = flipud(ctrl_h2osoi_dat[0])



for YR in range(YEAR, YEAR+1):
  print('Year: ', YR)
  for MON in range(MONTH, MONTH+1):
    
    smap_IRRG_FILE = netcdf.netcdf_file(SMAP_IRRG_DIR + 'X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF.clm2.h0.' + str(YR).zfill(4) + '-' + str('%02d'%(MON) ) + '.nc','r')
    smap_h2osoi_dat0 = smap_IRRG_FILE.variables['H2OSOI'][0,]	
    smap_h2osoi_dat01 = smap_h2osoi_dat0[0,].reshape(1,400,520) 
    smap_h2osoi_dat02 = smap_h2osoi_dat0[1,].reshape(1,400,520)
  
# consider the SM of top 5 cm soil layer as the SM
smap_h2osoi_dat = smap_h2osoi_dat01*(0.014201/0.041649) + smap_h2osoi_dat02*(0.027447/0.041649)

# IRRG_QIRRIG_dat = flipud(smap_IRRG_QIRRIG0[0])
smap_h2osoi_dat[0] = flipud(smap_h2osoi_dat[0])


#%%=================================== Area scaling  

ctrl_h2osoi_zero0 = zeros ( ( 1, 400, 520 ) )		
smap_h2osoi_zero0 = zeros ( ( 1, 400, 520 ) )	
	
for i in range(400):
    for j in range(520):
	# if AQdat_hlf[i,j] == 1:
        if ctrl_h2osoi_dat[0,i,j] >= 0 and ctrl_h2osoi_dat[0,i,j] <= 1:
            ctrl_h2osoi_zero0[:,i,j] = ctrl_h2osoi_dat[:,i,j]
            smap_h2osoi_zero0[:,i,j] = smap_h2osoi_dat[:,i,j]

fig = plt.figure(figsize=(27, 9))  
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
gs.update(wspace=0.02) 

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
cs0 = map0.imshow(ma.masked_equal(smap_sm_gridded_centralUS,0.0),origin='upper',interpolation='nearest',cmap=cm.jet_r, vmin=0.0, vmax=0.3)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=1.2)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_07_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_08_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_10_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_11_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_12_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_13_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_14_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_15_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_16_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map0.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_17_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
# cbar0 = map1.colorbar(cs0,location='bottom', extend='max',pad="5%")
# cbar0.ax.tick_params(labelsize=5)
# ax0.text(0.1, 0.05, '(b)', verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes,fontsize=25)

plt.title('SMAP Top 5cm SM ' + str(YR) + str(MON).zfill(2),fontsize=15)


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
cs1 = map1.imshow(ma.masked_equal(ctrl_h2osoi_zero0[0,],0.0),origin='upper',interpolation='nearest',cmap=cm.jet_r, vmin=0.0, vmax=0.3)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=1.2)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_07_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_08_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_10_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_11_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_12_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_13_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_14_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_15_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_16_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_17_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
# cbar1 = map.colorbar(cs1,location='bottom', extend='max',pad="5%")
# cbar1.ax.tick_params(labelsize=5)
# ax1.text(0.1, 0.05, '(a)', verticalalignment='bottom', horizontalalignment='right', transform=ax0.transAxes,fontsize=25)

plt.title('CLM Control Top 5cm SM ' + str(YR) + str(MON).zfill(2),fontsize=15)

ax2 = plt.subplot(gs[2])

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
cs2 = map2.imshow(ma.masked_equal(smap_h2osoi_zero0[0,],0.0),origin='upper',interpolation='nearest',cmap=cm.jet_r, vmin=0.0, vmax=0.3)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=1.2)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_07_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_08_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_10_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_11_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_12_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_13_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_14_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_15_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_16_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
map2.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_17_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.8)
# cbar2 = map.colorbar(cs2,location='bottom', extend='max',pad="5%")
# cbar2.ax.tick_params(labelsize=5)
# ax2.text(0.1, 0.05, '(a)', verticalalignment='bottom', horizontalalignment='right', transform=ax0.transAxes,fontsize=25)

plt.title('CLM SMAP_KF Top 5cm SM ' + str(YR) + str(MON).zfill(2),fontsize=15)

cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.03])    
cb = fig.colorbar(cs0, cax=cbar_ax,spacing='uniform',orientation='horizontal', extend='max')
cb.set_label('$mm^3/mm^3$',fontsize=10,labelpad=-28, x=1.05)
cb.ax.tick_params(labelsize=10)
plt.tight_layout(pad=0, w_pad=0, h_pad=0)


savefig(figDIR + 'Fig2_CLM_SMAPKF_centralUSA_5cmSM_' + str(YEAR) + '-' + str('%02d'%(MONTH) ) + '.png', dpi=600, bbox_inches='tight')
plt.close()
# savefig(figDIR + 'CLM_centralUSA_huc2_h2osoi_5cm_X021_duplicate_simulation' + str(YR) + '-' + str('%02d'%(MON) ) + '.pdf', dpi=1000, bbox_inches='tight')

