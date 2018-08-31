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
ctrl_IRRG_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_021_centralUSA_ICRUCLM45BGCCROPNLDAS_duplicate/'
SMAP_IRRG_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF/no_BiasCorrection/'
src = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/'

IRRmask3min = np.load(src + 'Portmann2010_Irrigated_areas_mask_3min_centralUS.npy')
IRRmask10min = np.load(src + 'Portmann2010_Irrigated_areas_mask_10min_centralUS.npy')


sample = '/mnt/home/felfelan/CESM_DIRs/InData_WFDEI05d/lnd/clm2/\
rawdata/mksrf_topo.10min.c080912.nc'

SMAP_DIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/SMAP/'

figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'
#%%=======================================================================
sYEAR = 2015
eYEAR = 2016
sMONTH = 6
eMONTH = 8
#==============================================================================
# get the 2-D lat and lon from a standard CLM input data (CRUNCEPT forcing)
#==============================================================================
sa_data = nc.Dataset(sample,'r')
LONG = sa_data.variables['LONGXY'][:]
LATI = sa_data.variables['LATIXY'][:]

#=================================== read SMAP data
# encoding='latin1' is for solving incompatibility of .npy filed created with python 2.7 with
# while reading it with python 3
# for yy in range(sYEAR, eYEAR + 1):
    # for mm in range(sMONTH, eMONTH + 1):
        # smap_sm0 = np.load(SMAP_DIR + 'SMAP_' + str(yy) + str('%02d'%(mm) ),encoding='latin1')
        # smap_sm0 = np.ma.masked_equal(smap_sm0,-9999.0)
        # print(SMAP_DIR + 'SMAP_' + str(yy) + str('%02d'%(mm) ))
        # if yy == sYEAR and mm == sMONTH:
            # smap_sm = smap_sm0
        # else:
            # smap_sm = np.ma.concatenate([smap_sm, smap_sm0], axis = 0)

# smap_sm_mean = np.ma.mean(smap_sm,axis = 0)        
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
#grid_latlon = np.zeros((len(smap_sm_mean),np.shape(LATI)[0],np.shape(LATI)[1]))
#
#grid_latlon = griddata(points, smap_sm_mean.reshape(406*964), (np.flipud(LATI), \
#                                                LONG), method='nearest')
#save(SMAP_DIR + 'SMAPregridded_10min_ave_for_' + str(sYEAR) + '_' + str(eYEAR),grid_latlon.filled(-9999.0))

smap_sm_gridded = load(SMAP_DIR + 'SMAPregridded_10min_ave_for_' + str(sYEAR) + '_' + str(eYEAR) + '.npy')

smap_sm_gridded = np.roll(np.ma.masked_equal(smap_sm_gridded, -9999.0),1080,axis=1)

smap_sm_gridded_centralUS = smap_sm_gridded[cyl10minxy(50,-116)[1]:cyl10minxy(30,-116)[1],\
                                              cyl10minxy(50,-116)[0]:cyl10minxy(50,-90)[0]]
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

ctrl_h2osoi_zero0 = zeros ( ( 1, 400, 520 ) )		
smap_h2osoi_zero0 = zeros ( ( 1, 400, 520 ) )	
	
for i in range(400):
    for j in range(520):
	# if AQdat_hlf[i,j] == 1:
        if ctrl_h2osoi_dat[i,j] >= 0 and ctrl_h2osoi_dat[i,j] <= 1:
            ctrl_h2osoi_zero0[0,i,j] = ctrl_h2osoi_dat[i,j]
            smap_h2osoi_zero0[0,i,j] = smap_h2osoi_dat[i,j]

fig = plt.figure(figsize=(18, 6))  
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
gs.update(wspace=0.02) 

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

plt.title('SMAP Satellite',fontsize=15)


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
cbar_ax = fig.add_axes([0.2, 0.15, 0.4, 0.03])    
cb = fig.colorbar(cs0, cax=cbar_ax, cmap=cmap, norm=norm,spacing='uniform', ticks=bounds_b,orientation='horizontal', extend='max')
#cb.set_label('$mm^3/mm^3$',fontsize=15,labelpad=-28, x=1.08)
cb.set_label('$mm^3/mm^3$',fontsize=15)
cb.ax.tick_params(labelsize=15)

ax2 = plt.subplot(gs[2])


cmap = plt.cm.nipy_spectral_r
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[10:-60]

#cmaplist[0] = (0,0,0,0)


#cmap = plt.cm.cool_r
#cmaplist = [cmap(i) for i in range(cmap.N)]

for ii in range(len(cmaplist)-20,len(cmaplist)):
    cmaplist[ii] = (0.1,0.1,0.1,0.05)

cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
#bounds_b = array([-0.09,-0.07,-0.05,-0.04,-0.03,-0.02,-0.01,-0.005,0.0])
#bounds = ['-0.09','-0.07','-0.05','-0.04','-0.03','-0.02','-0.01','-0.005','0.0']

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
cbar_ax2 = fig.add_axes([0.65, 0.15, 0.25, 0.03])  
cb2 = fig.colorbar(cs2, cax=cbar_ax2, cmap=cmap, norm=norm, extend='both',spacing='uniform', ticks=bounds_b, boundaries=bounds_b,orientation='horizontal')
#cb2.set_label('$mm^3/mm^3$',fontsize=10,labelpad=-28, x=1.05)
cb2.ax.tick_params(labelsize=15)
cb2.set_label('% Change',fontsize=15)
#cbar2 = map2.colorbar(cs2,location='bottom', extend='max',pad="5%")
#cbar2.ax.tick_params(labelsize=5)
ax2.text(0.3, 0.85, '(c)', verticalalignment='bottom', horizontalalignment='right', transform=ax2.transAxes,fontsize=20)


plt.tight_layout(pad=0, w_pad=0, h_pad=0)


savefig(figDIR + 'Fig2_SMAP_CLMctrl_CLMSMAPKF_centralUSA_5cmSM_ave_' + str(sYEAR) + '_' + str(eYEAR) + '_v3c.png', dpi=600, bbox_inches='tight')
savefig(figDIR + 'Fig2_SMAP_CLMctrl_CLMSMAPKF_centralUSA_5cmSM_ave_' + str(sYEAR) + '_' + str(eYEAR) + '_v3c.pdf', dpi=600, bbox_inches='tight')

plt.close()


