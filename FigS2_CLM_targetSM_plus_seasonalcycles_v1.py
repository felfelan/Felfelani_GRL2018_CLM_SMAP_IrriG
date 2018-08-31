#!~/anaconda3/DIR/bin/ipython

from __future__ import division # force division to be floating point in Python
import numpy as np
from pylab import *
from matplotlib import gridspec
from scipy.interpolate import griddata
import time
import os
import fnmatch
import pandas as pd
import netCDF4 as nc
from scipy.io import netcdf
from numpy import savetxt
from mpl_toolkits.basemap import Basemap,shiftgrid
from latlon2xy import cntralUSA3minxy

#=======================================================================
#===== Farshid Felfelani
#===== First version: 05/10/2017
#===== Reading CLM hlf degree simulations and plot spatial map for irrigation
#%%=======================================================================
SMAP_IRRG_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF/grndOBS_BiasCorrection_minIRR_tarSMAPoutput/'
src = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/'

IRRmask3min = np.load(src + 'Portmann2010_Irrigated_areas_mask_3min_centralUS.npy')
IRRmask3min[IRRmask3min == 0] = -999

figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'
#%%=======================================================================
sYEAR = 2010
eYEAR = 2010
sMONTH = 6
eMONTH = 8

#region 1 SRP
lat11 = 45
lat12 = 41
lon11 = -115.5
lon12 = -110

#region 2 ABA
lat21 = 34
lat22 = 31
lon21 = -115.7
lon22 = -111

#region 3 sHPA
#lat31 = 34.4
#lat32 = 32.3
#lon31 = -103.8
#lon32 = -101.5

#region 3 uHPA
lat31 = 41.7
lat32 = 40.0
lon31 = -102.9
lon32 = -97.5

R1UL = cntralUSA3minxy(lat11,lon11)
R1LR = cntralUSA3minxy(lat12,lon12) 

R2UL = cntralUSA3minxy(lat21,lon21)
R2LR = cntralUSA3minxy(lat22,lon22)

R3UL = cntralUSA3minxy(lat31,lon31)
R3LR = cntralUSA3minxy(lat32,lon32)  

#==============================================================================
# get the 2-D AREA of 3min US central
#==============================================================================

sa_data = nc.Dataset('/mnt/home/felfelan/CESM_DIRs/InData_WFDEI05d/lnd/clm2/\
surfdata_map/surfdata_3x3min_centralUSA_wSMAPdaily_simyr2000_c180108.nc','r')
UScentral_3minAREA = np.ma.masked_less_equal(flipud(sa_data.variables['AREA'][:]) * IRRmask3min , 0.0)

#=================================== read CLM data; Monthly

for yy in range(sYEAR, eYEAR + 1):
    for mm in range(sMONTH, eMONTH + 1):
        smap_IRRG_FILE = netcdf.netcdf_file(SMAP_IRRG_DIR + 'X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF.clm2.h0.' + str(yy).zfill(4) + '-' + str('%02d'%(mm) ) + '.nc','r')
        tar_h2osoi_dat_dummy = smap_IRRG_FILE.variables['H2OSOI_tar'][0,0:10,].reshape(10,400,520).mean(0) # take the average for the first 10 layers 
        tarSMAP_h2osoi_dat_dummy = smap_IRRG_FILE.variables['H2OSOI_tarSMAP'][0,0:10,].reshape(10,400,520).mean(0)

        if yy == sYEAR and mm == sMONTH:
            tar_h2osoi_dat = tar_h2osoi_dat_dummy.reshape(1,400,520)
            tarSMAP_h2osoi_dat = tarSMAP_h2osoi_dat_dummy.reshape(1,400,520)
        else:
            tar_h2osoi_dat = np.ma.concatenate([tar_h2osoi_dat, tar_h2osoi_dat_dummy.reshape(1,400,520)], axis = 0)
            tarSMAP_h2osoi_dat = np.ma.concatenate([tarSMAP_h2osoi_dat, tarSMAP_h2osoi_dat_dummy.reshape(1,400,520)], axis = 0)

tar_h2osoi_dat = tar_h2osoi_dat.mean(0)
tar_h2osoi_dat = flipud(tar_h2osoi_dat)

tarSMAP_h2osoi_dat = tarSMAP_h2osoi_dat.mean(0)
tarSMAP_h2osoi_dat = flipud(tarSMAP_h2osoi_dat)

#=================================== read CLM data; Daily
xx = 0
for file in sorted(os.listdir(SMAP_IRRG_DIR)):
	if fnmatch.fnmatch(file, '*00000.nc'):
	    smap_FILE = netcdf.netcdf_file(SMAP_IRRG_DIR + file,'r')
	    tar_dat_dummy = np.ma.masked_greater(smap_FILE.variables['H2OSOI_tar'][:,0:10,],1.0).mean(1) # take the average for the first 10 layers 
	    tarSMAP_dat_dummy = np.ma.masked_greater(smap_FILE.variables['H2OSOI_tarSMAP'][:,0:10,],1.0).mean(1)
	    mcdat = smap_FILE.variables['mcdate'][:]
	    print(file)
	    if xx == 0:
	        tar_dat = tar_dat_dummy
	        tarSMAP_dat = tarSMAP_dat_dummy
	        mcdate = mcdat 
	        xx = 1.0
	    else:
	        tar_dat = np.ma.concatenate([tar_dat, tar_dat_dummy], axis = 0)
	        tarSMAP_dat = np.ma.concatenate([tarSMAP_dat, tarSMAP_dat_dummy], axis = 0)
	        mcdate = concatenate((mcdate,mcdat))


tar_dat = np.ma.masked_less_equal(flipud(tar_dat) * IRRmask3min , 0.0) * UScentral_3minAREA
tarSMAP_dat = np.ma.masked_less_equal(flipud(tarSMAP_dat) * IRRmask3min , 0.0) * UScentral_3minAREA

df_TS = pd.DataFrame(columns=['Date','ctrl_R1','smap_R1','ctrl_R2','smap_R2','ctrl_R3','smap_R3' ])
df_TS['Date'] = pd.to_datetime(mcdate,format='%Y%m%d')
df_TS['Date'] = df_TS['Date'].dt.date  # just keep the date part and remove time part
df_TS['Date'] = list(map(str,df_TS['Date']))

df_TS['ctrl_R1'] = tar_dat[:,R1UL[1]:R1LR[1],R1UL[0]:R1LR[0]].sum(2).sum(1) / \
						(UScentral_3minAREA[R1UL[1]:R1LR[1],R1UL[0]:R1LR[0]]).sum(1).sum(0)
df_TS['smap_R1'] = tarSMAP_dat[:,R1UL[1]:R1LR[1],R1UL[0]:R1LR[0]].sum(2).sum(1) / \
						(UScentral_3minAREA[R1UL[1]:R1LR[1],R1UL[0]:R1LR[0]]).sum(1).sum(0)
print('R1, min = ', tar_dat[:,R1UL[1]:R1LR[1],R1UL[0]:R1LR[0]].min())
print('R1, Area min = ', (UScentral_3minAREA[R1UL[1]:R1LR[1],R1UL[0]:R1LR[0]]).min())
						
						
df_TS['ctrl_R2'] = tar_dat[:,R2UL[1]:R2LR[1],R2UL[0]:R2LR[0]].sum(2).sum(1) / \
						(UScentral_3minAREA[R2UL[1]:R2LR[1],R2UL[0]:R2LR[0]]).sum(1).sum(0)
df_TS['smap_R2'] = tarSMAP_dat[:,R2UL[1]:R2LR[1],R2UL[0]:R2LR[0]].sum(2).sum(1) / \
						(UScentral_3minAREA[R2UL[1]:R2LR[1],R2UL[0]:R2LR[0]]).sum(1).sum(0)

df_TS['ctrl_R3'] = tar_dat[:,R3UL[1]:R3LR[1],R3UL[0]:R3LR[0]].sum(2).sum(1) / \
						(UScentral_3minAREA[R3UL[1]:R3LR[1],R3UL[0]:R3LR[0]]).sum(1).sum(0)
df_TS['smap_R3'] = tarSMAP_dat[:,R3UL[1]:R3LR[1],R3UL[0]:R3LR[0]].sum(2).sum(1) / \
						(UScentral_3minAREA[R3UL[1]:R3LR[1],R3UL[0]:R3LR[0]]).sum(1).sum(0)
#%%=================================== Area scaling  

ctrl_h2osoi_zero0 = zeros ( ( 1, 400, 520 ) )		
smap_h2osoi_zero0 = zeros ( ( 1, 400, 520 ) )	
	
for i in range(400):
    for j in range(520):
	# if AQdat_hlf[i,j] == 1:
        if tar_h2osoi_dat[i,j] >= 0 and tar_h2osoi_dat[i,j] <= 1:
            ctrl_h2osoi_zero0[0,i,j] = tar_h2osoi_dat[i,j]
            smap_h2osoi_zero0[0,i,j] = tarSMAP_h2osoi_dat[i,j]

h2osoi_zero = np.ma.concatenate([ctrl_h2osoi_zero0,smap_h2osoi_zero0], axis = 0)

			
fig = plt.figure(figsize=(8, 5)) #w,h  
gs1 = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
gs1.update(bottom=0.38, top=0.98,wspace=0.02)

gs2 = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
gs2.update(bottom=0.11, top=0.31,wspace=0.15)  

cmap = plt.cm.jet_r
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[:-70]
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bounds_b = array([0,0.05, 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
bounds = ['0.0','0.05', '0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5']
# I changed the ticks labels to string to have variable number of digits to the right of decimal point 
norm = mpl.colors.BoundaryNorm(bounds_b, cmap.N)
title = ['CTRL Target SM', 'SMAP_kf_BC Target SM']
text = ['a','b']

for ii in range(2):
	ax = plt.subplot(gs1[ii])
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
	cs = map1.imshow(ma.masked_less_equal(h2osoi_zero[ii,]*IRRmask3min,0.0),origin='upper',interpolation='nearest',cmap=cmap,norm=norm, vmin=0.0)
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

	ax.text(0.3, 0.85, text[ii], verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,fontsize=20)
	plt.title(title[ii],fontsize=10)
	cbar_ax = fig.add_axes([0.2, 0.39, 0.6, 0.015])    
	cb = fig.colorbar(cs, cax=cbar_ax, cmap=cmap, norm=norm,spacing='uniform', ticks=bounds_b,orientation='horizontal', extend='max')
	cb.set_label('$mm^3/mm^3$',fontsize=10,labelpad=-29, x=1.1)
	cb.ax.tick_params(labelsize=7)
	plt.tight_layout(pad=0, w_pad=0, h_pad=0)

Rctrl_names = ['ctrl_R1','ctrl_R2','ctrl_R3']
Rsmap_names = ['smap_R1','smap_R2','smap_R3']
Rtitle = ['ABA','SRP','Northern HPA']

for jj in range(3):
	ax = plt.subplot(gs2[jj])
	sc1 = ax.scatter(df_TS.index,df_TS[Rctrl_names[jj]] ,s = 2 , c='tomato',marker="^", label='CTRL Target SM')	
	sc2 = ax.scatter(df_TS.index,df_TS[Rsmap_names[jj]] ,s = 2 , c='forestgreen',marker="h", label='SMAP_kf_BC Target SM')
	gca().set_xticklabels([''])  # remove the xtick labels
	ax.set_xticks(df_TS.index[0:len(df_TS.index):90])
	ax.set_xticklabels(df_TS.Date[0:len(df_TS.index):90], fontsize=5, rotation=-45)
	for tick in ax.xaxis.get_majorticklabels():
		tick.set_horizontalalignment("center")
	minorLocator = MultipleLocator(30)
	ax.xaxis.set_minor_locator(minorLocator)
	ax.tick_params(axis = 'both', which = 'major', labelsize = 5)
	plt.title(Rtitle[jj],fontsize=10)
	if jj == 1:
		lgnd = plt.legend(handles=[sc1,sc2],ncol=1, loc = 'upper right', borderaxespad=0., edgecolor='gray',prop={'size':5})

#savefig(figDIR + 'FigS4_targetSM_v1.png', dpi=600, bbox_inches='tight')
#savefig(figDIR + 'FigS4_targetSM_v1.pdf', bbox_inches='tight')
#
#plt.close()


