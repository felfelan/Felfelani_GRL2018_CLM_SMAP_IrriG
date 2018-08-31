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
ctrl_DIR      = '/mnt/scratch/felfelan/CESM_simulations/X_021_centralUSA_ICRUCLM45BGCCROPNLDAS_duplicate/'
src = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/'

figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/2018CLM_SMAP_PaperFigures/fig/'
#%%=======================================================================
sYEAR = 2010
eYEAR = 2010
sMONTH = 6
eMONTH = 8

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
		nt = len(smap_FILE.variables['QIRRIG'][:])
		tar_dat_dummy = np.flip(np.ma.masked_greater(smap_FILE.variables['H2OSOI_tar'][:,0:10,],1.0).mean(1),axis=1) # take the average for the first 10 layers 
		tarSMAP_dat_dummy = np.flip(np.ma.masked_greater(smap_FILE.variables['H2OSOI_tarSMAP'][:,0:10,],1.0).mean(1),axis=1)
		SMAP_QIRRIG_dummy = np.flip(smap_FILE.variables['QIRRIG'][:].reshape(nt,400,520),axis=1)
		SMAP_QIRRIG_dummy = SMAP_QIRRIG_dummy * 24 * 3600    # mm/s to mm/day
		
		mcdat = smap_FILE.variables['mcdate'][:]
		print(file)
		if xx == 0:
			tar_dat = tar_dat_dummy
			tarSMAP_dat = tarSMAP_dat_dummy
			SMAP_QIRRIG_dat = SMAP_QIRRIG_dummy
			mcdate = mcdat 
			xx = 1.0
		else:
			tar_dat = np.ma.concatenate([tar_dat, tar_dat_dummy], axis = 0)
			tarSMAP_dat = np.ma.concatenate([tarSMAP_dat, tarSMAP_dat_dummy], axis = 0)
			SMAP_QIRRIG_dat = np.ma.concatenate([SMAP_QIRRIG_dat, SMAP_QIRRIG_dummy], axis = 0)
			mcdate = concatenate((mcdate,mcdat))

# Read CTRL simulation
xx = 0
for file in sorted(os.listdir(ctrl_DIR)):
	if fnmatch.fnmatch(file, '*00000.nc')and int(file[47:51])>= sYEAR and int(file[47:51])<= eYEAR:
		ctrl_FILE = netcdf.netcdf_file(ctrl_DIR + file,'r')
		nt = len(ctrl_FILE.variables['QIRRIG'][:])
		ctrl_QIRRIG_dummy = np.flip(ctrl_FILE.variables['QIRRIG'][:].reshape(nt,400,520),axis=1)
		ctrl_QIRRIG_dummy = ctrl_QIRRIG_dummy * 24 * 3600    # mm/s to mm/day	
		mcdat2 = ctrl_FILE.variables['mcdate'][:]
		print(file)
		if xx == 0:
			ctrl_QIRRIG_dat = ctrl_QIRRIG_dummy
			mcdate2 = mcdat2 
			xx = 1.0
		else:
			ctrl_QIRRIG_dat = np.ma.concatenate([ctrl_QIRRIG_dat, ctrl_QIRRIG_dummy], axis = 0)
			mcdate2 = concatenate((mcdate2,mcdat2))
			

#%%=================================== Area scaling  

# lats and lons of the points

lats = [41.5,34.5,40.4,39.8,33,43]
lons = [-97,-102.7,-101.7,-99.3,-115.5,-112]

ctrl_h2osoi_zero0 = zeros ( ( 1, 400, 520 ) )		
smap_h2osoi_zero0 = zeros ( ( 1, 400, 520 ) )	
	
for i in range(400):
    for j in range(520):
	# if AQdat_hlf[i,j] == 1:
        if tar_h2osoi_dat[i,j] >= 0 and tar_h2osoi_dat[i,j] <= 1:
            ctrl_h2osoi_zero0[0,i,j] = tar_h2osoi_dat[i,j]
            smap_h2osoi_zero0[0,i,j] = tarSMAP_h2osoi_dat[i,j]

h2osoi_zero = np.ma.concatenate([ctrl_h2osoi_zero0,smap_h2osoi_zero0], axis = 0)

			
fig = plt.figure(figsize=(8, 7)) #w,h  
gs1 = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
gs1.update(bottom=0.58, top=0.98,wspace=0.07)

gs2 = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1])
gs2.update(bottom=0.11, top=0.51,wspace=0.3)  

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
	if ii == 0:		
		map1.drawparallels(np.arange(-90,90,5),labels=[1,0,0,0],linewidth=0.0,fontsize=8)
	map1.drawmeridians(np.arange(-180,180,5),labels=[0,0,0,1],linewidth=0.0,fontsize=8)
	
	cs = map1.imshow(ma.masked_less_equal(h2osoi_zero[ii,],0.0),origin='upper',interpolation='nearest',cmap=cmap,norm=norm, vmin=0.0)
	map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/HPA_outline_shapefile/HPA_outline', 'HPA_outline', linewidth=1)
	map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_07_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5, color='grey')
	map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_08_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5, color='grey')
	map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_10_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5, color='grey')
	map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_11_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5, color='grey')
	map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_12_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5, color='grey')
	map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_13_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5, color='grey')
	map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_14_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5, color='grey')
	map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_15_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5, color='grey')
	map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_16_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5, color='grey')
	map1.readshapefile('/mnt/home/felfelan/CESM_DIRs/MyAnalysis/shp_files/WBD_17_Shape/Shape/WBDHU2', 'WBDHU2', linewidth=0.5, color='grey')
	map1.scatter(x=lons,y=lats,latlon='True',facecolors='none',marker = 's',s=30,edgecolors='k')
	ax.text(0.3, 0.85, text[ii], verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,fontsize=20)
	plt.title(title[ii],fontsize=10)
	cbar_ax = fig.add_axes([0.2, 0.57, 0.6, 0.015])    
	cb = fig.colorbar(cs, cax=cbar_ax, cmap=cmap, norm=norm,spacing='uniform', ticks=bounds_b,orientation='horizontal', extend='max')
	cb.set_label('$mm^3/mm^3$',fontsize=10,labelpad=-22, x=1.1)
	cb.ax.tick_params(labelsize=7)
	plt.tight_layout(pad=0, w_pad=0, h_pad=0)



for jj in range(len(lats)):
	PointLatLon = cntralUSA3minxy(lats[jj],lons[jj])

	df_TS = pd.DataFrame(columns=['Date','ctrl_tar','smap_tar', 'smap_qirrig'])
	df_TS['Date'] = pd.to_datetime(mcdate,format='%Y%m%d')
	df_TS['Date'] = df_TS['Date'].dt.date  # just keep the date part and remove time part
	df_TS['Date'] = list(map(str,df_TS['Date']))
	
	df_TS['ctrl_tar'] = tar_dat[:,PointLatLon[1],PointLatLon[0]]
	df_TS['smap_tar'] = tarSMAP_dat[:,PointLatLon[1],PointLatLon[0]]
	df_TS['smap_qirrig'] =	SMAP_QIRRIG_dat[:,PointLatLon[1],PointLatLon[0]]
	df_TS['smap_qirrig'] =	df_TS['smap_qirrig'].replace(0.0, np.nan) #replace with nan
	
	df_TS2 = pd.DataFrame(columns=['Date','ctrl_qirrig'])
	df_TS2['Date'] = pd.to_datetime(mcdate2,format='%Y%m%d')
	df_TS2['Date'] = df_TS2['Date'].dt.date  # just keep the date part and remove time part
	df_TS2['Date'] = list(map(str,df_TS2['Date']))
	df_TS2['ctrl_qirrig'] =	ctrl_QIRRIG_dat[:,PointLatLon[1],PointLatLon[0]]
	df_TS2['ctrl_qirrig'] =	df_TS2['ctrl_qirrig'].replace(0.0, np.nan)
	
	df_TS = df_TS.merge(df_TS2, how='outer', on=['Date'])
	
	ax = plt.subplot(gs2[jj])
	axtwin = ax.twinx()

	sc1 = ax.scatter(df_TS.index,df_TS['ctrl_tar'] ,s = 2 , c = 'tomato', marker = "^", label = 'CTRL Tar_SM')	
	sc2 = ax.scatter(df_TS.index,df_TS['smap_tar'] ,s = 2 , c = 'forestgreen', marker = "h", label = 'SMAP_kf Tar_SM')
	sc3 = axtwin.scatter(df_TS.index,df_TS['ctrl_qirrig'] ,s = 2 , c = 'black', marker = "^", label = 'CTRL QIRRIG')
	sc4 = axtwin.scatter(df_TS.index,df_TS['smap_qirrig'] ,s = 2 , c = 'blue', marker = "h", label = 'SMAP_kf QIRRIG')
	
	gca().set_xticklabels([''])  # remove the xtick labels
	ax.set_xticks(df_TS.index[0:len(df_TS.index):90])
	if jj in [3,4,5] :
		ax.set_xticklabels(df_TS.Date[0:len(df_TS.index):90], fontsize=5, rotation=-45)
		for tick in ax.xaxis.get_majorticklabels():
			tick.set_horizontalalignment("center")
	minorLocator = MultipleLocator(30)
	ax.xaxis.set_minor_locator(minorLocator)
	ax.tick_params(axis = 'both', which = 'major', labelsize = 5)
	axtwin.tick_params(axis = 'both', which = 'major', labelsize = 5)
	plt.title('Lat/Lon ' + str(lats[jj]) + ' , ' + str(lons[jj]),fontsize=7)
	if jj == 0:
		lgnd = plt.legend(handles=[sc1,sc2,sc3,sc4],ncol=1, loc = 'upper left', borderaxespad=0., edgecolor='gray',prop={'size':5})
		ax.set_ylabel('Target SM '+ r'($\mathrm{mm^3/mm^3}$)', fontsize=7, color='k')
		ax.yaxis.set_label_coords(-0.16, -0.15)
	elif jj == 2:	
		axtwin.set_ylabel('Precipitation '+ r'($\mathrm{mm/day}$)', fontsize=7, color='k')
		axtwin.yaxis.set_label_coords(1.16, -0.15)


savefig(figDIR + 'FigS4_targetSM_v2.png', dpi=600, bbox_inches='tight')
savefig(figDIR + 'FigS4_targetSM_v2.pdf', bbox_inches='tight')

plt.close()


