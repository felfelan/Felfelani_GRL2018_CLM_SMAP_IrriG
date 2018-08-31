#!~/anaconda3/DIR/bin/ipython

from __future__ import division # force division to be floating point in Python
from numpy import *
from pylab import *
from matplotlib import gridspec
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

# My module to change the latlon to xy coordination in SMAP data
from latlon2xy import smapxy, cyl5minxy, cntralUSA3minxy
#%%=======================================================================
#===== Farshid Felfelani
#===== First version: 04/12/2018
#===== Vertical SM Profile from SCAN, CLM
#%%=======================================================================
CLM_ctrl_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_021_centralUSA_ICRUCLM45BGCCROPNLDAS_duplicate/'
X_024_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr/no_BiasCorrection/'
X_024_kf_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF/no_BiasCorrection/'
X_024_kf_BC_DIR = '/mnt/scratch/felfelan/CESM_simulations/X_024_centralUSA_ICRUCLM45BGCCROPNLDAS_SMAPdailyirr_KF/grndOBS_BiasCorrection_minIRR/'
SMAP_File = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/src/SMAPorig_12_monthly_ave_for_20150331_20170801.npy'
dirSTATIONS = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/scan_uscrn_snotel/'

SCAN_CRN_VP_DIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/scan_uscrn_snotel/CRN_SCAN_Selected_Stations/Vertical_Profile/all_scan_crn_snotel/'
figDIR = '/mnt/home/felfelan/CESM_DIRs/MyAnalysis/fig/SM_vertical_Profiles_seperateStations/'


# Dates for Vertical Profiles
sDate_V = '2005-01-01'
eDate_V = '2016-12-31'

sYR_V = int(sDate_V[:4])
eYR_V = int(eDate_V[:4])
sMON_V = int(sDate_V[5:7])
eMON_V = int(eDate_V[5:7])
sDY_V = int(sDate_V[8:10])
eDY_V = int(eDate_V[8:10]) 


fig = plt.figure() #figsize: w,h

#%%==============================================================================================================================================================
#%%============================================================================================================================================================== 
#%%======================================================================= Vertical Profiles ==================================================================== 
#%%============================================================================================================================================================== 
#%%============================================================================================================================================================== 
df_SCANstations = pd.read_csv(dirSTATIONS + 'SCAN_stations_ID_latlon.txt',sep='\t',header=0, error_bad_lines=False)
df_SNOTELstations = pd.read_csv('/mnt/home/felfelan/DATA/soil_moisture/SNOTEL/' + 'SNOTEL_stations_ID_latlon.txt',sep='\t',header=0, error_bad_lines=False)

nSTATION = 0 
# for file in sorted(os.listdir(SCAN_CRN_VP_DIR)):
    # if os.path.isfile(os.path.join(SCAN_CRN_VP_DIR, file)) and 'SCAN' not in file:  #list only files not directories; and only CRN files
        # print(file)
        # nSTATION = nSTATION + 1       
        #### Reading from the header below
        # df = pd.read_csv(SCAN_CRN_VP_DIR +file,sep=',', header=0, error_bad_lines=False)
        # df[df==-9999]=NaN
        # df[df==-99]=NaN
        # my_dict2[file] = df

		

my_dict2 = {}		
for file in sorted(os.listdir(SCAN_CRN_VP_DIR)):
    if os.path.isfile(os.path.join(SCAN_CRN_VP_DIR, file)):  #list only files not directories
        print(file)
        nSTATION = nSTATION + 1
        if ('SCAN' in file) or ('SNOTEL' in file):
        # Getting the row number of the header
            aa = open(SCAN_CRN_VP_DIR + file).readlines()
            for ii in range(len(aa)):
                if not fnmatch.fnmatch(aa[ii],'#*'):
                                   headerINDEX = ii
                                   break
            
            # Reading from the header below
            df = pd.read_csv(SCAN_CRN_VP_DIR +file,sep=',', header=headerINDEX, error_bad_lines=False)
        else:
            
            df = pd.read_csv(SCAN_CRN_VP_DIR +file,sep=',', header=0, error_bad_lines=False)
            df[df==-9999]=NaN
            df[df==-99]=NaN
            
        my_dict2[file] = df		
		
# Adding Columns from different dataframes (with matching date) stored in a python dictionary 
# Initialize 'sum_df'


# Iterate over dataframes of dictionary
for i, tables in enumerate(my_dict2):
	sum_df1 = pd.DataFrame(columns=['date'])
	sum_df2 = pd.DataFrame(columns=['date'])
	sum_df3 = pd.DataFrame(columns=['date'])
	sum_df4 = pd.DataFrame(columns=['date'])
	sum_df5 = pd.DataFrame(columns=['date'])
	print('STATION: ',tables)
		# Create dataframe
	df = my_dict2[tables]
	if ('SCAN' in tables) or ('SNOTEL' in tables):
		# Filter rows by 'date'
		df = df[(df['Date'] >= sDate_V) & (df['Date'] <= eDate_V)] 
		
		if ('SCAN' in tables):
			lati = df_SCANstations[df_SCANstations['site_name'].str.contains(tables[5:9])].latitude.iloc[0]
			long = df_SCANstations[df_SCANstations['site_name'].str.contains(tables[5:9])].longitude.iloc[0]
		elif ('SNOTEL' in tables):
			lati = df_SNOTELstations[df_SNOTELstations['site_name'].str.contains('(' + tables.split('_')[1] + ')')].lat.iloc[0]
			long = df_SNOTELstations[df_SNOTELstations['site_name'].str.contains('(' + tables.split('_')[1] + ')')].lon.iloc[0]		
		
		
		# Filter for all columns starting with "Soil Moisture Percent -2in" for top 5cm 
		filter_col1 = [col for col in list(df) if col.startswith('Soil Moisture Percent -2in')]
		filter_col2 = [col for col in list(df) if col.startswith('Soil Moisture Percent -4in')]    
		filter_col3 = [col for col in list(df) if col.startswith('Soil Moisture Percent -8in')]    
		filter_col4 = [col for col in list(df) if col.startswith('Soil Moisture Percent -20in')]    
		filter_col5 = [col for col in list(df) if col.startswith('Soil Moisture Percent -40in')]    

		# Keep only proper cols
		df_dummy1 = df[['Date'] + filter_col1]
		df_dummy2 = df[['Date'] + filter_col2]
		df_dummy3 = df[['Date'] + filter_col3]
		df_dummy4 = df[['Date'] + filter_col4]
		df_dummy5 = df[['Date'] + filter_col5]
		
		# removing the leap year days
		df1 = df_dummy1[~df_dummy1.Date.str.endswith('02-29')]
		df2 = df_dummy2[~df_dummy2.Date.str.endswith('02-29')]
		df3 = df_dummy3[~df_dummy3.Date.str.endswith('02-29')]
		df4 = df_dummy4[~df_dummy4.Date.str.endswith('02-29')]
		df5 = df_dummy5[~df_dummy5.Date.str.endswith('02-29')]

		# Join new columns from dictionary to old 'sum_df' dataframe
		# if i == 0:
		sum_df1 = df1.rename(columns={i:'{}'.format(tables) for i in filter_col1}).copy()
		sum_df2 = df2.rename(columns={i:'{}'.format(tables) for i in filter_col2}).copy()
		sum_df3 = df3.rename(columns={i:'{}'.format(tables) for i in filter_col3}).copy()
		sum_df4 = df4.rename(columns={i:'{}'.format(tables) for i in filter_col4}).copy()
		sum_df5 = df5.rename(columns={i:'{}'.format(tables) for i in filter_col5}).copy()

		# else:
			# df1 = df1.rename(columns={i:'{}'.format(tables) for i in filter_col1})
			# sum_df1 = df1.merge(sum_df1, how='outer', on=['Date'])
			# df2 = df2.rename(columns={i:'{}'.format(tables) for i in filter_col2})
			# sum_df2 = df2.merge(sum_df2, how='outer', on=['Date'])
			# df3 = df3.rename(columns={i:'{}'.format(tables) for i in filter_col3})
			# sum_df3 = df3.merge(sum_df3, how='outer', on=['Date'])
			# df4 = df4.rename(columns={i:'{}'.format(tables) for i in filter_col4})
			# sum_df4 = df4.merge(sum_df4, how='outer', on=['Date'])
			# df5 = df5.rename(columns={i:'{}'.format(tables) for i in filter_col5})
			# sum_df5 = df5.merge(sum_df5, how='outer', on=['Date'])

	if 'USCRN' in tables:
		df = df.rename(index=str, columns={"LST_DATE": "Date"})
		df['Date'] = pd.to_datetime(df.Date,format='%Y%m%d')
		df['Date'] = df['Date'].dt.date  # just keep the date part and remove time part
		df['Date'] = list(map(str,df['Date']))

		lati = df.LATITUDE[0]
		long = df.LONGITUDE[0]
		# Filter rows by 'date'
		df = df[(df['Date'] >= sDate_V) & (df['Date'] <= eDate_V)] 

		# Filter for all columns starting with "Soil Moisture Percent -2in" for top 5cm 
		filter_col1 = [col for col in list(df) if col.startswith('SOIL_MOISTURE_5_DAILY')]
		filter_col2 = [col for col in list(df) if col.startswith('SOIL_MOISTURE_10_DAILY')]    
		filter_col3 = [col for col in list(df) if col.startswith('SOIL_MOISTURE_20_DAILY')]    
		filter_col4 = [col for col in list(df) if col.startswith('SOIL_MOISTURE_50_DAILY')]    
		filter_col5 = [col for col in list(df) if col.startswith('SOIL_MOISTURE_100_DAILY')]    

		# Keep only proper cols
		df_dummy1 = df[['Date'] + filter_col1]
		df_dummy2 = df[['Date'] + filter_col2]
		df_dummy3 = df[['Date'] + filter_col3]
		df_dummy4 = df[['Date'] + filter_col4]
		df_dummy5 = df[['Date'] + filter_col5]
		
		# removing the leap year days
		df1 = df_dummy1[~df_dummy1.Date.str.endswith('02-29')]
		df2 = df_dummy2[~df_dummy2.Date.str.endswith('02-29')]
		df3 = df_dummy3[~df_dummy3.Date.str.endswith('02-29')]
		df4 = df_dummy4[~df_dummy4.Date.str.endswith('02-29')]
		df5 = df_dummy5[~df_dummy5.Date.str.endswith('02-29')]

		# Join new columns from dictionary to old 'sum_df' dataframe
		# if i == 0:
		sum_df1 = df1.rename(columns={i:'{}'.format(tables) for i in filter_col1}).copy()
		sum_df2 = df2.rename(columns={i:'{}'.format(tables) for i in filter_col2}).copy()
		sum_df3 = df3.rename(columns={i:'{}'.format(tables) for i in filter_col3}).copy()
		sum_df4 = df4.rename(columns={i:'{}'.format(tables) for i in filter_col4}).copy()
		sum_df5 = df5.rename(columns={i:'{}'.format(tables) for i in filter_col5}).copy()
		# else:
			# df1 = df1.rename(columns={i:'{}'.format(tables) for i in filter_col1})
			# sum_df1 = df1.merge(sum_df1, how='outer', on=['Date']) 
			# df2 = df2.rename(columns={i:'{}'.format(tables) for i in filter_col2})
			# sum_df2 = df2.merge(sum_df2, how='outer', on=['Date'])
			# df3 = df3.rename(columns={i:'{}'.format(tables) for i in filter_col3})
			# sum_df3 = df3.merge(sum_df3, how='outer', on=['Date'])
			# df4 = df4.rename(columns={i:'{}'.format(tables) for i in filter_col4})
			# sum_df4 = df4.merge(sum_df4, how='outer', on=['Date'])
			# df5 = df5.rename(columns={i:'{}'.format(tables) for i in filter_col5})
			# sum_df5 = df5.merge(sum_df5, how='outer', on=['Date'])

	# sort based on the Date and reindex
	sum_df1 = sum_df1.sort_values(["Date"])
	sum_df1 = sum_df1.reset_index(drop=True)
	sum_df2 = sum_df2.sort_values(["Date"])
	sum_df2 = sum_df2.reset_index(drop=True)
	sum_df3 = sum_df3.sort_values(["Date"])
	sum_df3 = sum_df3.reset_index(drop=True)
	sum_df4 = sum_df4.sort_values(["Date"])
	sum_df4 = sum_df4.reset_index(drop=True)
	sum_df5 = sum_df5.sort_values(["Date"])
	sum_df5 = sum_df5.reset_index(drop=True)

	# if 'SCAN' in tables:
		# TABLE_Both = tables[0:9]
		# TABLE_Both0 = tables[:-4]
	# else:
		# TABLE_Both = tables
		# TABLE_Both0 = tables
	TABLE_Both = tables
	TABLE_Both0 = tables
	
	if (lati >= 30) and (lati <= 50) and (long >= -116) and (long <= -90):
		#%%======================================================================= read Control CLM data
		xx = 0
		for file in sorted(os.listdir(CLM_ctrl_DIR)):
			if fnmatch.fnmatch(file, '*00000.nc') and int(file[47:51])>= sYR_V and int(file[47:51])<= eYR_V:
				# print(file)
				IRRG_FILE = netcdf.netcdf_file(CLM_ctrl_DIR + file,'r')
				nt = len(IRRG_FILE.variables['QIRRIG'][:])
	#        print(nt)

				mcdat4 = IRRG_FILE.variables['mcdate'][:]

				IRRG_QIRRIG0 = flip(IRRG_FILE.variables['QIRRIG'][:].reshape(nt,400,520),axis=1)	   
				IRRG_h2osoi_dat0 = IRRG_FILE.variables['H2OSOI'][:]	
				IRRG_h2osoi_dat01 = flip(IRRG_h2osoi_dat0[:,0,].reshape(nt,400,520),axis=1) 
				IRRG_h2osoi_dat02 = flip(IRRG_h2osoi_dat0[:,1,].reshape(nt,400,520),axis=1)
				IRRG_h2osoi_dat03 = flip(IRRG_h2osoi_dat0[:,2,].reshape(nt,400,520),axis=1)
				IRRG_h2osoi_dat04 = flip(IRRG_h2osoi_dat0[:,3,].reshape(nt,400,520),axis=1)
				IRRG_h2osoi_dat05 = flip(IRRG_h2osoi_dat0[:,4,].reshape(nt,400,520),axis=1)
				IRRG_h2osoi_dat06 = flip(IRRG_h2osoi_dat0[:,5,].reshape(nt,400,520),axis=1)
				IRRG_h2osoi_dat07 = flip(IRRG_h2osoi_dat0[:,6,].reshape(nt,400,520),axis=1)
				IRRG_h2osoi_dat08 = flip(IRRG_h2osoi_dat0[:,7,].reshape(nt,400,520),axis=1)
				# print(file[47:51],file[52:54],file[55:57])
				
				# if file[47:51] == str(sYR_V) and file[52:54] == str(sMON).zfill(2) and file[55:57] == str(sDY_T).zfill(2):
				if file[47:51] == str(sYR_V) and xx == 0:
					print('file0:',file)
					IRRG_QIRRIG_dat = IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					IRRG_h2osoi_dat1 = IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					IRRG_h2osoi_dat2 = IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					IRRG_h2osoi_dat3 = IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					IRRG_h2osoi_dat4 = IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					IRRG_h2osoi_dat5 = IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					IRRG_h2osoi_dat6 = IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					IRRG_h2osoi_dat7 = IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					IRRG_h2osoi_dat8 = IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					mcdate4 = mcdat4                
					xx = 1
					
				else:
					#print('file:',file)
					IRRG_QIRRIG_dat = concatenate(( IRRG_QIRRIG_dat, IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					IRRG_h2osoi_dat1 = concatenate(( IRRG_h2osoi_dat1, IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					IRRG_h2osoi_dat2 = concatenate(( IRRG_h2osoi_dat2, IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					IRRG_h2osoi_dat3 = concatenate(( IRRG_h2osoi_dat3, IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					IRRG_h2osoi_dat4 = concatenate(( IRRG_h2osoi_dat4, IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					IRRG_h2osoi_dat5 = concatenate(( IRRG_h2osoi_dat5, IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					IRRG_h2osoi_dat6 = concatenate(( IRRG_h2osoi_dat6, IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					IRRG_h2osoi_dat7 = concatenate(( IRRG_h2osoi_dat7, IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					IRRG_h2osoi_dat8 = concatenate(( IRRG_h2osoi_dat8, IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))	 
					mcdate4 = concatenate((mcdate4,mcdat4))				
		# consider the SM of top 5 cm soil layer as the SM
		h2osoi_5cm = IRRG_h2osoi_dat1*(0.014201/0.041649) + IRRG_h2osoi_dat2*(0.027447/0.041649) 	
		IRRG_QIRRIG_dat = IRRG_QIRRIG_dat * 24 * 3600  #mm/s to mm/day	

		df_ctrl_h2osoi01 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L1'])
		df_ctrl_h2osoi01['Date'] = pd.to_datetime(mcdate4,format='%Y%m%d')
		df_ctrl_h2osoi01['Date'] = df_ctrl_h2osoi01['Date'].dt.date  # just keep the date part and remove time part
		df_ctrl_h2osoi01['Date'] = list(map(str,df_ctrl_h2osoi01['Date']))
		df_ctrl_h2osoi01[TABLE_Both + '_ctrl_L1'] = IRRG_h2osoi_dat1 
		# the Date column of sum_df is str type, 
		#this (Date column of df_ctrl_h2osoi01) should be also str to be merged in sum_df, 
		#so I used map function to make all elements as string
		sum_df1 = sum_df1.merge(df_ctrl_h2osoi01, how='outer', on=['Date'])

		df_ctrl_h2osoi02 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L2'])
		df_ctrl_h2osoi02['Date'] = df_ctrl_h2osoi01['Date']
		df_ctrl_h2osoi02[TABLE_Both + '_ctrl_L2'] = IRRG_h2osoi_dat2
		sum_df1 = sum_df1.merge(df_ctrl_h2osoi02, how='outer', on=['Date'])

		df_ctrl_h2osoi1 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L3'])
		df_ctrl_h2osoi1['Date'] = df_ctrl_h2osoi01['Date']
		df_ctrl_h2osoi1[TABLE_Both + '_ctrl_L3'] = IRRG_h2osoi_dat3
		sum_df1 = sum_df1.merge(df_ctrl_h2osoi1, how='outer', on=['Date'])	
		
		df_ctrl_h2osoi2 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L4'])
		df_ctrl_h2osoi2['Date'] = df_ctrl_h2osoi01['Date']
		df_ctrl_h2osoi2[TABLE_Both + '_ctrl_L4'] = IRRG_h2osoi_dat4
		sum_df2 = sum_df2.merge(df_ctrl_h2osoi2, how='outer', on=['Date'])	

		df_ctrl_h2osoi3 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L5'])
		df_ctrl_h2osoi3['Date'] = df_ctrl_h2osoi01['Date']
		df_ctrl_h2osoi3[TABLE_Both + '_ctrl_L5'] = IRRG_h2osoi_dat5
		sum_df3 = sum_df3.merge(df_ctrl_h2osoi3, how='outer', on=['Date'])	
		
		df_ctrl_h2osoi4 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L6'])
		df_ctrl_h2osoi4['Date'] = df_ctrl_h2osoi01['Date']
		df_ctrl_h2osoi4[TABLE_Both + '_ctrl_L6'] = IRRG_h2osoi_dat6
		sum_df4 = sum_df4.merge(df_ctrl_h2osoi4, how='outer', on=['Date'])	
		
		df_ctrl_h2osoi5 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L7'])
		df_ctrl_h2osoi5['Date'] = df_ctrl_h2osoi01['Date']
		df_ctrl_h2osoi5[TABLE_Both + '_ctrl_L7'] = IRRG_h2osoi_dat7
		sum_df5 = sum_df5.merge(df_ctrl_h2osoi5, how='outer', on=['Date'])	  

		df_ctrl_h2osoi6 = pd.DataFrame(columns=['Date',TABLE_Both + '_ctrl_L8'])
		df_ctrl_h2osoi6['Date'] = df_ctrl_h2osoi01['Date']
		df_ctrl_h2osoi6[TABLE_Both + '_ctrl_L8'] = IRRG_h2osoi_dat8
		sum_df5 = sum_df5.merge(df_ctrl_h2osoi6, how='outer', on=['Date'])
		
	#%%======================================================================= read SMAP CLM data
		xx = 0
		for file in sorted(os.listdir(X_024_DIR)):
			if fnmatch.fnmatch(file, '*00000.nc') and int(file[60:64])>= sYR_V and int(file[60:64])<= eYR_V:
				SMAP_IRRG_FILE = netcdf.netcdf_file(X_024_DIR + file,'r')
				nt = len(SMAP_IRRG_FILE.variables['QIRRIG'][:])

				mcdat5 = SMAP_IRRG_FILE.variables['mcdate'][:]
				
				SMAP_IRRG_QIRRIG0 = flip(SMAP_IRRG_FILE.variables['QIRRIG'][:].reshape(nt,400,520),axis=1)	   
				SMAP_IRRG_h2osoi_dat0 = SMAP_IRRG_FILE.variables['H2OSOI'][:]	
				SMAP_IRRG_h2osoi_dat01 = flip(SMAP_IRRG_h2osoi_dat0[:,0,].reshape(nt,400,520),axis=1) 
				SMAP_IRRG_h2osoi_dat02 = flip(SMAP_IRRG_h2osoi_dat0[:,1,].reshape(nt,400,520),axis=1)
				SMAP_IRRG_h2osoi_dat03 = flip(SMAP_IRRG_h2osoi_dat0[:,2,].reshape(nt,400,520),axis=1)
				SMAP_IRRG_h2osoi_dat04 = flip(SMAP_IRRG_h2osoi_dat0[:,3,].reshape(nt,400,520),axis=1)
				SMAP_IRRG_h2osoi_dat05 = flip(SMAP_IRRG_h2osoi_dat0[:,4,].reshape(nt,400,520),axis=1)
				SMAP_IRRG_h2osoi_dat06 = flip(SMAP_IRRG_h2osoi_dat0[:,5,].reshape(nt,400,520),axis=1)
				SMAP_IRRG_h2osoi_dat07 = flip(SMAP_IRRG_h2osoi_dat0[:,6,].reshape(nt,400,520),axis=1)
				SMAP_IRRG_h2osoi_dat08 = flip(SMAP_IRRG_h2osoi_dat0[:,7,].reshape(nt,400,520),axis=1)

				if file[60:64] == str(sYR_V) and xx==0:
					print('file0:',file)
					SMAP_IRRG_QIRRIG_dat = SMAP_IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAP_IRRG_h2osoi_dat1 = SMAP_IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAP_IRRG_h2osoi_dat2 = SMAP_IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAP_IRRG_h2osoi_dat3 = SMAP_IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAP_IRRG_h2osoi_dat4 = SMAP_IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAP_IRRG_h2osoi_dat5 = SMAP_IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAP_IRRG_h2osoi_dat6 = SMAP_IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAP_IRRG_h2osoi_dat7 = SMAP_IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAP_IRRG_h2osoi_dat8 = SMAP_IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					mcdate5 = mcdat5
					xx = 1

				else:
					#print('file:',file)
					SMAP_IRRG_QIRRIG_dat = concatenate(( SMAP_IRRG_QIRRIG_dat, SMAP_IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAP_IRRG_h2osoi_dat1 = concatenate(( SMAP_IRRG_h2osoi_dat1, SMAP_IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAP_IRRG_h2osoi_dat2 = concatenate(( SMAP_IRRG_h2osoi_dat2, SMAP_IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAP_IRRG_h2osoi_dat3 = concatenate(( SMAP_IRRG_h2osoi_dat3, SMAP_IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAP_IRRG_h2osoi_dat4 = concatenate(( SMAP_IRRG_h2osoi_dat4, SMAP_IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAP_IRRG_h2osoi_dat5 = concatenate(( SMAP_IRRG_h2osoi_dat5, SMAP_IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAP_IRRG_h2osoi_dat6 = concatenate(( SMAP_IRRG_h2osoi_dat6, SMAP_IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAP_IRRG_h2osoi_dat7 = concatenate(( SMAP_IRRG_h2osoi_dat7, SMAP_IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAP_IRRG_h2osoi_dat8 = concatenate(( SMAP_IRRG_h2osoi_dat8, SMAP_IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					mcdate5 = concatenate((mcdate5,mcdat5))

		SMAP_IRRG_h2osoi_dat = SMAP_IRRG_h2osoi_dat1*(0.014201/0.041649) + SMAP_IRRG_h2osoi_dat2*(0.027447/0.041649) 	
		SMAP_IRRG_QIRRIG_dat = SMAP_IRRG_QIRRIG_dat * 24 * 3600  #mm/s to mm/day	

		SMAP_QIRRIG = SMAP_IRRG_QIRRIG_dat
		SMAP_h2osoi_5cm = SMAP_IRRG_h2osoi_dat

		df_smap_h2osoi01 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L1'])
		df_smap_h2osoi01['Date'] = pd.to_datetime(mcdate5,format='%Y%m%d')
		df_smap_h2osoi01['Date'] = df_smap_h2osoi01['Date'].dt.date  # just keep the date part and remove time part
		df_smap_h2osoi01['Date'] = list(map(str,df_smap_h2osoi01['Date']))	
		df_smap_h2osoi01[TABLE_Both + '_SMAP_L1'] = SMAP_IRRG_h2osoi_dat1
		sum_df1 = sum_df1.merge(df_smap_h2osoi01, how='outer', on=['Date'])

		df_smap_h2osoi02 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L2'])
		df_smap_h2osoi02['Date'] = df_smap_h2osoi01['Date']
		df_smap_h2osoi02[TABLE_Both + '_SMAP_L2'] = SMAP_IRRG_h2osoi_dat2
		sum_df1 = sum_df1.merge(df_smap_h2osoi02, how='outer', on=['Date'])	

		df_smap_h2osoi1 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L3'])
		df_smap_h2osoi1['Date'] = df_smap_h2osoi01['Date']
		df_smap_h2osoi1[TABLE_Both + '_SMAP_L3'] = SMAP_IRRG_h2osoi_dat3
		sum_df1 = sum_df1.merge(df_smap_h2osoi1, how='outer', on=['Date'])	
		
		df_smap_h2osoi2 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L4'])
		df_smap_h2osoi2['Date'] = df_smap_h2osoi01['Date']
		df_smap_h2osoi2[TABLE_Both + '_SMAP_L4'] = SMAP_IRRG_h2osoi_dat4
		sum_df2 = sum_df2.merge(df_smap_h2osoi2, how='outer', on=['Date'])	
		
		df_smap_h2osoi3 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L5'])
		df_smap_h2osoi3['Date'] = df_smap_h2osoi01['Date']
		df_smap_h2osoi3[TABLE_Both + '_SMAP_L5'] = SMAP_IRRG_h2osoi_dat5
		sum_df3 = sum_df3.merge(df_smap_h2osoi3, how='outer', on=['Date'])

		df_smap_h2osoi4 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L6'])
		df_smap_h2osoi4['Date'] = df_smap_h2osoi01['Date']
		df_smap_h2osoi4[TABLE_Both + '_SMAP_L6'] = SMAP_IRRG_h2osoi_dat6
		sum_df4 = sum_df4.merge(df_smap_h2osoi4, how='outer', on=['Date'])

		df_smap_h2osoi5 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L7'])
		df_smap_h2osoi5['Date'] = df_smap_h2osoi01['Date']
		df_smap_h2osoi5[TABLE_Both + '_SMAP_L7'] = SMAP_IRRG_h2osoi_dat7
		sum_df5 = sum_df5.merge(df_smap_h2osoi5, how='outer', on=['Date'])  

		df_smap_h2osoi6 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAP_L8'])
		df_smap_h2osoi6['Date'] = df_smap_h2osoi01['Date']
		df_smap_h2osoi6[TABLE_Both + '_SMAP_L8'] = SMAP_IRRG_h2osoi_dat8
		sum_df5 = sum_df5.merge(df_smap_h2osoi6, how='outer', on=['Date'])	
		
	#%%======================================================================= read SMAP CLM X_024 daily, KF
		xx = 0
		for file in sorted(os.listdir(X_024_kf_DIR)):
			# print(file)
			if fnmatch.fnmatch(file, '*00000.nc') and int(file[63:67])>= sYR_V and int(file[63:67])<= eYR_V:
				SMAPdailyCVEX_IRRG_FILE = netcdf.netcdf_file(X_024_kf_DIR + file,'r')
				nt = len(SMAPdailyCVEX_IRRG_FILE.variables['QIRRIG'][:])

				mcdat6 = SMAPdailyCVEX_IRRG_FILE.variables['mcdate'][:]
				
				SMAPdailyCVEX_IRRG_QIRRIG0 = flip(SMAPdailyCVEX_IRRG_FILE.variables['QIRRIG'][:].reshape(nt,400,520),axis=1)	   
				SMAPdailyCVEX_IRRG_h2osoi_dat0 = SMAPdailyCVEX_IRRG_FILE.variables['H2OSOI'][:]	
				SMAPdailyCVEX_IRRG_h2osoi_dat01 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,0,].reshape(nt,400,520),axis=1) 
				SMAPdailyCVEX_IRRG_h2osoi_dat02 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,1,].reshape(nt,400,520),axis=1)
				SMAPdailyCVEX_IRRG_h2osoi_dat03 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,2,].reshape(nt,400,520),axis=1)
				SMAPdailyCVEX_IRRG_h2osoi_dat04 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,3,].reshape(nt,400,520),axis=1)
				SMAPdailyCVEX_IRRG_h2osoi_dat05 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,4,].reshape(nt,400,520),axis=1)
				SMAPdailyCVEX_IRRG_h2osoi_dat06 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,5,].reshape(nt,400,520),axis=1)
				SMAPdailyCVEX_IRRG_h2osoi_dat07 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,6,].reshape(nt,400,520),axis=1)
				SMAPdailyCVEX_IRRG_h2osoi_dat08 = flip(SMAPdailyCVEX_IRRG_h2osoi_dat0[:,7,].reshape(nt,400,520),axis=1)

				if file[63:67] == str(sYR_V) and xx==0:
					print('file0:',file)
					SMAPdailyCVEX_IRRG_QIRRIG_dat = SMAPdailyCVEX_IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAPdailyCVEX_IRRG_h2osoi_dat1 = SMAPdailyCVEX_IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAPdailyCVEX_IRRG_h2osoi_dat2 = SMAPdailyCVEX_IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAPdailyCVEX_IRRG_h2osoi_dat3 = SMAPdailyCVEX_IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAPdailyCVEX_IRRG_h2osoi_dat4 = SMAPdailyCVEX_IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAPdailyCVEX_IRRG_h2osoi_dat5 = SMAPdailyCVEX_IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAPdailyCVEX_IRRG_h2osoi_dat6 = SMAPdailyCVEX_IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAPdailyCVEX_IRRG_h2osoi_dat7 = SMAPdailyCVEX_IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					SMAPdailyCVEX_IRRG_h2osoi_dat8 = SMAPdailyCVEX_IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]]
					mcdate6 = mcdat6
					xx = 1

				else:
					#print('file:',file)
					SMAPdailyCVEX_IRRG_QIRRIG_dat = concatenate(( SMAPdailyCVEX_IRRG_QIRRIG_dat, SMAPdailyCVEX_IRRG_QIRRIG0[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAPdailyCVEX_IRRG_h2osoi_dat1 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat1, SMAPdailyCVEX_IRRG_h2osoi_dat01[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAPdailyCVEX_IRRG_h2osoi_dat2 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat2, SMAPdailyCVEX_IRRG_h2osoi_dat02[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAPdailyCVEX_IRRG_h2osoi_dat3 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat3, SMAPdailyCVEX_IRRG_h2osoi_dat03[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAPdailyCVEX_IRRG_h2osoi_dat4 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat4, SMAPdailyCVEX_IRRG_h2osoi_dat04[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAPdailyCVEX_IRRG_h2osoi_dat5 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat5, SMAPdailyCVEX_IRRG_h2osoi_dat05[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAPdailyCVEX_IRRG_h2osoi_dat6 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat6, SMAPdailyCVEX_IRRG_h2osoi_dat06[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAPdailyCVEX_IRRG_h2osoi_dat7 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat7, SMAPdailyCVEX_IRRG_h2osoi_dat07[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					SMAPdailyCVEX_IRRG_h2osoi_dat8 = concatenate(( SMAPdailyCVEX_IRRG_h2osoi_dat8, SMAPdailyCVEX_IRRG_h2osoi_dat08[:,cntralUSA3minxy(lati,long)[1],cntralUSA3minxy(lati,long)[0]] ))
					mcdate6 = concatenate((mcdate6,mcdat6))	

		SMAPdailyCVEX_IRRG_h2osoi_dat = SMAPdailyCVEX_IRRG_h2osoi_dat1*(0.014201/0.041649) + SMAPdailyCVEX_IRRG_h2osoi_dat2*(0.027447/0.041649) 	
		SMAPdailyCVEX_IRRG_QIRRIG_dat = SMAPdailyCVEX_IRRG_QIRRIG_dat * 24 * 3600  #mm/s to mm/day	

		SMAPdailyCVEX_QIRRIG = SMAPdailyCVEX_IRRG_QIRRIG_dat
		SMAPdailyCVEX_h2osoi_5cm = SMAPdailyCVEX_IRRG_h2osoi_dat

		
		df_SMAPdailyCVEX_h2osoi01 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L1'])
		df_SMAPdailyCVEX_h2osoi01['Date'] = pd.to_datetime(mcdate6,format='%Y%m%d')
		df_SMAPdailyCVEX_h2osoi01['Date'] = df_SMAPdailyCVEX_h2osoi01['Date'].dt.date  # just keep the date part and remove time part
		df_SMAPdailyCVEX_h2osoi01['Date'] = list(map(str,df_SMAPdailyCVEX_h2osoi01['Date']))	
		df_SMAPdailyCVEX_h2osoi01[TABLE_Both + '_SMAPdailyCVEX_L1'] = SMAPdailyCVEX_IRRG_h2osoi_dat1
		sum_df1 = sum_df1.merge(df_SMAPdailyCVEX_h2osoi01, how='outer', on=['Date'])

		df_SMAPdailyCVEX_h2osoi02 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L2'])
		df_SMAPdailyCVEX_h2osoi02['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
		df_SMAPdailyCVEX_h2osoi02[TABLE_Both + '_SMAPdailyCVEX_L2'] = SMAPdailyCVEX_IRRG_h2osoi_dat2
		sum_df1 = sum_df1.merge(df_SMAPdailyCVEX_h2osoi02, how='outer', on=['Date'])	

		df_SMAPdailyCVEX_h2osoi1 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L3'])
		df_SMAPdailyCVEX_h2osoi1['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
		df_SMAPdailyCVEX_h2osoi1[TABLE_Both + '_SMAPdailyCVEX_L3'] = SMAPdailyCVEX_IRRG_h2osoi_dat3
		sum_df1 = sum_df1.merge(df_SMAPdailyCVEX_h2osoi1, how='outer', on=['Date'])	
		
		df_SMAPdailyCVEX_h2osoi2 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L4'])
		df_SMAPdailyCVEX_h2osoi2['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
		df_SMAPdailyCVEX_h2osoi2[TABLE_Both + '_SMAPdailyCVEX_L4'] = SMAPdailyCVEX_IRRG_h2osoi_dat4
		sum_df2 = sum_df2.merge(df_SMAPdailyCVEX_h2osoi2, how='outer', on=['Date'])	
		
		df_SMAPdailyCVEX_h2osoi3 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L5'])
		df_SMAPdailyCVEX_h2osoi3['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
		df_SMAPdailyCVEX_h2osoi3[TABLE_Both + '_SMAPdailyCVEX_L5'] = SMAPdailyCVEX_IRRG_h2osoi_dat5
		sum_df3 = sum_df3.merge(df_SMAPdailyCVEX_h2osoi3, how='outer', on=['Date'])

		df_SMAPdailyCVEX_h2osoi4 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L6'])
		df_SMAPdailyCVEX_h2osoi4['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
		df_SMAPdailyCVEX_h2osoi4[TABLE_Both + '_SMAPdailyCVEX_L6'] = SMAPdailyCVEX_IRRG_h2osoi_dat6
		sum_df4 = sum_df4.merge(df_SMAPdailyCVEX_h2osoi4, how='outer', on=['Date'])

		df_SMAPdailyCVEX_h2osoi5 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L7'])
		df_SMAPdailyCVEX_h2osoi5['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
		df_SMAPdailyCVEX_h2osoi5[TABLE_Both + '_SMAPdailyCVEX_L7'] = SMAPdailyCVEX_IRRG_h2osoi_dat7
		sum_df5 = sum_df5.merge(df_SMAPdailyCVEX_h2osoi5, how='outer', on=['Date']) 	

		df_SMAPdailyCVEX_h2osoi6 = pd.DataFrame(columns=['Date',TABLE_Both + '_SMAPdailyCVEX_L8'])
		df_SMAPdailyCVEX_h2osoi6['Date'] = df_SMAPdailyCVEX_h2osoi01['Date']
		df_SMAPdailyCVEX_h2osoi6[TABLE_Both + '_SMAPdailyCVEX_L8'] = SMAPdailyCVEX_IRRG_h2osoi_dat8
		sum_df5 = sum_df5.merge(df_SMAPdailyCVEX_h2osoi6, how='outer', on=['Date']) 	
		
	#%%======================================================================= Read SMAP data
	#	SMAP_monthly = np.load(SMAP_File)[int(AveMon)-1,smapxy(lati,long)[1],smapxy(lati,long)[0]]
		SMAP_monthly = np.load(SMAP_File)[5:8,smapxy(lati,long)[1],smapxy(lati,long)[0]].mean()
		
	#%%======================================================================= Plotting the Whole timeseries

		if ('SCAN' in tables) or ('SNOTEL' in tables):
			sum_df1[tables] = sum_df1[tables]/100.00
			sum_df2[tables] = sum_df2[tables]/100.00
			sum_df3[tables] = sum_df3[tables]/100.00
			sum_df4[tables] = sum_df4[tables]/100.00
			sum_df5[tables] = sum_df5[tables]/100.00

		#Drop the rows where observation element is missing.
		sum_df1 = sum_df1.dropna(subset=[tables])
		sum_df2 = sum_df2.dropna(subset=[tables])    
		sum_df3 = sum_df3.dropna(subset=[tables])
		sum_df4 = sum_df4.dropna(subset=[tables])
		sum_df5 = sum_df5.dropna(subset=[tables])
		
		# Extarct fro June July and August month
		sum_df1_JUNE = sum_df1[sum_df1.Date.str[5:7] == '06']
		sum_df2_JUNE = sum_df2[sum_df2.Date.str[5:7] == '06']
		sum_df3_JUNE = sum_df3[sum_df3.Date.str[5:7] == '06']
		sum_df4_JUNE = sum_df4[sum_df4.Date.str[5:7] == '06']
		sum_df5_JUNE = sum_df5[sum_df5.Date.str[5:7] == '06']    

		sum_df1_JULY = sum_df1[sum_df1.Date.str[5:7] == '07']
		sum_df2_JULY = sum_df2[sum_df2.Date.str[5:7] == '07']
		sum_df3_JULY = sum_df3[sum_df3.Date.str[5:7] == '07']
		sum_df4_JULY = sum_df4[sum_df4.Date.str[5:7] == '07']
		sum_df5_JULY = sum_df5[sum_df5.Date.str[5:7] == '07']  

		sum_df1_AUG = sum_df1[sum_df1.Date.str[5:7] == '08']
		sum_df2_AUG = sum_df2[sum_df2.Date.str[5:7] == '08']
		sum_df3_AUG = sum_df3[sum_df3.Date.str[5:7] == '08']
		sum_df4_AUG = sum_df4[sum_df4.Date.str[5:7] == '08']
		sum_df5_AUG = sum_df5[sum_df5.Date.str[5:7] == '08']  


		sum_df1_Mon = pd.concat([sum_df1_JUNE,sum_df1_JULY,sum_df1_AUG])
		sum_df2_Mon = pd.concat([sum_df2_JUNE,sum_df2_JULY,sum_df2_AUG])
		sum_df3_Mon = pd.concat([sum_df3_JUNE,sum_df3_JULY,sum_df3_AUG])
		sum_df4_Mon = pd.concat([sum_df4_JUNE,sum_df4_JULY,sum_df4_AUG])
		sum_df5_Mon = pd.concat([sum_df5_JUNE,sum_df5_JULY,sum_df5_AUG])  



		# sum_df1_Mon = sum_df1.copy()
		# sum_df2_Mon = sum_df2.copy()
		# sum_df3_Mon = sum_df3.copy()
		# sum_df4_Mon = sum_df4.copy()
		# sum_df5_Mon = sum_df5.copy()   
		
		sum_df01_ind = sum_df1_Mon[['Date'] + [TABLE_Both + '_ctrl_L1'] + [TABLE_Both + '_SMAP_L1'] + [TABLE_Both + '_SMAPdailyCVEX_L1']] 	
		sum_df02_ind = sum_df1_Mon[['Date'] + [TABLE_Both + '_ctrl_L2'] + [TABLE_Both + '_SMAP_L2'] + [TABLE_Both + '_SMAPdailyCVEX_L2']] 
		
		sum_df1_ind = sum_df1_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L3'] + [TABLE_Both + '_SMAP_L3'] + [TABLE_Both + '_SMAPdailyCVEX_L3']]    
		sum_df2_ind = sum_df2_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L4'] + [TABLE_Both + '_SMAP_L4'] + [TABLE_Both + '_SMAPdailyCVEX_L4']]    
		sum_df3_ind = sum_df3_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L5'] + [TABLE_Both + '_SMAP_L5'] + [TABLE_Both + '_SMAPdailyCVEX_L5']]    
		sum_df4_ind = sum_df4_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L6'] + [TABLE_Both + '_SMAP_L6'] + [TABLE_Both + '_SMAPdailyCVEX_L6']]    
		sum_df5_ind = sum_df5_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L7'] + [TABLE_Both + '_SMAP_L7'] + [TABLE_Both + '_SMAPdailyCVEX_L7']]    
		sum_df6_ind = sum_df5_Mon[['Date'] + [TABLE_Both0] + [TABLE_Both + '_ctrl_L8'] + [TABLE_Both + '_SMAP_L8'] + [TABLE_Both + '_SMAPdailyCVEX_L8']] 
		
		array_scan = [sum_df1_ind.mean(0)[0], sum_df2_ind.mean(0)[0], sum_df3_ind.mean(0)[0], sum_df4_ind.mean(0)[0], sum_df5_ind.mean(0)[0]]

		array_ctrl = [sum_df01_ind.mean(0)[0], sum_df02_ind.mean(0)[0], sum_df1_ind.mean(0)[1], sum_df2_ind.mean(0)[1], sum_df3_ind.mean(0)[1], sum_df4_ind.mean(0)[1], sum_df5_ind.mean(0)[1], sum_df6_ind.mean(0)[1]]
		array_smap = [sum_df01_ind.mean(0)[1], sum_df02_ind.mean(0)[1], sum_df1_ind.mean(0)[2], sum_df2_ind.mean(0)[2], sum_df3_ind.mean(0)[2], sum_df4_ind.mean(0)[2], sum_df5_ind.mean(0)[2], sum_df6_ind.mean(0)[2]]

		array_smapDailyCVEX = [sum_df01_ind.mean(0)[2], sum_df02_ind.mean(0)[2], sum_df1_ind.mean(0)[3], sum_df2_ind.mean(0)[3], sum_df3_ind.mean(0)[3], sum_df4_ind.mean(0)[3], sum_df5_ind.mean(0)[3], sum_df6_ind.mean(0)[3]]

		
		plt.scatter(SMAP_monthly,0.025 ,s = 200 , c='black',marker='*', facecolors='black',label='SMAP')

		if 'SCAN' in tables:	
			plt.scatter(array_scan, array((0.05,0.1,0.2,0.5,1))  ,s = 20 , c='black',marker='s', facecolors='black',label='SCAN')
		elif 'SNOTEL' in tables:	
			plt.scatter(array_scan, array((0.05,0.1,0.2,0.5,1))  ,s = 20 , c='black',marker='s', facecolors='black',label='SNOTEL')
		else:
			plt.scatter(array_scan, array((0.05,0.1,0.2,0.5,1))  ,s = 20 , c='black',marker='s', facecolors='black',label='USCRN')

		plt.plot(array_scan, array((0.05,0.1,0.2,0.5,1)),c='black', linestyle =':', linewidth = 2)
		
		plt.scatter(array_ctrl, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038))  ,s = 20 , c='tomato',marker='^', facecolors='tomato',label='CLM_ctrl')
		plt.plot(array_ctrl, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038)),c='tomato', linestyle =':', linewidth = 1.5)

		plt.scatter(array_smap, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038))  ,s = 20 , c='dodgerblue',marker='o', facecolors='dodgerblue',label='CLM SMAP_raw')
		plt.plot(array_smap, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038)),c='dodgerblue', linestyle =':', linewidth = 1.5) 

		plt.scatter(array_smapDailyCVEX, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038))  ,s = 20 , c='forestgreen',marker='h', facecolors='forestgreen',label='CLM SMAP_kf')
		plt.plot(array_smapDailyCVEX, array((0.0071, 0.0279, 0.0623, 0.1189, 0.2122, 0.3661, 0.6198, 1.038)),c='forestgreen', linestyle =':', linewidth = 1.5) 	

		plt.gca().invert_yaxis()
		plt.xlim(0,max(SMAP_monthly,max(array_scan),max(array_ctrl),max(array_smap),max(array_smapDailyCVEX))+0.05)
		plt.ylim(1.1,-0.1)
		plt.tick_params(axis = 'both', which = 'major', labelsize = 10)
		# IrrPercentage = salmon_data[cyl5minxy(lati,long)[1],cyl5minxy(lati,long)[0]]/area[cyl5minxy(lati,long)[1],cyl5minxy(lati,long)[0]]*100
		
		plt.title(TABLE_Both, fontsize=10)

		plt.ylabel('Soil Depth (m)', fontsize=10, color='k')
			
		plt.xlabel('Soil Moisture Content ' + r'($\mathrm{mm^3/mm^3}$)', fontsize=10, color='k')

	  
		savefig( figDIR + 'FigS3_Ave_SM_vertical_X021_X024_X024kf_JJA_' + sDate_V + '_' + eDate_V + '_' + TABLE_Both + '.png', bbox_inches='tight', dpi=100 )
	# savefig( figDIR + 'FigS3_Ave_SM_vertical_X021_X024_X024kf_JJA_' + sDate_V + '_' + eDate_V + '_v1.pdf', bbox_inches='tight')

		plt.close()

