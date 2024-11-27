from __future__ import division
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import rioxarray as rxr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import ast
import netCDF4 as nc
from pyhdf.SD import SD, SDC 
from sklearn.linear_model import LinearRegression 
import fiona
from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import BallTree   

from base import *

    
class _File_delay_Error(Exception):
    def __init__(self, message="data delay exceed the max days"):
        super().__init__(message)   


class _File_not_enough_Error(Exception):
    def __init__(self, message="required data are not enough"):
        super().__init__(message)   


class _Invdisttree:
    def __init__(self, X, z, leafsize=10, stat=0):
        assert len(X) == len(z), "len(Xhan) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None

    def __call__( self, q, nnear=4, eps=0, p=2, weights=None ):
        '''
        - nnear: nearest neighbours of each query point
        - p: control weights of each neighbour point (1/disr**p)
        '''
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:  ## to prevent infinity weight
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]


class Preprocess_data:
    '''
        process all variables to predefined grid.

        - grid_obj: class object from define_grid.py
        - start_day: e.g., 2023-01-01
        - end_day: e.g., 2023-12-31
        - fire_type: GFAS, QFED, or GFED
    '''
    process_independent_dir_name = ['GEOS', 'MODIS', 'CAMS', "GC", "HMS"]
    process_dependent_dir_name = "AirQuality"

    def __init__(self, output_base_dir, grid_obj, fire_type, start_day, end_day) -> None:
        self.grid_obj = grid_obj
        self.output_base_dir = output_base_dir
        self.fire_type = fire_type
        self.start_day = start_day
        self.end_day = end_day
        self.date_list = pd.date_range(start_day, end_day)
        self.year_start = pd.to_datetime(start_day).year
        self.year_end = pd.to_datetime(end_day).year
        self.doy_start = pd.to_datetime(start_day).dayofyear
        self.doy_end = pd.to_datetime(end_day).dayofyear
        if self.year_start != self.year_end:  
            self.year_list = list(range(self.year_start, self.year_end+1))
        else:
            self.year_list = [self.year_start, ]
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)  
            
    def bilinear_imputation(self, lon, lat, data):
        '''
        Parameters: 
            One-dimension lon, lat, data
        Returns:
            One-dimension imputation outputs
        '''
        known_points = np.array([lon, lat, data]).T
        data_interp = griddata((known_points[:,0], known_points[:,1]), known_points[:,2], 
                               (self.grid_obj.model_grid['lon'], self.grid_obj.model_grid['lat']), method='linear')
        return data_interp
    
    def idw_imputation(self, lon, lat, data, nnear=4, eps=0, p=2):
        known_points = np.array([lon, lat, data]).T
        idw_tree = _Invdisttree(known_points[:, :2], known_points[:, 2])
        data_interp = idw_tree(np.vstack((self.grid_obj.model_grid['lon'], self.grid_obj.model_grid['lat'])).T, nnear, eps, p)
        return data_interp
    
    def process_OpenAQ(self, dir_name):
        '''
        include OpenAQ and CNEMC data
        '''
        print('-'*20 + 'start processing Airquality data' + '-'*20)
        start_time = time.time()
        output_dir = os.path.join(self.output_base_dir, 'AirQuality')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for date in self.date_list:
            if os.path.exists(os.path.join(output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv")):
                continue
            print(f"processing: {date.strftime('%Y%m%d')}")
            date_local = date
            year_local = pd.to_datetime(date_local).year
            doy_local = pd.to_datetime(date_local).dayofyear
            ## 1. CNEMC
            file_path = f"{TAP_CNEMC_raw_dir}/{year_local}/{date_local.strftime('%Y%m%d')}.csv"
            if year_local >=2021: ## daily data
                try:
                    data_cn = pd.read_csv(file_path, skiprows=1)
                    data_cn = data_cn.loc[:, ['经度', '纬度', 'pm25']] 
                    data_cn.columns = ['lon', 'lat', 'pm25']
                    data_cn = data_cn.dropna()
                    data_cn = data_cn[data_cn['pm25']>=0]
                    data_cn = data_cn[data_cn['pm25']<2000] # remove pm25>2000
                    data_cn['year'] = year_local
                    data_cn['doy'] = doy_local
                    data_cn = data_cn[['lon', 'lat', 'year', 'doy', 'pm25']]
                    has_China = True
                except:
                    print(f"CNEMC has no data on {date_local.strftime('%Y%m%d')}")
                    has_China = False
            else: ## hourly data
                try:
                    data_cn = pd.read_csv(file_path, skiprows=1)
                    data_cn = data_cn.loc[:, ['站点ID', '经度', '纬度', '数据时间', 'pm25']] 
                    data_cn.columns = ['ID', 'lon', 'lat', 'time', 'pm25']
                    data_cn = data_cn.dropna()
                    data_cn = data_cn[data_cn['pm25']>=0]
                    data_cn = data_cn[data_cn['pm25']<2000] # remove pm25>2000
                    print(f"before filtering (>=16h), there are {len(np.unique(data_cn['ID']))} sites")
                    df_count = pd.DataFrame({'ID': data_cn['ID']}).value_counts().reset_index(name='count')
                    selected_site = df_count[df_count['count']>=16]['ID']  ## select the sites with daily observations over 16
                    data_cn = data_cn[data_cn['ID'].isin(selected_site)]
                    print(f"after filtering (>=16h), there are {len(np.unique(data_cn['ID']))} sites")
                    print(f"before filtering (only contain one unique value), there are {len(np.unique(data_cn['ID']))} sites")
                    df_day_int = data_cn.copy(deep=True)
                    df_day_int['pm25'] = df_day_int['pm25'].astype(int)
                    location_ids_unique_pm = df_day_int.groupby('ID')['pm25'].nunique()
                    location_ids_unique_pm = location_ids_unique_pm[location_ids_unique_pm <= 3].index
                    location_ids_mean_pm = df_day_int.groupby('ID')['pm25'].mean()
                    location_ids_mean_pm = location_ids_mean_pm[location_ids_mean_pm >= 100].index
                    if len(location_ids_unique_pm)>0:
                        data_cn = data_cn[~(data_cn['ID'].isin(location_ids_unique_pm) & data_cn['ID'].isin(location_ids_mean_pm))]
                    print(f"after filtering (only contain one unique value), there are {len(np.unique(data_cn['ID']))} sites")
                    if data_cn.shape[0]==0:
                        print(f"CNEMC has no data on {date_local.strftime('%Y%m%d')} after filtering")
                        raise FileNotFoundError   
                    data_cn = data_cn.groupby(['ID', 'lon', 'lat'])['pm25'].mean().reset_index()
                    data_cn['year'] = year_local
                    data_cn['doy'] = doy_local
                    data_cn = data_cn[['lon', 'lat', 'year', 'doy', 'pm25']]
                    has_China = True
                except:
                    print(f"CNEMC has no data on {date_local.strftime('%Y%m%d')}")
                    has_China = False                    
            ## 2. OpenAQ
            extended_start_date = pd.to_datetime(date) - pd.Timedelta(days=1)
            extended_end_date = pd.to_datetime(date) + pd.Timedelta(days=1)
            extended_date_list = pd.date_range(start=extended_start_date, end=extended_end_date)
            df_day = []
            for date in extended_date_list:
                year = pd.to_datetime(date).year
                month = pd.to_datetime(date).month
                day = pd.to_datetime(date).day
                for hour in range(0, 24):
                    try:
                        file_path = f"{TAP_OpenAQ_raw_dir}/{year}/{month:02d}/{day:02d}/{hour:02d}/item_pm25.csv"
                        data = pd.read_csv(file_path)
                    except FileNotFoundError:
                        print(f'lack OpenAQ data: file not found for {file_path}')
                        continue
                    data = data[(data['entity'] != "Person") & (data['sensorType'] != "low-cost sensor")] # exclude low-quality data
                    data['time'] = pd.to_datetime(data['date'].str[47:66])
                    data['year'] = data['time'].dt.year
                    data['doy'] = data['time'].dt.dayofyear
                    data['hour'] = data['time'].dt.hour
                    data['lon'] = data['coordinates'].apply(lambda x: ast.literal_eval(x).get('longitude')).astype(float)
                    data['lat'] = data['coordinates'].apply(lambda x: ast.literal_eval(x).get('latitude')).astype(float)
                    data = data[data['value']>=0]
                    data = data[data['value']<2000] # remove pm25>2000
                    data = data[data['value'].notna()]
                    if data.shape[0]>0:
                        df_day.append(data)
            try:
                df_day = pd.concat(df_day, ignore_index=True)
                df_day = df_day[(df_day['doy']==doy_local) & (df_day['year']==year_local)]
                print(f"before filtering (>=16h), there are {len(np.unique(df_day['locationId']))} sites")
                df_count = pd.DataFrame({'locationId': df_day['locationId']}).value_counts().reset_index(name='count')
                selected_site = df_count[df_count['count']>=16]['locationId']  ## select the sites with daily observations over 16
                df_day = df_day[df_day['locationId'].isin(selected_site)]
                print(f"after filtering (>=16h), there are {len(np.unique(df_day['locationId']))} sites")
                print(f"before filtering (only contain one unique value), there are {len(np.unique(df_day['locationId']))} sites")
                df_day_int = df_day.copy(deep=True)
                df_day_int['value'] = df_day_int['value'].astype(int)
                location_ids_unique_pm = df_day_int.groupby('locationId')['value'].nunique()
                location_ids_unique_pm = location_ids_unique_pm[location_ids_unique_pm <= 3].index
                location_ids_mean_pm = df_day_int.groupby('locationId')['value'].mean()
                location_ids_mean_pm = location_ids_mean_pm[location_ids_mean_pm >= 100].index
                if len(location_ids_unique_pm)>0:
                    df_day = df_day[~(df_day['locationId'].isin(location_ids_unique_pm) & df_day['locationId'].isin(location_ids_mean_pm))]
                print(f"after filtering (only contain one unique value), there are {len(np.unique(df_day['locationId']))} sites")
                df_day.rename(columns={'value': 'pm25'}, inplace=True)
                if df_day.shape[0]==0:
                    print(f"OpenAQ has no data on {date_local.strftime('%Y%m%d')} after filtering")
                    raise FileNotFoundError                
                df_day = df_day[['lon', 'lat', 'year', 'doy', 'pm25']]
                df_day = df_day.groupby(['lon', 'lat', 'year', 'doy'])['pm25'].mean().reset_index()
                if has_China:
                    df_day = pd.concat([df_day, data_cn], ignore_index=True)               
                df_day.to_csv(os.path.join(output_dir, f"AirQuality_{date_local.strftime('%Y%m%d')}_raw" + ".csv"), index=False)
                df_day['row'] = self.grid_obj.lat_to_row(df_day['lat'])
                df_day['col'] = self.grid_obj.lon_to_col(df_day['lon'])
                df_day = df_day.groupby(['row', 'col', 'year', 'doy'])['pm25'].mean().reset_index()
                df_day = pd.merge(df_day, self.grid_obj.model_grid[['row', 'col']], on=['row', 'col'])
                df_day.to_csv(os.path.join(output_dir, f"AirQuality_{date_local.strftime('%Y%m%d')}" + ".csv"), index=False)
                has_OpenAQ = True
            except:
                print(f"OpenAQ has no data on {date_local.strftime('%Y%m%d')}") # no data files
                has_OpenAQ = False
                if has_China:
                    df_day = data_cn
                    df_day.to_csv(os.path.join(output_dir, f"AirQuality_{date_local.strftime('%Y%m%d')}_raw" + ".csv"), index=False)
                    df_day['row'] = self.grid_obj.lat_to_row(df_day['lat'])
                    df_day['col'] = self.grid_obj.lon_to_col(df_day['lon'])
                    df_day = df_day.groupby(['row', 'col', 'year', 'doy'])['pm25'].mean().reset_index()
                    df_day = pd.merge(df_day, self.grid_obj.model_grid[['row', 'col']], on=['row', 'col'])
                    df_day.to_csv(os.path.join(output_dir, f"AirQuality_{date_local.strftime('%Y%m%d')}" + ".csv"), index=False)    
                else:
                    print(f"no data for both OpenAQ and CNEMC on {date_local.strftime('%Y%m%d')}")          
        end_time = time.time()
        print(f"Airquality processing time: {(end_time - start_time)/60} minutes")
        print('-'*20 + 'end processing Airquality data' + '-'*20)

    def process_GC(self, dir_name):
        """
        2*2.5 res
        fire_type: GFAS; QFED; GFED
        """
        print('-'*20 + 'start processing GC data' + '-'*20)
        if self.fire_type=='GFAS':
            GC_totalpm_output_dir = TAP_GC_GFAS_raw_dir
        elif self.fire_type=='GFED':
            GC_totalpm_output_dir = TAP_GC_GFED_raw_dir
        else:
            GC_totalpm_output_dir = TAP_GC_QFED_raw_dir
        totalpm_output_dir = os.path.join(self.output_base_dir, dir_name, self.fire_type)
        if not os.path.exists(totalpm_output_dir):
            os.makedirs(totalpm_output_dir)
        firepm_output_dir = os.path.join(self.output_base_dir, dir_name, 'nofire')
        if not os.path.exists(firepm_output_dir):
            os.makedirs(firepm_output_dir)
        start_time = time.time()
        for date in self.date_list:
            if os.path.exists(os.path.join(totalpm_output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv")) and (os.path.exists(os.path.join(firepm_output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv"))):
                continue         
            print(f"processing: {date.strftime('%Y%m%d')}")             
            year = pd.to_datetime(date).year
            doy = pd.to_datetime(date).dayofyear
            ## 1. total pm2.5
            file_path = os.path.join(GC_totalpm_output_dir, f"GEOSChem.AerosolMass.{date.strftime('%Y%m%d')}_0000z.nc4")
            try:
                nc_file = nc.Dataset(file_path, 'r') 
                lon = nc_file.variables['lon'][:]
                lon = np.append(lon, 180)
                lat = nc_file.variables['lat'][:]
                data = nc_file.variables['PM25'][:,0,:,:] # surface layer
                data = np.mean(data, axis=0) # daily mean
                data = np.ma.concatenate((data, data[:, 0].reshape(-1, 1)), axis=1)
                nc_file.close()
                lon, lat = np.meshgrid(lon, lat)
                data_interp_BL = self.bilinear_imputation(list(lon.flatten()), list(lat.flatten()), list(data.flatten()))
                data_interp_IDW = self.idw_imputation(list(lon.flatten()), list(lat.flatten()), list(data.flatten()))
                data_interp = (data_interp_BL+data_interp_IDW)/2
                fire_colname = f'pm25_GC_{self.fire_type}'
                df = pd.DataFrame({'row': self.grid_obj.model_grid['row'], 'col': self.grid_obj.model_grid['col'], 
                                'year': year, 'doy':doy, **{fire_colname: data_interp}})
                df = df.dropna()
                df.to_csv(os.path.join(totalpm_output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv"), index=False)
            except:
                print(file_path + ' not found')
                continue                
            ## 2. nofire pm2.5
            file_path = os.path.join(TAP_GC_nofire_raw_dir, f"GEOSChem.AerosolMass.{date.strftime('%Y%m%d')}_0000z.nc4")
            try:
                nc_file = nc.Dataset(file_path, 'r') 
                lon = nc_file.variables['lon'][:]
                lon = np.append(lon, 180)
                lat = nc_file.variables['lat'][:]
                data = nc_file.variables['PM25'][:,0,:,:] # surface layer
                data = np.mean(data, axis=0) # daily mean
                data = np.ma.concatenate((data, data[:, 0].reshape(-1, 1)), axis=1)
                nc_file.close()
                lon, lat = np.meshgrid(lon, lat)
                data_interp_BL = self.bilinear_imputation(list(lon.flatten()), list(lat.flatten()), list(data.flatten()))
                data_interp_IDW = self.idw_imputation(list(lon.flatten()), list(lat.flatten()), list(data.flatten()))
                data_interp = (data_interp_BL+data_interp_IDW)/2
                df = pd.DataFrame({'row': self.grid_obj.model_grid['row'], 'col': self.grid_obj.model_grid['col'], 
                                'year': year, 'doy':doy, 'pm25_GC_nofire': data_interp})
                df = df.dropna()
                df.to_csv(os.path.join(firepm_output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv"), index=False)     
            except:
                print(file_path + ' not found')
                continue                
        end_time = time.time()
        print(f"GC processing time: {(end_time - start_time)/60} minutes")
        print('-'*20 + 'end processing GC data' + '-'*20)
    
    def process_CAMS(self, dir_name):
        '''
        include two different products:
        - air composition product:
            Black carbon aerosol optical depth at 550 nm, Organic matter aerosol optical depth at 550 nm, 
            Particulate matter d < 2.5 µm (PM2.5), Total aerosol optical depth at 550 nm, Total column carbon monoxide,
            0.4*0.4 res
        - fire product:    
            wildfire radiative power (FRP), wildfire combustion rate  
            0.1*0.1 res
        '''
        print('-'*20 + 'start processing CAMS data' + '-'*20)
        start_time = time.time()
        output_dir = os.path.join(self.output_base_dir, dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for date in self.date_list:
            year = pd.to_datetime(date).year
            ## 1. cams-global-atmospheric-composition-forecasts
            if os.path.exists(os.path.join(output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv")):
                continue
            print(f"processing: {date.strftime('%Y%m%d')}")
            year = pd.to_datetime(date).year
            doy = pd.to_datetime(date).dayofyear
            try:
                file_path = os.path.join(TAP_CAMS_raw_dir, 'cams-global-atmospheric-composition-forecasts', str(year), f"{date.strftime('%Y%m%d')}.nc")
                nc_file = nc.Dataset(file_path, 'r')  
                lon = nc_file.variables['longitude'][:]
                lon[lon>=180] = lon[lon>=180] - 360
                index = int(len(lon)/2)
                lon = np.insert(lon, index, 180)
                lat = nc_file.variables['latitude'][:]
                omaod550 = nc_file.variables['omaod550'][0, 0, :,:]
                omaod550 = np.insert(omaod550, index, omaod550[:, index], axis=1)
                pm2p5 = nc_file.variables['pm2p5'][0, 0, :,:]
                pm2p5 = np.insert(pm2p5, index, pm2p5[:, index], axis=1)
                aod550 = nc_file.variables['aod550'][0, 0, :,:]
                aod550 = np.insert(aod550, index, aod550[:, index], axis=1)
                tcco = nc_file.variables['tcco'][0, 0, :,:]
                tcco = np.insert(tcco, index, tcco[:, index], axis=1)
                bcaod550 = nc_file.variables['bcaod550'][0, 0, :,:]
                bcaod550 = np.insert(bcaod550, index, bcaod550[:, index], axis=1)
                lon, lat = np.meshgrid(lon, lat)
                omaod550 = self.idw_imputation(list(lon.flatten()), list(lat.flatten()), list(omaod550.flatten()))
                pm2p5 = self.idw_imputation(list(lon.flatten()), list(lat.flatten()), list(pm2p5.flatten()))
                aod550 = self.idw_imputation(list(lon.flatten()), list(lat.flatten()), list(aod550.flatten()))
                tcco = self.idw_imputation(list(lon.flatten()), list(lat.flatten()), list(tcco.flatten()))
                bcaod550 = self.idw_imputation(list(lon.flatten()), list(lat.flatten()), list(bcaod550.flatten()))
                nc_file.close()
            except:
                print(file_path + ' not found')    
                continue
            ## 2. cams-global-fire-emissions-gfas
            try:
                file_path = os.path.join(TAP_CAMS_raw_dir, 'cams-global-fire-emissions-gfas', str(year), f"{date.strftime('%Y%m%d')}.nc")
                nc_file = nc.Dataset(file_path, 'r')
                lon = nc_file.variables['longitude'][:]
                lon[lon>180] = lon[lon>180] - 360
                lat = nc_file.variables['latitude'][:]
                lon, lat = np.meshgrid(lon, lat)
                CombustionRate = nc_file.variables['crfire'][:]
                unique, counts = np.unique(CombustionRate, return_counts=True)
                modal_value = unique[np.argmax(counts)]
                CombustionRate[CombustionRate == modal_value] = 0
                RadiatPower = nc_file.variables['frpfire'][:]        
                unique, counts = np.unique(RadiatPower, return_counts=True)
                modal_value = unique[np.argmax(counts)]
                RadiatPower[RadiatPower == modal_value] = 0     
                CombustionRate = self.idw_imputation(list(lon.flatten()), list(lat.flatten()), list(CombustionRate.flatten()))
                RadiatPower = self.idw_imputation(list(lon.flatten()), list(lat.flatten()), list(RadiatPower.flatten()))            
                df = pd.DataFrame({'row': self.grid_obj.model_grid['row'], 'col': self.grid_obj.model_grid['col'],
                                'year': year, 'doy': doy, 'omaod550': omaod550, 'pm2p5': pm2p5,
                                'aod550': aod550,'tcco': tcco, 'caod550': bcaod550, 'CombustionRate': CombustionRate, 'RadiatPower': RadiatPower})
                df = df.dropna()
                df.to_csv(os.path.join(output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv"), index=False)
                nc_file.close()
            except:
                print(file_path + ' not found')
        end_time = time.time()
        print(f"CAMS processing time: {(end_time - start_time)/60} minutes")
        print('-'*20 + 'end processing CAMS data' + '-'*20)
    
    def process_GEOS(self, dir_name):
        """
            0.25° lat x 0.3125° lon
        """
        print('-'*20 + 'start processing GEOS-FP data' + '-'*20)
        start_time = time.time()
        output_dir = os.path.join(self.output_base_dir, dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        vars = ['Lon', 'Lat', "QV2m", "PBLH", "T2m", "U10m", "V10m", 
                "Wind10", "EFLUX", "EVAP", "TO3", "PRECTOT", "PS"]
        for date in self.date_list:
            if os.path.exists(os.path.join(output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv")):
                continue
            print(f"processing: {date.strftime('%Y%m%d')}")
            year = pd.to_datetime(date).year
            doy = pd.to_datetime(date).dayofyear
            file_path = os.path.join(TAP_GEOSFP_raw_dir, str(year), f'Global_GEOS5_FP_{year}_{doy:03d}_daily.csv')
            try:
                df = pd.read_csv(file_path)
                df_edge = df[df['Lon']==-180]
                df_edge.loc[:, 'Lon'] = 180
                df = pd.concat([df, df_edge], ignore_index=True)
            except:
                print(file_path + ' not found')
                continue
            data_dict = {}
            data_dict['row'] = self.grid_obj.model_grid['row']
            data_dict['col'] = self.grid_obj.model_grid['col']
            for var in vars[2:]:
                data = self.idw_imputation(list(df['Lon']), list(df['Lat']), list(df[var]))
                data_dict[var] = data
            df = pd.DataFrame(data_dict)
            df = df.dropna()
            df.columns = ['row', 'col', "Humidity_2m", "PBLH", "Temp_2m", "Wind_U_10m", "Wind_V_10m", 
                        "Wind_10m", "Eflux", "Evap", "TO3", "prectot", "Surface_pressure"]
            df.loc[:,'year'] = year
            df.loc[:,'doy'] = doy
            # print(f"there are {df.shape[0]} samples, full grid samples are {self.grid_obj.model_grid.shape[0]}")
            df.to_csv(os.path.join(output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv"), index=False)
        end_time = time.time()
        print(f"GEOS processing time: {(end_time - start_time)/60} minutes")
        print('-'*20 + 'end processing GEOS-FP data' + '-'*20)
    
    def process_HMS(self, dir_name):
        print('-'*20 + 'start processing HMS data' + '-'*20)
        start_time = time.time()
        output_dir = os.path.join(self.output_base_dir, dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for date in self.date_list:
            if os.path.exists(os.path.join(output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv")):
                continue
            print(f"processing: {date.strftime('%Y%m%d')}")
            year = pd.to_datetime(date).year
            doy = pd.to_datetime(date).dayofyear
            file_path = os.path.join(TAP_HMS_raw_dir, f"{year}/hms_smoke{date.strftime('%Y%m%d')}.shp")
            try:
                smoke_shp = gpd.read_file(file_path)
            except:
                print(file_path + ' not found')
                # df = pd.DataFrame({'row': self.grid_obj.model_grid['row'], 'col': self.grid_obj.model_grid['col'], 'HMS_Density': 0, 'year': year, 'doy': doy})
                # df.to_csv(os.path.join(output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv"), index=False)
                continue          
            df_points = self.grid_obj.model_grid
            gdf_points = gpd.GeoDataFrame(df_points, geometry=gpd.points_from_xy(df_points['lon'], df_points['lat']))
            gdf_points.crs = smoke_shp.crs
            points_in_poly = gpd.sjoin(gdf_points, smoke_shp, how='left', predicate='within')
            points_in_poly.loc[points_in_poly['Density']=='Light', 'Density'] = 1
            points_in_poly.loc[points_in_poly['Density']=='Medium', 'Density'] = 2
            points_in_poly.loc[points_in_poly['Density']=='Heavy', 'Density'] = 3
            points_in_poly['Density'] = points_in_poly['Density'].astype('float')
            points_in_poly.loc[np.isnan(points_in_poly['Density']), 'Density'] = 0
            points_in_poly['Density'] = points_in_poly['Density'].astype('int')
            df = points_in_poly[['row', 'col', 'Density']]
            df.columns = ['row', 'col', 'HMS_Density']
            df = df.groupby(['row','col'])['HMS_Density'].max().reset_index()
            df.loc[:,'year'] = year
            df.loc[:,'doy'] = doy
            df.to_csv(os.path.join(output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv"), index=False)
        end_time = time.time()
        print(f"HMS processing time: {(end_time - start_time)/60} minutes")
        print('-'*20 + 'end processing HMS data' + '-'*20)      
    
    def process_AOD(self, dir_name):
        '''
            process MOD_04 and MYD_04 5min L2 data
            10 km
        '''
        print('-'*20 + 'start processing AOD data' + '-'*20)
        start_time = time.time()
        output_dir = os.path.join(self.output_base_dir, dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        Aqua_file_dir = os.path.join(TAP_MODIS_raw_dir, 'Aqua')
        Aqua_file_list = []
        for year in self.year_list:
            if os.path.exists(os.path.join(Aqua_file_dir, str(year))):
                file_list_year = os.listdir(os.path.join(Aqua_file_dir, str(year)))
                Aqua_file_list.extend(file_list_year)
            else:
                continue
        Terra_file_dir = os.path.join(TAP_MODIS_raw_dir, 'Terra')
        Terra_file_list = []
        for year in self.year_list:
            if os.path.exists(os.path.join(Terra_file_dir, str(year))):
                file_list_year = os.listdir(os.path.join(Terra_file_dir, str(year)))
                Terra_file_list.extend(file_list_year)
            else:
                continue
        for date in self.date_list:
            if os.path.exists(os.path.join(output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv")):
                continue
            print(f"processing: {date.strftime('%Y%m%d')}")
            year = pd.to_datetime(date).year
            doy = pd.to_datetime(date).dayofyear
            file_list_Aqua = [file for file in Aqua_file_list if f"A{date.strftime('%Y%j')}" in file and file.endswith('hdf')]
            file_list_Terra = [file for file in Terra_file_list if f"A{date.strftime('%Y%j')}" in file and file.endswith('hdf')]
            if len(file_list_Aqua) == 0:
                print(f"Aqua data are not found on {date.strftime('%Y%m%d')}")
            else:
                print(f"Aqua has {len(file_list_Aqua)} data files on {date.strftime('%Y%m%d')}")
                df_Aqua_day = self.read_daily_aod(file_dir=os.path.join(Aqua_file_dir, str(year)), file_list=file_list_Aqua)
                df_Aqua_day = self.linear_regression_imputation(df_Aqua_day, 'db_aod', 'dt_aod', 'Aqua_aod')[['row', 'col', 'Aqua_aod']]
            if len(file_list_Terra) == 0:
                print(f"Terra data are not found on {date.strftime('%Y%m%d')}")
            else:
                print(f"Terra has {len(file_list_Terra)} data files on {date.strftime('%Y%m%d')}")                
                df_Terra_day = self.read_daily_aod(file_dir=os.path.join(Terra_file_dir, str(year)), file_list=file_list_Terra)
                df_Terra_day = self.linear_regression_imputation(df_Terra_day, 'db_aod', 'dt_aod', 'Terra_aod')[['row', 'col', 'Terra_aod']]
            if len(file_list_Aqua) == 0 and len(file_list_Terra) == 0:
                print(f"no data found on {date.strftime('%Y%m%d')} for both Terra and Aqua")
                continue
            elif len(file_list_Aqua) > 0 and len(file_list_Terra) == 0:
                df_day = df_Aqua_day
                df_day = df_day.dropna()
                df_day.rename(columns={'Aqua_aod': 'aod'}, inplace=True)
                df_day.loc[df_day['aod']<-0.1, 'aod'] = -0.1 ## remove outliers
                df_day.loc[df_day['aod']>5, 'aod'] = 5  ## remove outliers
            elif len(file_list_Aqua) == 0 and len(file_list_Terra) > 0:
                df_day = df_Terra_day
                df_day = df_day.dropna()
                df_day.rename(columns={'Terra_aod': 'aod'}, inplace=True)
                df_day.loc[df_day['aod']<-0.1, 'aod'] = -0.1 ## remove outliers
                df_day.loc[df_day['aod']>5, 'aod'] = 5  ## remove outliers
            else:
                df_day = pd.merge(df_Terra_day, df_Aqua_day, on=['row', 'col'], how='outer')
                df_day = self.linear_regression_imputation(df_day, 'Terra_aod', 'Aqua_aod', 'aod')[['row', 'col', 'aod']]
                df_day = df_day.dropna()
                df_day.loc[df_day['aod']<-0.1, 'aod'] = -0.1 ## remove outliers
                df_day.loc[df_day['aod']>5, 'aod'] = 5  ## remove outliers
            df_day = df_day.astype({'row': 'int', 'col': 'int'})
            df_day = pd.merge(df_day, self.grid_obj.model_grid[['row', 'col']], on=['row', 'col'], how='outer') # outer
            df_day = self.aod_interpolation(df_day) # handle satellite obiting tracks
            df_day['year'] = year
            df_day['doy'] = doy
            df_day.to_csv(os.path.join(output_dir, dir_name + f"_{date.strftime('%Y%m%d')}" + ".csv"), index=False)
        end_time = time.time()
        print(f"AOD processing time: {(end_time - start_time)/60} minutes")
        print('-'*20 + 'end processing AOD data' + '-'*20)      
        
    def read_daily_aod(self, file_dir, file_list):
        '''
            derive inegrated AOD data from multiple seperated HDF files for each day.
            iterate through a list of HDF files, reads relevant AOD data and geographic coordinates (longitude and latitude), and applies quality filters to the data.

            Parameters:
            - file_dir: The directory where the HDF files are stored.
            - file_list: A list of HDF file names to be processed.

            Returns:
            - A Pandas DataFrame containing the mean AOD values for each grid cell, aligned with the predefined model grid in define_grid.py.
        '''
        unique_file_prefix = list(np.unique([".".join(file.split(".")[:4]) for file in file_list])) ## to prevent from reading duplicated file
        df_day = []
        for file_prefix in unique_file_prefix:
            file = [file for file in file_list if file_prefix in file][0]
            hdf_file = os.path.join(file_dir, file)
            try:
                hdfFile = SD(hdf_file)
            except Exception as e:
                print(f'{hdf_file} is broken: ', e)
                continue
            lon = hdfFile.select('Longitude')[:]
            lat = hdfFile.select('Latitude')[:]
            
            db_aod = hdfFile.select('Deep_Blue_Aerosol_Optical_Depth_550_Land')[:].astype(float)
            db_aod[db_aod==-9999] = np.nan
            db_aod = db_aod*0.0010000000474974513
            db_flag = hdfFile.select('Deep_Blue_Aerosol_Optical_Depth_550_Land_QA_Flag')[:]
            db_aod[db_flag<1] = np.nan

            dt_aod = hdfFile.select('Image_Optical_Depth_Land_And_Ocean')[:].astype(float)
            dt_aod[dt_aod==-9999] = np.nan
            dt_aod = dt_aod*0.0010000000474974513
            dt_flag = hdfFile.select('Land_Ocean_Quality_Flag')[:]
            dt_aod[dt_flag<1] = np.nan

            df = pd.DataFrame({'lon':lon.flatten(), 'lat':lat.flatten(), 'db_aod': db_aod.flatten(), 'dt_aod': dt_aod.flatten()})
            df = df[(df['lon']>=-180) & (df['lon']<=180) & (df['lat']<=90) & (df['lat']>=-90)]
            df['row'] = self.grid_obj.lat_to_row(df['lat'])
            df['col'] = self.grid_obj.lon_to_col(df['lon'])
            df_day.append(df)
            hdfFile.end()
        df_day = pd.concat(df_day, ignore_index=True)
        df_day = df_day.groupby(['row', "col"])[['db_aod','dt_aod']].mean().reset_index()
        df_day = df_day.dropna(subset=['db_aod', 'dt_aod'], how='all').reset_index(drop=True)
        df_day = pd.merge(df_day, self.grid_obj.model_grid[['row', 'col']], on=['row', 'col'])
        return df_day
    
    def linear_regression_imputation(self, dat, x_col, y_col, new_col):
        '''
            Perform linear regression imputation on a given DataFrame for specified columns and create a new column with their mean values.

            Parameters:
            - dat: DataFrame to be processed.
            - x_col: The name of the column to be used as the feature for linear regression.
            - y_col: The name of the column to be used as the target variable for linear regression.
            - new_col: The name for the new column to be created, which will contain the mean of x_col and y_col.

            The function first checks if there are more than 500 non-missing collocated observations for both x_col and y_col. 
            If so, it build two linear regression models: one with x_col as the feature and y_col as the target, and the other with the roles reversed. 
            It then predicts and fills missing values in each column based on the two regression models.

            After handling missing values or if the pair count is 500 or less, it calculates the mean of x_col and y_col for each row (ignoring NA values) and stores this in the new column specified by new_col.

            Returns:
            - The DataFrame with imputed values and the new column added.
        '''
        if dat.loc[dat[x_col].notna() & dat[y_col].notna()].shape[0] > dat.shape[0]*0.2:  ## prevent the circumstance that either Terra or Aqua has few data on a given day
            X_train = dat.loc[dat[x_col].notna() & dat[y_col].notna(), [x_col]]
            y_train = dat.loc[dat[x_col].notna() & dat[y_col].notna(), y_col]
            try:
                model = LinearRegression().fit(X_train, y_train)
                r_squared = model.score(X_train, y_train)
                print(f"R2: {r_squared}")
                X_predict = dat.loc[dat[y_col].isna() & dat[x_col].notna(), [x_col]]
                dat.loc[dat[y_col].isna() & dat[x_col].notna(), y_col] = model.predict(X_predict)
                print("Linear regression success") 
            except Exception as e:
                print("Linear regression failed: ", e)
            X_train = dat.loc[dat[x_col].notna() & dat[y_col].notna(), [y_col]]
            y_train = dat.loc[dat[x_col].notna() & dat[y_col].notna(), x_col]
            try:
                model = LinearRegression().fit(X_train, y_train)
                r_squared = model.score(X_train, y_train)
                print(f"R2: {r_squared}")
                X_predict = dat.loc[dat[x_col].isna() & dat[y_col].notna(), [y_col]]
                dat.loc[dat[x_col].isna() & dat[y_col].notna(), x_col] = model.predict(X_predict)
                print("Linear regression success") 
            except Exception as e:
                print("Linear regression failed: ", e)                
            dat[new_col] = dat[[x_col, y_col]].mean(axis=1, skipna=True)
        else:
            print('not enough data for Linear regression')
            dat[new_col] = dat[[x_col, y_col]].mean(axis=1, skipna=True)
        return dat
    
    def aod_interpolation(self, data):
        '''
        - with limited nearest interpolation
        '''
        df = data.copy()
        df_withoutAOD = df[df['aod'].isna()].reset_index(drop=True)
        df_withAOD = df[df['aod'].notna()].reset_index(drop=True)
        tree = BallTree(df_withAOD[['row', 'col']], metric='chebyshev')
        dist, idx = tree.query(df_withoutAOD[['row', 'col']], k=24)
        df_index = pd.DataFrame({'dist': dist[:, 0], 'idx': idx[:, 0], 'near_number': np.sum(dist <= 2, axis=1)})
        df_index['aod'] = df_withAOD.loc[df_index['idx'], 'aod'].values
        df_withoutAOD = pd.concat([df_withoutAOD.drop(columns=['aod']), df_index], axis=1)
        df_withoutAOD.loc[(df_withoutAOD['near_number'] < 8), 'aod'] = np.nan
        df_fill = pd.concat([df_withAOD, df_withoutAOD], ignore_index=True)    
        df_fill = df_fill.drop(columns=['dist', 'idx', 'near_number'])
        return df_fill

    def process_pop(self):
        df = pd.read_csv(TAP_pop_raw_path)
        df['row'] = self.grid_obj.lat_to_row(df['lat'])
        df['col'] = self.grid_obj.lon_to_col(df['lon'])
        df = df[['row', 'col', 'WP_2020']]
        df.columns = ['row', 'col', 'pop']
        # df = pd.merge(df, self.grid_obj.model_grid[['row', 'col']], on=['row', 'col'])
        df = pd.merge(df, self.grid_obj.model_grid[['row', 'col']], on=['row', 'col'], how='outer')
        df['pop'] = df['pop'].fillna(0)
        df.to_csv(os.path.join(self.output_base_dir, 'pop.csv'), index=False)
    
    def deriving_daily_training_data(self, date, all_fire, current_index=0):
        process_dirname = [self.process_dependent_dir_name]+self.process_independent_dir_name
        if current_index >= len(process_dirname):
            return None
        var = process_dirname[current_index]
        try:
            df = self.read_single_day_output(var, date, all_fire)           
        except:
            print(f"lack necessary files for deriving training data on {date.strftime('%Y%m%d')}: {var}")
            raise _File_not_enough_Error()
        current_index += 1
        next_df = self.deriving_daily_training_data(date, all_fire, current_index)
        if next_df is None:
            return df
        else:
            df = pd.merge(df, next_df, on=['row', 'col', 'year', 'doy']) ## inner
            return df

    def process_training(self, dir_name, all_fire=False):
        print('-'*20 + 'start deriving training data' + '-'*20)
        start_time = time.time()
        output_dir = os.path.join(self.output_base_dir, dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for date in self.date_list:
            date = pd.to_datetime(date)
            dow = date.dayofweek
            if os.path.exists(os.path.join(output_dir, f"training_{date.strftime('%Y%m%d')}" + ".csv")):
                continue
            print(f"processing: {date.strftime('%Y%m%d')}")
            try:
                df = self.deriving_daily_training_data(date=date, all_fire=all_fire, current_index=0)
                df_pop = pd.read_csv(os.path.join(self.output_base_dir, 'pop.csv'))
                df = pd.merge(df, df_pop, on=['row', 'col'])
                df['dow'] = dow            
                df.to_csv(os.path.join(output_dir, f"training_{date.strftime('%Y%m%d')}.csv"), index=False)
            except:
                print(f"training data on {date.strftime('%Y%m%d')} cannot be derived")
                continue
        end_time = time.time()
        print(f"deriving training data time: {(end_time - start_time)/60} minutes")
        print('-'*20 + 'end deriving training data' + '-'*20)       

    def deriving_daily_prediction_data(self, date, current_index=0):
        '''
        Returns:
            - full-coverage daily prediction data
        '''
        date = pd.to_datetime(date)
        year = date.year
        doy = date.dayofyear
        process_dirname = self.process_independent_dir_name
        if current_index >= len(process_dirname):
            return None 
        var = process_dirname[current_index]
        try:
            df = self.read_single_day_output(var, date) 
        except:
            print(f"Warning: no data available for {var} on {date.strftime('%Y%m%d')}")
            ### replace by nearest processed file
            date_replace = self.find_nearest_file(var=var, date=date)
            df = self.read_single_day_output(var, date_replace) 
            df['year'] = year
            df['doy'] = doy
            print(f"replaced by {date_replace}")
        current_index += 1
        next_df = self.deriving_daily_prediction_data(date, current_index)
        if next_df is None:
            return df
        else:
            df = pd.merge(df, next_df, on=['row', 'col', 'year', 'doy'], how='outer') ## outer
            return df

    def process_prediction(self, dir_name):
        '''
            derive daily prediction data for end_day
        '''
        print('-'*20 + 'start deriving prediction data' + '-'*20)
        start_time = time.time()
        output_dir = os.path.join(self.output_base_dir, dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        date = pd.to_datetime(self.end_day)
        print(f"processing: {date.strftime('%Y%m%d')}")
        dow = date.dayofweek
        doy = date.dayofyear
        df = self.deriving_daily_prediction_data(date=date)
        df['doy'] = doy
        df['dow'] = dow
        df_pop = pd.read_csv(os.path.join(self.output_base_dir, 'pop.csv'))
        df = pd.merge(df, df_pop, on=['row', 'col'])
        df.to_csv(os.path.join(output_dir, f"prediction_{date.strftime('%Y%m%d')}.csv"), index=False)
        end_time = time.time()
        print(f"deriving prediction data time: {(end_time - start_time)/60} minutes")
        print('-'*20 + 'end deriving prediction data' + '-'*20)       
        
    def find_nearest_file(self, var, date, days_delta=200):
        if var == 'GC':
            file_path = os.path.join(self.output_base_dir, var, self.fire_type)
        else:
            file_path = os.path.join(self.output_base_dir, var)
        file_list = os.listdir(file_path)
        search_start_date = pd.to_datetime(date) - pd.Timedelta(days=days_delta)
        search_date_list = pd.date_range(start=search_start_date, end=pd.to_datetime(date)- pd.Timedelta(days=1))
        for i, search_date in enumerate(search_date_list[::-1]):
            replace_file = [file for file in file_list if search_date.strftime('%Y%m%d') in file]
            if len(replace_file)!=0:
                replace_file = replace_file[0]
                return search_date.strftime('%Y%m%d')
            else:
                if i==len(search_date_list)-1:
                    raise _File_delay_Error()
                else:
                    continue

    def read_single_day_output(self, var, date, all_fire=False):
        ''' 
        return data of single variable for single day
        '''        
        date = pd.to_datetime(date)
        if var == 'GC':
            if all_fire:
                df_GFAS = pd.read_csv(os.path.join(self.output_base_dir, var, 'GFAS', f"GC_{date.strftime('%Y%m%d')}" + ".csv"))
                df_QFED = pd.read_csv(os.path.join(self.output_base_dir, var, 'QFED', f"GC_{date.strftime('%Y%m%d')}" + ".csv"))  
                # df_GFED = pd.read_csv(os.path.join(self.output_base_dir, var, 'GFED', f"GC_{date.strftime('%Y%m%d')}" + ".csv")) 
                df_nofire = pd.read_csv(os.path.join(self.output_base_dir, var, 'nofire', f"GC_{date.strftime('%Y%m%d')}" + ".csv")) 
                # df = pd.merge(df_GFAS, df_GFED, on=['row', 'col', 'year', 'doy'])
                df = pd.merge(df_GFAS, df_QFED, on=['row', 'col', 'year', 'doy'])
                df = pd.merge(df, df_nofire, on=['row', 'col', 'year', 'doy'])
            else:
                df_fire = pd.read_csv(os.path.join(self.output_base_dir, var, self.fire_type, f"GC_{date.strftime('%Y%m%d')}" + ".csv"))
                df_nofire = pd.read_csv(os.path.join(self.output_base_dir, var, 'nofire', f"GC_{date.strftime('%Y%m%d')}" + ".csv"))
                df = pd.merge(df_fire, df_nofire, on=['row', 'col', 'year', 'doy'])
        else:
            file_path = os.path.join(self.output_base_dir, var, var + f"_{date.strftime('%Y%m%d')}" + ".csv")
            df = pd.read_csv(file_path)
        return df

    def read_daily_output(self, var):
        ''' 
        return data of single variable from start_day to end_day (used for reading training data in model object)
        please refer to model.py
        '''
        df_day = []
        for date in self.date_list:
            try:
                file_path = os.path.join(self.output_base_dir, var, var + f"_{date.strftime('%Y%m%d')}" + ".csv")
                df = pd.read_csv(file_path)
            except FileNotFoundError:
                print(f'Warning: file not found for {file_path}')
                continue
            df_day.append(df)
        df_day = pd.concat(df_day, ignore_index=True)
        return df_day