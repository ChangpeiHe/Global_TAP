import os
import time
import sys
import pandas as pd
import numpy as np
import netCDF4 as nc
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_validate, cross_val_predict, KFold
from sklearn.cluster import KMeans    
from sklearn.neighbors import BallTree       
from sklearn.linear_model import LinearRegression  
import seaborn as sns
import matplotlib.pyplot as plt
from base import *

'''
    此版本withaod用了全部的训练数据
    第二阶段的模型用了直接算出来的high-index
'''


class Tap_RF_model:
    '''
        TAP three-step Random Forest model for mapping PM2.5
    '''
    n_estimators=150
    min_samples_split=10
    n_jobs=24
    max_depth=None
    min_samples_leaf=10
    max_leaf_nodes=None
    random_state=1
    
    def __init__(self, independent_v, independent_v_name, dependent_v, output_dir, figure_dir, process_obj, is_global=True, has_HMS=True) -> None:
        self.process_obj = process_obj
        self.independent_v = independent_v
        self.dependent_v = dependent_v
        self.independent_v_name = independent_v_name
        self.independent_v_withoutAOD = [var for var in independent_v if var!='aod']
        self.independent_v_residual = [var for var in independent_v if var!='aod']
        self.is_global = is_global
        self.has_HMS = has_HMS
        training_data = process_obj.read_daily_output('training')
        if not self.is_global:
            training_data = pd.merge(training_data, self.process_obj.grid_obj.model_grid[['row', 'col']]).reset_index(drop=True)
        training_data = self.preprocess_training_dataset(training_data)
        self.training_dataset = training_data
        # self.training_dataset = training_data.sample(frac=0.3, random_state=1).reset_index(drop=True)
        
        # build model
        training_dataset_withAOD = self.training_dataset[self.training_dataset['aod'].notna()]
        training_dataset_withoutAOD = self.training_dataset.drop(columns=['aod'])
        self.model_high_withAOD, self.model_pm_withAOD, df_withAOD_residual = self.build_base_model(training_dataset_withAOD, self.independent_v, self.dependent_v)
        self.model_high_withoutAOD, self.model_pm_withoutAOD, df_withoutAOD_residual = self.build_base_model(training_dataset_withoutAOD, self.independent_v_withoutAOD, self.dependent_v)
        df_residual = pd.merge(df_withoutAOD_residual, df_withAOD_residual[['row', 'col', 'year', 'doy', 'pm_residual']], on=['row', 'col', 'year', 'doy'], how='outer')
        df_residual['pm_residual'] = df_residual['pm_residual_y'].combine_first(df_residual['pm_residual_x'])
        df_residual = df_residual.drop(columns=['pm_residual_x', 'pm_residual_y'])
        self.model_pm_residual_withfire, self.model_pm_residual_withoutfire = self.build_residual_model(df_residual, self.independent_v_residual, 'pm_residual')
        
        del training_dataset_withAOD
        del training_dataset_withoutAOD
        del df_withAOD_residual
        del df_withoutAOD_residual
        del df_residual
                
        self.output_dir = output_dir
        self.figure_dir = figure_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir)
        
    def preprocess_training_dataset(self, data):
        data['date'] = pd.to_datetime(data['year'].astype(str) + data['doy'].astype(str), format='%Y%j')
        data['month'] = data['date'].dt.month
        data['dow'] = data['date'].dt.dayofweek
        data = self.data_outlier_filter(data)   
        return data.reset_index(drop=True)
    
    def data_outlier_filter(self, data):
        '''
            remove pm25 observation outliers
        '''
        print(f'original samples are {data.shape[0]}')
        data = data[~((data['aod'] < 0.1) & (data['pm25'] > 100))]
        data = data[~((data['aod'] < 0.2) & (data['pm25'] > 200))]
        data = data[~((data['aod'] < 0.3) & (data['pm25'] > 300))]
        data = data[~((data['aod'] < 0.4) & (data['pm25'] > 400))]
        data = data[~((data['aod'] < 0.5) & (data['pm25'] > 500))]
        data = data[data['pm25'] < 2000]
        print(f'after filtering, samples are {data.shape[0]}')
        return data
    
    def label_high(self, data):
        '''
        define high pollution envent and add a column to label whether there is high pollution for each grid and each day
            - for each month, > m+2*sd
            - HMS_Density > 0
        '''
        avg = data.groupby(['year', 'month', 'row', 'col'])['pm25'].mean().reset_index(name='avg_pm25')
        sd = data.groupby(['year', 'month', 'row', 'col'])['pm25'].std().reset_index(name='sd_pm25')
        data = pd.merge(data, avg, on=['year', 'month', 'row', 'col'])
        data = pd.merge(data, sd, on=['year', 'month', 'row', 'col'])
        data['high'] = 0
        data.loc[data['pm25'] > (data['avg_pm25'] + 2*data['sd_pm25']), 'high'] = 1 # > mean+2*sd
        data.loc[data['pm25'] < 35, 'high'] = 0  
        if self.has_HMS:
            data.loc[data['HMS_Density'] > 0, 'high'] = 1 # HMS_Density > 0
        data = data.drop(['sd_pm25', 'avg_pm25'], axis=1)
        return data
    
    def smoke(self, data):
        '''
        To balance the training data set, use smote to oversampling
        '''
        low_num = len(data.loc[data['high']==0, 'high'])
        high_num = len(data.loc[data['high']==1, 'high'])
        object_num = low_num//5
        if high_num < object_num:
            smote = SMOTE(sampling_strategy={1:int(object_num)}, random_state=self.random_state, k_neighbors=5)
            x, y = data.drop('high', axis=1), data['high']
            print(f"before Smoke:{Counter(y)}")
            x_smo, y_smo = smote.fit_resample(x, y)
            print(f"after Smoke:{Counter(y_smo)}")
            data = pd.concat([x_smo, y_smo], axis=1)
            return data.reset_index(drop=True)
        else:
            return data
    
    def build_base_model(self, data, independent_v, dependent_v):
        ## high-index model and first-layer pm2.5 model
        pm25_GC = [var for var in independent_v if 'pm25_GC' in var][0]
        data = self.label_high(data)
        df_train = data[independent_v + [dependent_v] + ['high']]
        df_train_smoke = self.smoke(df_train)
        model_high = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split, random_state=self.random_state, n_jobs=self.n_jobs, max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes)
        print(f"{len(df_train_smoke['high'])} samples are trained in high model")
        model_high.fit(df_train_smoke[independent_v], df_train_smoke['high'])
        model_pm = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split, random_state=self.random_state, n_jobs=self.n_jobs, max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes)
        X_train, y_train = df_train[independent_v + ['high']], df_train[dependent_v]
        print(f"{len(y_train)} samples are trained in pm model")
        model_pm.fit(X_train, y_train)
        data[dependent_v] = model_pm.predict(X_train)
        ## use GC output to replace TAP output if GC > TAP（not use）
        # if self.has_HMS:
        #     index = (data[dependent_v] < data[pm25_GC]) & ((data['HMS_Density'] > 0) | (data['CombustionRate'] > 0))
        # else:
        #     index = (data[dependent_v] < data[pm25_GC]) & (data['CombustionRate'] > 0)
        # data.loc[index, dependent_v] = data.loc[index, pm25_GC]
        data['pm_residual'] = y_train - data[dependent_v]
        data['fire'] = 0
        if self.has_HMS:
            data.loc[(df_train['CombustionRate']>0) | (data['HMS_Density']>0), 'fire'] = 1  
        else:
            data.loc[df_train['CombustionRate']>0, 'fire'] = 1       
        return model_high, model_pm, data
    
    def build_residual_model(self, data, independent_v, dependent_v):
        # second-layer pm2.5 residual model (with and without fire)
        df_withfire = data[data['fire']==1]
        X_train, y_train = df_withfire[independent_v], df_withfire[dependent_v]
        model_pm_residual_withfire = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split, random_state=self.random_state, n_jobs=self.n_jobs, max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes)
        print(f"{len(y_train)} samples are trained in pm_residual_withfire model")
        model_pm_residual_withfire.fit(X_train, y_train)
        df_withoutfire = data[data['fire']==0]
        X_train, y_train = df_withoutfire[independent_v], df_withoutfire[dependent_v]
        model_pm_residual_withoutfire = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split, random_state=self.random_state, n_jobs=self.n_jobs, max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes)
        print(f"{len(y_train)} samples are trained in pm_residual_withoutfire model")
        model_pm_residual_withoutfire.fit(X_train, y_train)
        return model_pm_residual_withfire, model_pm_residual_withoutfire
    
    def build_benchmark_base_model(self, data, independent_v, dependent_v):
        model_pm = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split, random_state=self.random_state, n_jobs=self.n_jobs, max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes)
        X_train, y_train = data[independent_v], data[dependent_v]
        print(f'{len(y_train)} samples are trained in benchmark model')
        model_pm.fit(X_train, y_train)   
        return model_pm

    def predict(self, data, model_high_withAOD, model_high_withoutAOD, model_pm_withAOD, model_pm_withoutAOD, model_pm_residual_withfire, model_pm_residual_withoutfire):
        if self.dependent_v in data.columns:
            data = data.drop(columns=[self.dependent_v])
        pm25_GC = [var for var in self.independent_v if 'pm25_GC' in var][0]
        data_withAOD = data[data['aod'].notna()]
        data_withoutAOD = data[data['aod'].isna()]
        data_withoutAOD = data_withoutAOD.drop(columns=['aod'])
        if data_withAOD.shape[0]>0:
            data_withAOD['fire'] = 0
            if self.has_HMS:
                data_withAOD.loc[(data_withAOD['CombustionRate']>0) | (data_withAOD['HMS_Density']>0), 'fire'] = 1  
            else:
                data_withAOD.loc[data_withAOD['CombustionRate']>0, 'fire'] = 1  
            data_withoutAOD['fire'] = 0
            if self.has_HMS:
                data_withoutAOD.loc[(data_withoutAOD['CombustionRate']>0) | (data_withoutAOD['HMS_Density']>0), 'fire'] = 1   
            else:
                data_withoutAOD.loc[data_withoutAOD['CombustionRate']>0, 'fire'] = 1   
            # step1: derive high pollution index
            data_withAOD['high'] = model_high_withAOD.predict(data_withAOD[self.independent_v])
            data_withoutAOD['high'] = model_high_withoutAOD.predict(data_withoutAOD[self.independent_v_withoutAOD])
            # step2: predict first-layer pm2.5
            data_withAOD[self.dependent_v] = model_pm_withAOD.predict(data_withAOD[self.independent_v+['high']])
            data_withoutAOD[self.dependent_v] = model_pm_withoutAOD.predict(data_withoutAOD[self.independent_v_withoutAOD+['high']])
            ## use GC output to replace TAP output if GC > TAP（not use）
            # if self.has_HMS:
            #     withAOD_index = (data_withAOD[self.dependent_v] < data_withAOD[pm25_GC]) & ((data_withAOD['HMS_Density'] > 0) | (data_withAOD['CombustionRate'] > 0))
            # else:
            #     withAOD_index = (data_withAOD[self.dependent_v] < data_withAOD[pm25_GC]) & (data_withAOD['CombustionRate'] > 0)     
            # data_withAOD.loc[withAOD_index, self.dependent_v] = data_withAOD.loc[withAOD_index, pm25_GC]
            # if self.has_HMS:
            #     withoutAOD_index = (data_withoutAOD[self.dependent_v] < data_withoutAOD[pm25_GC]) & ((data_withoutAOD['HMS_Density'] > 0) | (data_withoutAOD['CombustionRate'] > 0))
            # else:
            #     withoutAOD_index = (data_withoutAOD[self.dependent_v] < data_withoutAOD[pm25_GC]) & (data_withoutAOD['CombustionRate'] > 0)
            # data_withoutAOD.loc[withoutAOD_index, self.dependent_v] = data_withoutAOD.loc[withoutAOD_index, pm25_GC]    
            df = pd.concat([data_withAOD, data_withoutAOD], ignore_index=True)
        else:
            data_withoutAOD['fire'] = 0
            if self.has_HMS:
                data_withoutAOD.loc[(data_withoutAOD['CombustionRate']>0) | (data_withoutAOD['HMS_Density']>0), 'fire'] = 1   
            else:
                data_withoutAOD.loc[data_withoutAOD['CombustionRate']>0, 'fire'] = 1           
            data_withoutAOD['high'] = model_high_withoutAOD.predict(data_withoutAOD[self.independent_v_withoutAOD])
            data_withoutAOD[self.dependent_v] = model_pm_withoutAOD.predict(data_withoutAOD[self.independent_v_withoutAOD+['high']])
            ## use GC output to replace TAP output if GC > TAP（not use）
            # if self.has_HMS:
            #     withoutAOD_index = (data_withoutAOD[self.dependent_v] < data_withoutAOD[pm25_GC]) & ((data_withoutAOD['HMS_Density'] > 0) | (data_withoutAOD['CombustionRate'] > 0))
            # else:
            #     withoutAOD_index = (data_withoutAOD[self.dependent_v] < data_withoutAOD[pm25_GC]) & (data_withoutAOD['CombustionRate'] > 0)
            # data_withoutAOD.loc[withoutAOD_index, self.dependent_v] = data_withoutAOD.loc[withoutAOD_index, pm25_GC]    
            df = data_withoutAOD   
        # step3: predict second-layer pm2.5 residuals
        data_withfire = df[df['fire']==1]
        data_withoutfire = df[df['fire']==0]
        if data_withfire.shape[0] > 0:
            data_withfire['pm_residual'] = model_pm_residual_withfire.predict(data_withfire[self.independent_v_residual])
        data_withoutfire['pm_residual'] = model_pm_residual_withoutfire.predict(data_withoutfire[self.independent_v_residual])
        # step4: retrieve ultimate pm2.5
        if data_withfire.shape[0] > 0:
            data_withfire[self.dependent_v] = data_withfire[self.dependent_v]+data_withfire['pm_residual']
        data_withoutfire[self.dependent_v] = data_withoutfire[self.dependent_v]+data_withoutfire['pm_residual']
        if data_withfire.shape[0] > 0:
            df = pd.concat([data_withfire, data_withoutfire], ignore_index=True)
        else:
            df = data_withoutfire
        data = pd.merge(data, df[['row', 'col', 'year', 'doy', self.dependent_v]], on=['row', 'col', 'year', 'doy'])
        return data

    def predict_NRT(self, date, prediction_type):
        ''' 
            derive daily pm2.5 predictions
            - prediction_type: web or publish. 
                if web, prediction data will be read from 'prediction_web' dir
                if publish, prediction data will be read from 'prediction' dir
        '''
        date = pd.to_datetime(date)
        pm25_GC = [var for var in self.independent_v if 'pm25_GC' in var][0]
        if prediction_type=='web':
            prediction_path = os.path.join(self.process_obj.output_base_dir, 'prediction_web', f"prediction_{date.strftime('%Y%m%d')}" + ".csv")
        else:
            prediction_path = os.path.join(self.process_obj.output_base_dir, 'prediction', f"prediction_{date.strftime('%Y%m%d')}" + ".csv")
        prediction_data = pd.read_csv(prediction_path)
        if not self.is_global:
            prediction_data = pd.merge(prediction_data, self.process_obj.grid_obj.model_grid[['row', 'col']], on=['row', 'col'])
        prediction_data = self.predict(prediction_data, self.model_high_withAOD, self.model_high_withoutAOD, self.model_pm_withAOD, self.model_pm_withoutAOD, self.model_pm_residual_withfire, self.model_pm_residual_withoutfire)
        # use GC simulated pm2.5 to fill greenland pm2.5 (for global model only)
        if self.process_obj.grid_obj.greenland_grid is not None:
            greenland_grid = self.process_obj.grid_obj.greenland_grid[['row', 'col']]
            greenland_grid.loc[:, 'is_greenland'] = 1
            prediction_data = pd.merge(prediction_data, greenland_grid, on=['row', 'col'], how='left')
            prediction_data.loc[~(prediction_data['is_greenland'].isna()), self.dependent_v] = prediction_data.loc[~(prediction_data['is_greenland'].isna()), pm25_GC]
        prediction_data = prediction_data[['row', 'col', 'year', 'doy', self.dependent_v]]
        return prediction_data 

    def sep_pm25(self, date):
        output_dir = os.path.join(self.output_dir, self.process_obj.fire_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pm25_GC = [var for var in self.independent_v if 'pm25_GC' in var][0]
        df_TAP_totalpm = pd.read_csv(os.path.join(output_dir, f"totalpm_{date.strftime('%Y%m%d')}" + ".csv"))
        df_GC_totalpm = pd.read_csv(os.path.join(os.path.join(self.process_obj.output_base_dir, 'GC'), self.process_obj.fire_type, f"GC_{date.strftime('%Y%m%d')}" + ".csv"))
        df_GC_nofirepm = pd.read_csv(os.path.join(os.path.join(self.process_obj.output_base_dir, 'GC'), 'nofire', f"GC_{date.strftime('%Y%m%d')}" + ".csv"))
        df = pd.merge(df_GC_totalpm, df_GC_nofirepm[['row', 'col', 'pm25_GC_nofire']], on=['row', 'col'])
        df['ratio'] = (df[pm25_GC]-df['pm25_GC_nofire'])/df[pm25_GC]
        ## set fire ratio < 0 as 0
        print(f"GC_fire_ratio < 0 number: {(df['ratio'] < 0).sum()}")
        df_ratio = df[['row', 'col', 'ratio']]
        df_ratio.loc[df_ratio['ratio']<0, 'ratio'] = 0
        df = pd.merge(df_TAP_totalpm, df_ratio, on=['row', 'col'])
        df[self.dependent_v] = df[self.dependent_v]*df['ratio']
        df = df[['row', 'col', 'year', 'doy', self.dependent_v]]
        return df        
        
    def sample_CV_TAP(self, n_fold):
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=self.random_state)
        y_ = self.training_dataset[self.dependent_v].values
        predictions = np.zeros_like(y_, dtype=float)        
        for train_index, test_index in kf.split(self.training_dataset):
            # build model
            df_train, df_test = self.training_dataset.iloc[train_index], self.training_dataset.iloc[test_index]
            training_dataset_withAOD = df_train[df_train['aod'].notna()]
            training_dataset_withoutAOD = df_train.drop(columns=['aod'])
            model_high_withAOD, model_pm_withAOD, df_withAOD_residual = self.build_base_model(training_dataset_withAOD, self.independent_v, self.dependent_v)
            model_high_withoutAOD, model_pm_withoutAOD, df_withoutAOD_residual = self.build_base_model(training_dataset_withoutAOD, self.independent_v_withoutAOD, self.dependent_v)
            df_residual = pd.merge(df_withoutAOD_residual, df_withAOD_residual[['row', 'col', 'year', 'doy', 'pm_residual']], on=['row', 'col', 'year', 'doy'], how='outer')
            df_residual['pm_residual'] = df_residual['pm_residual_y'].combine_first(df_residual['pm_residual_x'])
            df_residual = df_residual.drop(columns=['pm_residual_x', 'pm_residual_y'])            
            model_pm_residual_withfire, model_pm_residual_withoutfire = self.build_residual_model(df_residual, self.independent_v_residual, 'pm_residual')
            # predict
            data = self.predict(df_test, model_high_withAOD, model_high_withoutAOD, model_pm_withAOD, model_pm_withoutAOD, model_pm_residual_withfire, model_pm_residual_withoutfire)
            predictions[test_index] = data[self.dependent_v].values
        CV_df = self.training_dataset.copy(deep=True)
        CV_df['pm25_pre'] = predictions
        return CV_df
    
    def spatial_CV_TAP(self, n_fold):
        self.training_dataset['GridID'] = self.training_dataset['row'].astype(str) + '_' + self.training_dataset['col'].astype(str)
        GridID_list = np.unique(self.training_dataset['GridID'])
        np.random.shuffle(GridID_list) # randomly shuffle the GridID
        fold_grid_size = len(GridID_list) // n_fold
        y_ = self.training_dataset[self.dependent_v].values
        predictions = np.zeros_like(y_, dtype=float)        
        for ifold in range(n_fold):
            if ifold == n_fold-1:
                grid_list_fold = GridID_list[ifold*fold_grid_size:]
            else:
                grid_list_fold = GridID_list[ifold*fold_grid_size: (ifold+1)*fold_grid_size]
            # build model
            test_index = self.training_dataset['GridID'].isin(grid_list_fold)
            df_train = self.training_dataset[~test_index]
            df_test = self.training_dataset[test_index]
            training_dataset_withAOD = df_train[df_train['aod'].notna()]
            training_dataset_withoutAOD = df_train.drop(columns=['aod'])
            model_high_withAOD, model_pm_withAOD, df_withAOD_residual = self.build_base_model(training_dataset_withAOD, self.independent_v, self.dependent_v)
            model_high_withoutAOD, model_pm_withoutAOD, df_withoutAOD_residual = self.build_base_model(training_dataset_withoutAOD, self.independent_v_withoutAOD, self.dependent_v)
            df_residual = pd.merge(df_withoutAOD_residual, df_withAOD_residual[['row', 'col', 'year', 'doy', 'pm_residual']], on=['row', 'col', 'year', 'doy'], how='outer')
            df_residual['pm_residual'] = df_residual['pm_residual_y'].combine_first(df_residual['pm_residual_x'])
            df_residual = df_residual.drop(columns=['pm_residual_x', 'pm_residual_y'])            
            model_pm_residual_withfire, model_pm_residual_withoutfire = self.build_residual_model(df_residual, self.independent_v_residual, 'pm_residual')
            # predict
            data = self.predict(df_test, model_high_withAOD, model_high_withoutAOD, model_pm_withAOD, model_pm_withoutAOD, model_pm_residual_withfire, model_pm_residual_withoutfire)
            predictions[test_index] = data[self.dependent_v].values
        CV_df = self.training_dataset.copy(deep=True)
        CV_df['pm25_pre'] = predictions
        return CV_df    
    
    def cluster_CV_TAP(self, n_cluster):
        df_grid_matched = self.training_dataset[['row', 'col']].drop_duplicates(subset=['row', 'col']).reset_index(drop=True)
        df_grid_matched['lon'] = self.process_obj.grid_obj.col_to_lon(df_grid_matched['col'])
        df_grid_matched['lat'] = self.process_obj.grid_obj.row_to_lat(df_grid_matched['row'])
        kmeans = KMeans(n_clusters=n_cluster, n_init=25) # 75 clusters; 25 random initing
        df_grid_matched['Cluster'] = kmeans.fit_predict(df_grid_matched[['lon', 'lat']])
        df_grid_matched = pd.merge(self.training_dataset, df_grid_matched[['row', 'col', 'Cluster']], on=['row', 'col']) # training_dataset resorted by row and col
        # plt.figure(figsize=(10, 6))
        # sns.scatterplot(data=df_grid_matched, x='lon', y='lat', hue='Cluster', palette='viridis', s=100)
        # plt.savefig('./cluster.png', dpi=300)
        # plt.close()
        clusters = np.unique(df_grid_matched['Cluster'])
        y_ = df_grid_matched[self.dependent_v].values
        predictions = np.zeros_like(y_, dtype=float)        
        for icluster in clusters:
            # build model
            test_index = df_grid_matched['Cluster']==icluster
            df_train = df_grid_matched[~test_index]
            df_test = df_grid_matched[test_index]
            training_dataset_withAOD = df_train[df_train['aod'].notna()]
            training_dataset_withoutAOD = df_train.drop(columns=['aod'])
            model_high_withAOD, model_pm_withAOD, df_withAOD_residual = self.build_base_model(training_dataset_withAOD, self.independent_v, self.dependent_v)
            model_high_withoutAOD, model_pm_withoutAOD, df_withoutAOD_residual = self.build_base_model(training_dataset_withoutAOD, self.independent_v_withoutAOD, self.dependent_v)
            df_residual = pd.merge(df_withoutAOD_residual, df_withAOD_residual[['row', 'col', 'year', 'doy', 'pm_residual']], on=['row', 'col', 'year', 'doy'], how='outer')
            df_residual['pm_residual'] = df_residual['pm_residual_y'].combine_first(df_residual['pm_residual_x'])
            df_residual = df_residual.drop(columns=['pm_residual_x', 'pm_residual_y'])            
            model_pm_residual_withfire, model_pm_residual_withoutfire = self.build_residual_model(df_residual, self.independent_v_residual, 'pm_residual')
            # predict
            df_test = self.predict(df_test, model_high_withAOD, model_high_withoutAOD, model_pm_withAOD, model_pm_withoutAOD, model_pm_residual_withfire, model_pm_residual_withoutfire)
            predictions[test_index] = df_test[self.dependent_v].values
            del test_index
            del df_train
            del df_test
            del training_dataset_withAOD
            del training_dataset_withoutAOD
            del model_high_withAOD
            del model_high_withoutAOD
            del model_pm_residual_withfire
            del model_pm_residual_withoutfire
            del model_pm_withAOD
            del model_pm_withoutAOD
            del df_withAOD_residual
            del df_withoutAOD_residual
            del df_residual
        df_grid_matched['pm25_pre'] = predictions
        return df_grid_matched  
        
    def fit_performance_TAP(self):
        df_train = self.training_dataset.copy(deep=True)
        data = self.predict(df_train, self.model_high_withAOD, self.model_high_withoutAOD, self.model_pm_withAOD, self.model_pm_withoutAOD, self.model_pm_residual_withfire, self.model_pm_residual_withoutfire)
        df_train['pm25_pre'] = data[self.dependent_v]
        return df_train
    
    def CV_benchmark(self, n_fold):
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=self.random_state)
        y_ = self.training_dataset[self.dependent_v].values
        predictions = np.zeros_like(y_, dtype=float)        
        for train_index, test_index in kf.split(self.training_dataset):
            # build model
            df_train, df_test = self.training_dataset.iloc[train_index], self.training_dataset.iloc[test_index]
            training_dataset_withAOD = df_train[df_train['aod'].notna()]
            training_dataset_withoutAOD = df_train.drop(columns=['aod'])
            model_pm_withAOD = self.build_benchmark_base_model(training_dataset_withAOD, self.independent_v, self.dependent_v)
            model_pm_withoutAOD = self.build_benchmark_base_model(training_dataset_withoutAOD, self.independent_v_withoutAOD, self.dependent_v)
            # predict
            data_withAOD = df_test[df_test['aod'].notna()]
            data_withoutAOD = df_test[df_test['aod'].isna()]
            data_withoutAOD = data_withoutAOD.drop(columns=['aod'])
            data_withAOD['pm25_pre'] = model_pm_withAOD.predict(data_withAOD[self.independent_v])
            data_withoutAOD['pm25_pre'] = model_pm_withoutAOD.predict(data_withoutAOD[self.independent_v_withoutAOD])
            df = pd.concat([data_withAOD, data_withoutAOD], ignore_index=True)
            df = df[['row', 'col', 'year', 'doy', 'pm25_pre']]
            df_test = pd.merge(df_test, df, on=['row', 'col', 'year', 'doy'])
            predictions[test_index] = df_test['pm25_pre'].values
        CV_df = self.training_dataset.copy(deep=True)
        CV_df['pm25_pre'] = predictions
        return CV_df
        
    def fit_performance_Benchmark(self):
        df_train = self.training_dataset.copy(deep=True) # deep=True, self.training_dataset can not be modified
        training_dataset_withAOD = df_train[df_train['aod'].notna()]
        training_dataset_withoutAOD = df_train.drop(columns=['aod'])  
        model_pm_withAOD = self.build_benchmark_base_model(training_dataset_withAOD, self.independent_v, self.dependent_v)
        model_pm_withoutAOD = self.build_benchmark_base_model(training_dataset_withoutAOD, self.independent_v_withoutAOD, self.dependent_v)
        training_dataset_withAOD['pm25_pre'] = model_pm_withAOD.predict(training_dataset_withAOD[self.independent_v])
        training_dataset_withoutAOD['pm25_pre'] = model_pm_withoutAOD.predict(training_dataset_withoutAOD[self.independent_v_withoutAOD])   
        data = pd.merge(training_dataset_withAOD, training_dataset_withoutAOD[['row', 'col', 'year', 'doy', 'pm25_pre']], on=['row', 'col', 'year', 'doy'], how='outer')
        data['pm25_pre'] = data['pm25_pre_x'].combine_first(data['pm25_pre_y'])
        data = data.drop(columns=['pm25_pre_x', 'pm25_pre_y'])  
        data = pd.merge(df_train, data[['row', 'col', 'year', 'doy', 'pm25_pre']], on=['row', 'col', 'year', 'doy'])
        return data
    
    # def CV_benchmark_withAOD(self, n_fold):
    #     df_cv = self.training_dataset[self.training_dataset['aod'].notna()].reset_index(drop=True)         
    #     kf = KFold(n_splits=n_fold)
    #     y_ = df_cv[self.dependent_v].values
    #     predictions = np.zeros_like(y_, dtype=float)   
    #     for train_index, test_index in kf.split(df_cv):
    #         # build model
    #         df_train, df_test = df_cv.iloc[train_index], df_cv.iloc[test_index]
    #         model_pm_withAOD = self.build_benchmark_base_model(df_train, self.independent_v, self.dependent_v)
    #         # predict
    #         df_test['pm25_pre'] = model_pm_withAOD.predict(df_test[self.independent_v])
    #         predictions[test_index] = df_test['pm25_pre'].values
    #     return predictions  
      
    # def CV_benchmark_withoutAOD(self, n_fold):
    #     kf = KFold(n_splits=n_fold)
    #     y_ = self.training_dataset[self.dependent_v].values
    #     predictions = np.zeros_like(y_, dtype=float)        
    #     for train_index, test_index in kf.split(self.training_dataset):
    #         # build model
    #         df_train, df_test = self.training_dataset.iloc[train_index], self.training_dataset.iloc[test_index]
    #         training_dataset_withoutAOD = df_train.drop(columns=['aod'])
    #         model_pm_withoutAOD = self.build_benchmark_base_model(training_dataset_withoutAOD, self.independent_v_withoutAOD, self.dependent_v)
    #         # predict
    #         df_test = df_test.drop(columns=['aod'])
    #         df_test['pm25_pre'] = model_pm_withoutAOD.predict(df_test[self.independent_v_withoutAOD])
    #         predictions[test_index] = df_test['pm25_pre'].values
    #     return predictions     

                    

    
    
    
    
