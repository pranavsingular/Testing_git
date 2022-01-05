# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 05:10:43 2021

@author: Administrator
"""




" COVID MOBILITY PREDICTIVE MODELLING "


print("----------------------------------------------------------------------------------------------------------------")
print(" Covid mobility predictive modelling in progress...............................")

import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import joblib
import array as arr
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import plot_partial_dependence
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import seaborn as sns
from sklearn.svm import SVR
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
import joblib
from sklearn.preprocessing import PolynomialFeatures




" Enter dates according to the data "


Hist_end_date = '2021-10-01'
Covid_start_date = '2020-01-01'

" Start date for COVID modelling considered to be few months before the actual COVID start date "
Covid_model_start_date = '2019-10-01' 

Covid_start_month = 3
Covid_start_year = 2020

Qtr_Range = range(1,5)
Year_Range = range(2018,2024)

Prev_Year_Range = range(2018,2023)



DATAF_FILTER = pd.read_excel("china covid.xlsx")



OUTPUT_VAR_LIST = ['TRAFFIC_WEIGHT']
REG_LIST = DATAF_FILTER.UL_GEO_ID.unique()




s = DATAF_FILTER.shape[0]

" Creating new input features "


TIME_COVID = np.zeros((s), dtype=float)

for i in range(s):
    YEAR = DATAF_FILTER.YEAR.iloc[i]
    MONTH = DATAF_FILTER.MONTH.iloc[i]
    
    num_months = (YEAR - Covid_start_year) * 12 + (MONTH - Covid_start_month)
    
    if num_months<0:
        num_months=0
        
        
    TIME_COVID[i] = num_months
    
DATAF_FILTER.insert(3, "TIME_COVID", TIME_COVID)




DATAF_MOBILITY = DATAF_FILTER[(DATAF_FILTER['PERIOD_BEGIN_DATE']>=Covid_model_start_date)] 

DATAF_MOBILITY_HIST = DATAF_MOBILITY[(DATAF_MOBILITY['PERIOD_BEGIN_DATE']<=Hist_end_date)] 

DATAF_MOBILITY_PROJ = DATAF_MOBILITY[(DATAF_MOBILITY['PERIOD_BEGIN_DATE']>Hist_end_date)] 


"--------------------------------------------------------------------------------------------------------------------------"
#" AUTO ARIMA "
#
#DATAF = DATAF_MOBILITY[(DATAF_MOBILITY.UL_GEO_ID==28)]
#
#DATAF = DATAF.reset_index()
#
#DATAF = DATAF.drop([0])
#
#Input_cols = DATAF[['UL_GEO_ID','TIME_COVID', 'NEW_CASES_BASELINE','NEW_DEATHS_BASELINE']]
#
#Output_cols = DATAF['AVG(RETAIL_AND_RECREATION_PCT_CHANGE)']
#
#x_tr = Input_cols.iloc[:14]
#y_tr = Output_cols.iloc[:14]
#x_te = Input_cols.iloc[14:15]
#y_te = Output_cols.iloc[14:15]
#
#
#x_tr.reset_index(drop=True,inplace =True)
#y_tr.reset_index(drop=True,inplace =True)
#x_te.reset_index(drop=True,inplace =True)
#y_te.reset_index(drop=True,inplace =True)
#
#import pmdarima as pm
#from pmdarima import model_selection
#import numpy as np
#from matplotlib import pyplot as plt
#
#arima = pm.auto_arima(y_tr,X=x_tr,max_p=5, max_d=5, max_q=5,max_P=5, max_D=5, max_Q=5, n_fits=50)
#
#y_pred_train = arima.predict(n_periods=y_tr.shape[0], X=x_tr)
#
#y_pred_train_series = pd.Series(y_pred_train)
#
#y_pred_test = arima.predict(n_periods=y_te.shape[0], X=x_te)
#
#y_pred_test_series = pd.Series(y_pred_test)
#
#df_act = pd.DataFrame(y_tr.append(y_te,ignore_index = True))
#df_pred = pd.DataFrame(y_pred_train_series.append(y_pred_test_series,ignore_index = True))
#
#plt.rcParams["figure.figsize"] = (20,3)
#plt.plot(df_act.values,color='blue')
#plt.plot(df_pred.values,color='red')
#





"-------------------------------------------------------------------------------------------------------------------------"
" REGRESSION APPROACH"


for j in OUTPUT_VAR_LIST:
    for i in REG_LIST: 
        print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
        print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
        
        print("OUTPUT VARIABLE ",j)
        
        DATAF = DATAF_MOBILITY_HIST[(DATAF_MOBILITY_HIST.UL_GEO_ID==i)]
        
        #print("REGION ",DATAF.UL_REGION_NAME.iloc[0])
        
        print("REGION ID",DATAF.UL_GEO_ID.iloc[0])
        
        DATAF = DATAF.reset_index()
        
        #DATAF = DATAF.drop([0])
        
        DATAF = DATAF.sample(frac=1)
            
        INP_COL = ['NEW_CASES_BASELINE','TIME_COVID','TRAFFIC_WEIGHT_trend','TRAFFIC_WEIGHT_seasonality']
        #
        ##OUT_COL = ['AVG(RESIDENTIAL_PCT_CHANGE)']
        OUT_COL = [j]
        #
        #
        X = DATAF[INP_COL].values
        Y = DATAF[OUT_COL].values 
        #
        ##"Input Data Scaling"
        ##    
        ##Data_X_Max = X.max(axis=0)
        ##Data_X_Min = X.min(axis=0)
        ##
        ##Data_X_Max = np.array(Data_X_Max)
        ##Data_X_Min = np.array(Data_X_Min)
        ##
        ##X = (X - Data_X_Min) / (Data_X_Max - Data_X_Min)       
        #
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2, stratify = None)
        
                    
                
        modelfwd = XGBRegressor(monotone_constraints = "(0,0)",max_depth=2)
        modelfwd = modelfwd.fit(X,Y)
        
        " Saving model file "
        
        filename = 'COVID_Mobility_Prediction_'+j+"_REGION_"+str(i)
        joblib.dump(modelfwd, filename)
         

        " Model validation "
        
        YPred_Train = modelfwd.predict(X_train) 
            
        RSQ_Train_FWD = np.round(r2_score(Y_train, YPred_Train),2)
        
        RMSE_Train_FWD = np.round(np.sqrt(mean_squared_error(Y_train, YPred_Train)),2)  
        
        YPred_Test = modelfwd.predict(X_test)
            
        RSQ_Test_FWD = np.round(r2_score(Y_test, YPred_Test),2)
        
        RMSE_Test_FWD = np.round(np.sqrt(mean_squared_error(Y_test, YPred_Test)),2) 
        
        YPred_Full = modelfwd.predict(X) 
        
        print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
        print("Model training and validation metrics ")
        
        print("RMSE Train FWD:",RMSE_Train_FWD)
        
        print("RMSE Test FWD:",RMSE_Test_FWD)
        
        print("RSQ Train FWD:",RSQ_Train_FWD)
        
        print("RSQ Test FWD:",RSQ_Test_FWD)
        
        print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
        print("Predicting Mobility ")
        
        DATAF = DATAF.sort_index()
        
        X = DATAF[INP_COL].values
        Y = DATAF[OUT_COL].values
        
        YPred = modelfwd.predict(X) 
        
        DATAF.insert(2, "MOBILITY_PRED", YPred)   
        
        
         
        
def COVID_Mobility_Prediction(DATAF_ALL_HIST_PROJ):
  
    " Filtering Historic & Forecast Period "
    DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['PERIOD_BEGIN_DATE']<=Hist_end_date)]
    DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['PERIOD_BEGIN_DATE']>Hist_end_date)]
    
    " Forecasting "
    
    REG_LIST = DATAF_ALL_PROJ.UL_GEO_ID.unique()
    
    for i in REG_LIST:
      
      DATAF = DATAF_ALL_PROJ[(DATAF_ALL_PROJ.UL_GEO_ID==i)]
      
      "RETAIL_AND_RECREATION_PCT_CHANGE PREDICTION"
      
      filename = 'COVID_Mobility_Prediction_'+'TRAFFIC_WEIGHT'+"_REGION_"+str(i)
      
      #load the model from disk
      loaded_model = joblib.load(filename)
      '''
      ['UL_GEO_ID', 'PERIOD_BEGIN_DATE', 'YEAR', 'TIME_COVID', 'MONTH',
       'NEW_CASES_BASELINE', 'NEW_CASES_CONSUMER_BOOM', 'NEW_CASES_LONG_COVID',
       'NEW_DEATHS_BASELINE', 'NEW_DEATHS_CONSUMER_BOOM',
       'NEW_DEATHS_LONG_COVID', 'TRAFFIC_WEIGHT']
      
      '''
      INP_COL = ['NEW_CASES_LONG_COVID','TIME_COVID','TRAFFIC_WEIGHT_trend','TRAFFIC_WEIGHT_seasonality']
      print(INP_COL)
      
      X = DATAF[INP_COL].values
      
      Y_PRED = loaded_model.predict(X)
      
      DATAF['TRAFFIC_WEIGHT'] = Y_PRED
      
    return DATAF


historic_data = DATAF_FILTER[(DATAF_FILTER['PERIOD_BEGIN_DATE']<=Hist_end_date)]
mobility = COVID_Mobility_Prediction(DATAF_FILTER)
mobility_data = pd.concat([historic_data,mobility])

mobility_data.to_csv("china_traffic_mobility_long_covid.csv")