


" Framework for other scenarios "

get_ipython().magic('clear')
import warnings
warnings.filterwarnings("ignore")
import os

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
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.svm import SVR


all_data_all_scenarios_for_analysis = pd.DataFrame()
all_data_all_scenarios_for_upload = pd.DataFrame()
all_shap_all_scenarios = pd.DataFrame()
all_data = pd.read_csv("china_baseline_rerun_for_seperate_online.csv")

for sub_cats in all_data.SUBCAT_CD.unique():
    
    " Run for all the scenarios 1,2,3,4,5 "
       
    
    scenario_list = [1,2,3,4,5]
    
    complete_data_for_rahul = pd.DataFrame()
    complete_data_for_sanjay = pd.DataFrame()
    
    complete_shap = pd.DataFrame()
    
    for SCENARIO_NO in scenario_list:
    
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        " Enter the directory path "
        
        path = os.getcwd()
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        print(" Data Acquisition ")
        
        " Enter the output file (prediction sales) acquired from the Total_Framework run "
        DATAF_ALL_HIST_PROJ = pd.DataFrame()
        DATAF_ALL_HIST_PROJ = all_data[all_data.SUBCAT_CD == sub_cats]
        
        
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        df = pd.read_excel('China feature selection.xlsx')
        
        df.columns = ['UL_GEO_ID', 'SUBCAT_CD', 'Features', 'Independents',
               'Dependents', 'Constraint', '13_cor', '13_ptop', '13_vif', 'Columns_to_Keep']
        
        def config_read(data = df, subcat=13, target_col='PRICE_PER_VOL'):
            '''
            This function aims to read input file from business experts and extract columns and constraint from them.
            The idea to use this function is to automate reading of business input and results in columns and constraint
            for price, tdp and sales.
            
            Columns to be used in input file:
            'UL_GEO_ID',
            'CATG_CD', 
            'SUBCAT_CD', 
            'Independents',
            'Dependents',
            'Constraint'
            'Columns_to_Keep'
            
            '''
            feat = df[(df['Independents']==target_col)&(df['SUBCAT_CD']==subcat)&(df['Columns_to_Keep']==1)]['Dependents'].to_list()
            cons = tuple(df[(df['Independents']==target_col)&(df['SUBCAT_CD']==subcat)&(df['Columns_to_Keep']==1)]['Constraint'].to_list())
            return feat, str(cons)
        
        
        feat = []
        cons = []
        out = []
        for subcat in DATAF_ALL_HIST.SUBCAT_CD.unique():
            for i in df.Independents.unique():
                
                mx = config_read(df,subcat, i)
                
                feat.append(mx[0])
                cons.append(mx[1])
                out.append(i)
            
                    
                    
            price_input_col = feat[0]
            tdp_input_col = feat[2]
            sales_input_col = feat[1]
            price_cons = cons[0]
            tdp_cons = cons[2]
            sales_cons = cons[1] 
            price_output_col = out[0]
            tdp_output_col = out[2]
            sales_output_col = out[1]
        
        
        
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        " Enter dates according to the data "
        
        Start_month = 1
        Start_year = 2019
        
        End_month = 6
        End_year = 2021
        
        
        
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        print(" Adding forecasts for Macro-economic variables ")
        
        DATAF_ALL_MACRO = pd.read_csv('Macro.csv')
        
        DATAF_ALL_MACRO = DATAF_ALL_MACRO[(DATAF_ALL_MACRO['SECTOR_SCENARIO_CD']==SCENARIO_NO)]
        
        s = DATAF_ALL_HIST_PROJ.shape[0]
        
        for i in range(s):
            YEAR = DATAF_ALL_HIST_PROJ.YEAR.iloc[i]
            MONTH = DATAF_ALL_HIST_PROJ.MONTH.iloc[i]
            
            if SCENARIO_NO==1:
                SCENARIO_NAME = 'Baseline forecast'
            elif SCENARIO_NO==2:
                SCENARIO_NAME = 'Consumer boom'
            elif SCENARIO_NO==3:
                SCENARIO_NAME = 'Consumer hesitancy'
            elif SCENARIO_NO==4:
                SCENARIO_NAME = 'Limited vaccine effectiveness'
            elif SCENARIO_NO==5:
                SCENARIO_NAME = 'Return of inflation'
            
            DATAF_ALL_HIST_PROJ['SECTOR_SCENARIO_CD'].iloc[i] = SCENARIO_NO
            DATAF_ALL_HIST_PROJ['SECTOR_SCENARIO_DESC'].iloc[i] = SCENARIO_NAME
            DATAF_MACRO_TEMP = DATAF_ALL_MACRO[(DATAF_ALL_MACRO.YEAR==YEAR) & (DATAF_ALL_MACRO.MONTH==MONTH) & (DATAF_ALL_MACRO.SECTOR_SCENARIO_CD==SCENARIO_NO)]
            DATAF_ALL_HIST_PROJ['AVG(CONSUMER_PRICE_INDEX)'].iloc[i] = DATAF_MACRO_TEMP.CONSUMER_PRICE_INDEX.iloc[0]
            DATAF_ALL_HIST_PROJ['AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)'].iloc[i] = DATAF_MACRO_TEMP.PERSONAL_DISPOSABLE_INCOME_REAL_LCU.iloc[0]
            DATAF_ALL_HIST_PROJ['AVG(GDP_NOMINAL_LCU)'].iloc[i] = DATAF_MACRO_TEMP.GDP_NOMINAL_LCU.iloc[0]
            DATAF_ALL_HIST_PROJ['AVG(GDP_REAL_LCU)'].iloc[i] = DATAF_MACRO_TEMP.GDP_REAL_LCU.iloc[0]
            DATAF_ALL_HIST_PROJ['AVG(RETAIL_PRICES_INDEX)'].iloc[i] = DATAF_MACRO_TEMP.RETAIL_PRICES_INDEX.iloc[0]
            DATAF_ALL_HIST_PROJ['AVG(SHARE_PRICE_INDEX)'].iloc[i] = DATAF_MACRO_TEMP.SHARE_PRICE_INDEX.iloc[0]
            DATAF_ALL_HIST_PROJ['AVG(UNEMP_RATE)'].iloc[i] = DATAF_MACRO_TEMP.UNEMP_RATE.iloc[0]
            
            
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        " Adding future numbers for COVID cases and deaths "
        
        DATAF_COVID_NOS = pd.read_excel('china covid.xlsx')
        
        DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
        DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]
        
        s = DATAF_ALL_PROJ.shape[0]
        
        for i in range(s):
            Year = DATAF_ALL_PROJ.YEAR.iloc[i]
            Month = DATAF_ALL_PROJ.MONTH.iloc[i]
            
            DATAF_COVID_FIL = DATAF_COVID_NOS[(DATAF_COVID_NOS['YEAR']==Year) & (DATAF_COVID_NOS['MONTH']==Month)]
            
            if SCENARIO_NO==3 or SCENARIO_NO==4:
                DATAF_ALL_PROJ['SUM(NEW_CASES)'].iloc[i] = DATAF_COVID_FIL.NEW_CASES_LONG_COVID.iloc[0]
                DATAF_ALL_PROJ['SUM(NEW_DEATHS)'].iloc[i] = DATAF_COVID_FIL.NEW_DEATHS_LONG_COVID.iloc[0]
            elif SCENARIO_NO==2:
                DATAF_ALL_PROJ['SUM(NEW_CASES)'].iloc[i] = DATAF_COVID_FIL.NEW_CASES_CONSUMER_BOOM.iloc[0]
                DATAF_ALL_PROJ['SUM(NEW_DEATHS)'].iloc[i] = DATAF_COVID_FIL.NEW_DEATHS_CONSUMER_BOOM.iloc[0]
            else:
                DATAF_ALL_PROJ['SUM(NEW_CASES)'].iloc[i] = DATAF_COVID_FIL.NEW_CASES_BASELINE.iloc[0]
                DATAF_ALL_PROJ['SUM(NEW_DEATHS)'].iloc[i] = DATAF_COVID_FIL.NEW_DEATHS_BASELINE.iloc[0]
            
          
            
        DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])  
        
        "-----------------------------------------------------------------------------------------------------------------------------"  
        
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        print(" Adding forecasts for baseline mobility variables ")
        if SCENARIO_NO==3 or SCENARIO_NO==4:
            DATAF_ALL_mobility = pd.read_csv('china_traffic_mobility_long_covid.csv')
        elif SCENARIO_NO==2:
            DATAF_ALL_mobility = pd.read_csv('china_traffic_mobility_consumer_boom.csv')
        else:
            DATAF_ALL_mobility = pd.read_csv('china_traffic_mobility_baseline.csv')
    
        
        s = DATAF_ALL_HIST_PROJ.shape[0]
        
        for i in range(s):
            YEAR = DATAF_ALL_HIST_PROJ.YEAR.iloc[i]
            MONTH = DATAF_ALL_HIST_PROJ.MONTH.iloc[i]
            
            DATAF_mobility_TEMP = DATAF_ALL_mobility[(DATAF_ALL_mobility.YEAR==YEAR) & (DATAF_ALL_mobility.MONTH==MONTH)]
            DATAF_ALL_HIST_PROJ['AVG(TRAFFIC_WEIGHT)'].iloc[i] = DATAF_mobility_TEMP["TRAFFIC_WEIGHT"].iloc[0]
            
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        " Price Forecast "
        
        " Filtering Historic & Forecast Period "
        DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
        DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]
        
        runfile(path+'/PRICE_PREDICTION_FUNCTION_TOTAL.py', wdir=path)
        
        DATAF_ALL_PROJ = Price_Prediction(DATAF_ALL_PROJ,price_input_col)
        
        DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        " Incorporating Price Assumptions for Return of inflation scenario "
        
        if SCENARIO_NO==5:
            f_name = 'roi_prices_for_'+str(DATAF_ALL_HIST_PROJ.SUBCAT_NAME.unique()[0])+'.csv'
            DATAF_ALL_PRICE_ASSUMPTIONS = pd.read_csv(f_name)
            s = DATAF_ALL_HIST_PROJ.shape[0]
            for i in range(s):
                SUBCAT = DATAF_ALL_HIST_PROJ.SUBCAT_CD.iloc[i]
                YEAR = DATAF_ALL_HIST_PROJ.YEAR.iloc[i]
                MONTH = DATAF_ALL_HIST_PROJ.MONTH.iloc[i]
                DATAF_ALL_PRICE_ASSUMPTIONS_FIL = DATAF_ALL_PRICE_ASSUMPTIONS[(DATAF_ALL_PRICE_ASSUMPTIONS.SUBCAT_CD==SUBCAT) & (DATAF_ALL_PRICE_ASSUMPTIONS.YEAR==YEAR) & (DATAF_ALL_PRICE_ASSUMPTIONS.MONTH==MONTH)]    
                DATAF_ALL_HIST_PROJ.PRICE_PER_VOL.iloc[i] = DATAF_ALL_PRICE_ASSUMPTIONS_FIL.predicted_PRICE_PER_VOL_adjusted.iloc[0]
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        " TDP Forecast "
        
        " Filtering Historic & Forecast Period "
        DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
        DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]
        
        
        runfile(path+'/TDP_PREDICTION_FUNCTION_TOTAL.py', wdir=path)
        
        DATAF_ALL_PROJ = TDP_Prediction(DATAF_ALL_PROJ,tdp_input_col)
        
        DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        
        " Adding constant column for Preference "
        
        " Preference value not applicable for total level modelling. Hence, adding constant value of 1 "
        
        #DATAF_ALL_HIST_PROJ.insert(DATAF_ALL_HIST_PROJ.columns.get_loc("SALES_VOLUME")+1, "PREF_VALUE", 1)
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        
        " Sales Forecasting "
        
        DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
        DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]
        
        
        runfile(path+'/SALES_PREDICTION_FUNCTION.py', wdir=path)
        
        DATAF_ALL_PROJ = Sales_Prediction(DATAF_ALL_PROJ,sales_input_col)
        
        DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        " Shapley values calculation "
        
        runfile(path+'/SHAP_VALUES_TOTAL_OTHER_SCENARIOS.py', wdir=path)
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        " Deleting additional columns such as SUBCAT_NAME, CHANNEL_NAME which were added earlier "
        
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        " Rearranging columns in the DATAF_ALL_HIST_PROJ dataframe "
        
        " This also includes the SUBCAT, CHANNEL, FORMAT names added before "
        
        column_names = ['UL_GEO_ID', 'UL_REGION_NAME', 'SUBCAT_NAME', 'CHANNEL_NAME',
           'FORMAT_NAME', 'CATG_CD', 'CHNL_CD', 'FMT_CD', 'SUBCAT_CD',
           'PERIOD_BEGIN_DATE', 'YEAR', 'MONTH', 'HOLIDAY_CD', 'SALES_VALUE',
           'SALES_VOLUME', 'SALES_VOLUME_PREDICTED', 'PREF_VALUE', 'VOL_UNIT_CD',
           'PRICE_PER_VOL', 'TDP', 'AVG(MAX_TEMP_CELSIUS)',
           'AVG(MIN_TEMP_CELSIUS)', 'AVG(AVG_TEMP_CELSIUS)', 'AVG(HUMID_PCT)',
           'AVG(FEELS_LIKE_CELSIUS)', 'AVG(AVG_WIND_MPH)', 'AVG(PRECIP_MM)',
           'AVG(SNOW_CM)', 'SUM(NEW_CASES)', 'SUM(NEW_DEATHS)', 'SUM(CUMU_CASES)',
           'SUM(CUMU_DEATHS)', 'AVG(TRAFFIC_WEIGHT)',
           'STRINGENCY_INDEX', 'SECTOR_SCENARIO_CD',
           'SECTOR_SCENARIO_DESC', 'AVG(CONSUMER_PRICE_INDEX)',
           'AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)', 'AVG(GDP_NOMINAL_LCU)',
           'AVG(GDP_REAL_LCU)', 'AVG(RETAIL_PRICES_INDEX)',
           'AVG(SHARE_PRICE_INDEX)', 'AVG(UNEMP_RATE)', 'ANXIETY_CONCERNED_PCT',
           'ANXIETY_CASES', 'AVG(ANXIETY_CONCERNED_PCT)',
           'AVG(CHANGE_SINCE_PREV_FORTNIGHT)', 'RATIO_COVID', 'TIME_COVID',
           'SEASONALITY_INDEX_SALES', 'SEASONALITY_INDEX_TDP',
           'SEASONALITY_INDEX_PRICE', 'SALES_TREND_CAL']
        
        DATAF_ALL_HIST_PROJ = DATAF_ALL_HIST_PROJ.reindex(columns=column_names)
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        " Renaming some of the variables names i.e. removing AVG() and SUM() from the names "
        " Following code may have to be modified if the variables changes "
        complete_data_for_rahul = pd.concat([complete_data_for_rahul,DATAF_ALL_HIST_PROJ])
        complete_shap = pd.concat([complete_shap,DATAF_SHAP_ALL])
        sub_name = str(DATAF_ALL_HIST_PROJ.SUBCAT_NAME.unique()[0])
        del DATAF_ALL_HIST_PROJ["SUBCAT_NAME"]
        del DATAF_ALL_HIST_PROJ["CHANNEL_NAME"]
        del DATAF_ALL_HIST_PROJ["FORMAT_NAME"]
        
          
        
        
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(MAX_TEMP_CELSIUS)':'MAX_TEMP_CELSIUS'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(AVG_TEMP_CELSIUS)':'AVG_TEMP_CELSIUS'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(MIN_TEMP_CELSIUS)':'MIN_TEMP_CELSIUS'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(FEELS_LIKE_CELSIUS)':'FEELS_LIKE_CELSIUS'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(AVG_WIND_MPH)':'AVG_WIND_MPH'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(HUMID_PCT)':'HUMID_PCT'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(PRECIP_MM)':'PRECIP_MM'}, inplace = True)
        
        DATAF_ALL_HIST_PROJ.rename(columns = {'SUM(NEW_CASES)':'NEW_CASES'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'SUM(NEW_DEATHS)':'NEW_DEATHS'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'SUM(CUMU_CASES)':'CUMU_CASES'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'SUM(CUMU_DEATHS)':'CUMU_DEATHS'}, inplace = True)
        
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(TRAFFIC_WEIGHT)':'TRAFFIC_WEIGHT'}, inplace = True)
       
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(CONSUMER_PRICE_INDEX)':'CONSUMER_PRICE_INDEX'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)':'PERSONAL_DISPOSABLE_INCOME_REAL_LCU'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(GDP_NOMINAL_LCU)':'GDP_NOMINAL_LCU'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(GDP_REAL_LCU)':'GDP_REAL_LCU'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(RETAIL_PRICES_INDEX)':'RETAIL_PRICES_INDEX'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(SHARE_PRICE_INDEX)':'SHARE_PRICE_INDEX'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(UNEMP_RATE)':'UNEMP_RATE'}, inplace = True)
        
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(ANXIETY_CONCERNED_PCT)':'ANXIETY_CONCERNED_PCT'}, inplace = True)
        DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(CHANGE_SINCE_PREV_FORTNIGHT)':'CHANGE_SINCE_PREV_FORTNIGHT'}, inplace = True)
        
        "-----------------------------------------------------------------------------------------------------------------------------" 
        
        
        " Exporting the file "
        
        file = 'CHN_'+sub_name+'_total_level_SCENARIO_'+str(SCENARIO_NO)+'.csv'
        
        DATAF_ALL_HIST_PROJ.to_csv(file,index=False)
        complete_data_for_sanjay = pd.concat([complete_data_for_sanjay,DATAF_ALL_HIST_PROJ])
        
        
        
    complete_data_for_rahul.to_csv('CHN_'+sub_name+'_total_level_sales_ALL_SCENARIO_for_rahul.csv',index =False)
    complete_data_for_sanjay.to_csv('CHN_'+sub_name+'_total_level_sales_ALL_SCENARIO_for_sanjay.csv',index =False)
    
    
    complete_shap.to_csv('CHN_'+sub_name+'_total_level_shap_ALL_SCENARIO.csv',index =False)
    all_data_all_scenarios_for_analysis = pd.concat([all_data_all_scenarios_for_analysis,complete_data_for_rahul])
    all_data_all_scenarios_for_upload = pd.concat([all_data_all_scenarios_for_upload,complete_data_for_sanjay])
    all_shap_all_scenarios = pd.concat([all_shap_all_scenarios,complete_shap])
    
    
all_data_all_scenarios_for_analysis.to_csv("China_cells_total_level_without_online_for_analysis.csv",index =False)
all_data_all_scenarios_for_upload.to_csv("China_cells_total_level_without_online_for_upload.csv",index =False)
all_shap_all_scenarios.to_csv("China_cells_total_level_shap_values_without_online_for_upload.csv",index =False)
