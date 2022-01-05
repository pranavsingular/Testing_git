


" Framework for other scenarios "

get_ipython().magic('clear')
import warnings
warnings.filterwarnings("ignore")

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

"-----------------------------------------------------------------------------------------------------------------------------" 

" Enter the directory path "

path = 'D:/unilever_gdm/preference/Total_Framework'

"-----------------------------------------------------------------------------------------------------------------------------" 

print(" Data Acquisition ")

" Enter the output file (prediction sales) acquired from the Total_Framework run "

DATAF_ALL_HIST_PROJ = pd.read_csv('DATAF_ALL_HIST_PROJ_TOTALSALES.csv')

"-----------------------------------------------------------------------------------------------------------------------------" 

" Enter dates according to the data "

Start_month = 7
Start_year = 2018

End_month = 6
End_year = 2021

Hist_end_date = '2021-06-01'
Covid_start_date = '2020-03-01'
Covid_model_start_date = '2019-11-01'
Covid_start_month = 3
Covid_start_year = 2020

Qtr_Range = range(1,5)
Year_Range = range(2018,2024)

Prev_Year_Range = range(2018,2023)

"-----------------------------------------------------------------------------------------------------------------------------" 

" Run for all the scenarios 1,2,3,4,5 "

SCENARIO_NO = 1

"-----------------------------------------------------------------------------------------------------------------------------" 

print(" Adding forecasts for Macro-economic variables ")

DATAF_ALL_MACRO = pd.read_csv('MacroIndia.csv')

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

DATAF_COVID_NOS = pd.read_csv('COVID_NUMBERS_TOTAL.csv')

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

" Creating new input features for COVID"

print(" Adding New Input features for COVID predictive modelling ")

del DATAF_ALL_HIST_PROJ["RATIO_COVID"]
del DATAF_ALL_HIST_PROJ["TIME_COVID"]

s = DATAF_ALL_HIST_PROJ.shape[0]

RATIO_COVID = np.zeros((s), dtype=float)
TIME_COVID = np.zeros((s), dtype=float)

for i in range(s):
    YEAR = DATAF_ALL_HIST_PROJ.YEAR.iloc[i]
    MONTH = DATAF_ALL_HIST_PROJ.MONTH.iloc[i]
    
    num_months = (YEAR - Covid_start_year) * 12 + (MONTH - Covid_start_month)
    
    if num_months<0:
        num_months=0
        
        
    TIME_COVID[i] = num_months
    
    if DATAF_ALL_HIST_PROJ['SUM(NEW_CASES)'].iloc[i]==0 or DATAF_ALL_HIST_PROJ['SUM(NEW_DEATHS)'].iloc[i]==0:
        RATIO_COVID[i]=0
    else:
        RATIO_COVID[i] = DATAF_ALL_HIST_PROJ['SUM(NEW_DEATHS)'].iloc[i]/DATAF_ALL_HIST_PROJ['SUM(NEW_CASES)'].iloc[i]
        
        
DATAF_ALL_HIST_PROJ.insert(2, "RATIO_COVID", RATIO_COVID)     
DATAF_ALL_HIST_PROJ.insert(3, "TIME_COVID", TIME_COVID)


"-----------------------------------------------------------------------------------------------------------------------------" 

" Covid mobility forecast "

" Filtering Historic & Forecast Period "
DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]

runfile(path+'/COVID_MOBILITY_PREDICTION_FUNCTION.py', wdir=path)

DATAF_ALL_PROJ = COVID_Mobility_Prediction(DATAF_ALL_PROJ)

DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])

"-----------------------------------------------------------------------------------------------------------------------------" 

" Plotting Covid Mobility for check "

SUBCAT_LIST = DATAF_ALL_HIST_PROJ.SUBCAT_CD.unique()

DATAF_FILTER = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.SUBCAT_CD==SUBCAT_LIST[0])]

GEO_LIST = DATAF_FILTER.UL_GEO_ID.unique()

for i in GEO_LIST:
    DATAF = DATAF_FILTER[(DATAF_FILTER.UL_GEO_ID==i)]
    print(i)
    print("Historic and Forecasted New Cases ")
    plt.figure(figsize=(12,6))
    plt.plot(DATAF['SUM(NEW_CASES)'].values)
    plt.show() 
    
    print("Historic and Forecasted New Deaths ")
    plt.figure(figsize=(12,6))
    plt.plot(DATAF['SUM(NEW_DEATHS)'].values)
    plt.show() 
    
    print("Historic and Forecasted Retail Mobility ")
    plt.figure(figsize=(12,6))
    plt.plot(DATAF['AVG(RETAIL_AND_RECREATION_PCT_CHANGE)'].values)
    plt.show() 
    
    print("Historic and Forecasted Residential Mobility ")
    plt.figure(figsize=(12,6))
    plt.plot(DATAF['AVG(RESIDENTIAL_PCT_CHANGE)'].values)
    plt.show() 

"-----------------------------------------------------------------------------------------------------------------------------" 

" Price Forecast "

" Filtering Historic & Forecast Period "
DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]

runfile(path+'/PRICE_PREDICTION_FUNCTION_TOTAL.py', wdir=path)

DATAF_ALL_PROJ = Price_Prediction(DATAF_ALL_PROJ)

DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])

"-----------------------------------------------------------------------------------------------------------------------------" 

" TDP Forecast "

" Filtering Historic & Forecast Period "
DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]


runfile(path+'/TDP_PREDICTION_FUNCTION_TOTAL.py', wdir=path)

DATAF_ALL_PROJ = TDP_Prediction(DATAF_ALL_PROJ)

DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])

"-----------------------------------------------------------------------------------------------------------------------------" 


" Adding constant column for Preference "

" Preference value not applicable for total level modelling. Hence, adding constant value of 1 "

DATAF_ALL_HIST_PROJ.insert(DATAF_ALL_HIST_PROJ.columns.get_loc("SALES_VOLUME")+1, "PREF_VALUE", 1)

"-----------------------------------------------------------------------------------------------------------------------------" 


" Sales Forecasting "

DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]


runfile(path+'/SALES_PREDICTION_FUNCTION.py', wdir=path)

DATAF_ALL_PROJ = Sales_Prediction(DATAF_ALL_PROJ)

DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])

"-----------------------------------------------------------------------------------------------------------------------------" 

" Shapley values calculation "

runfile(path+'/SHAP_VALUES_TOTAL_OTHER_SCENARIOS.py', wdir=path)

"-----------------------------------------------------------------------------------------------------------------------------" 

" Deleting additional columns such as SUBCAT_NAME, CHANNEL_NAME which were added earlier "

del DATAF_ALL_HIST_PROJ["SUBCAT_NAME"]
del DATAF_ALL_HIST_PROJ["CHANNEL_NAME"]
del DATAF_ALL_HIST_PROJ["FORMAT_NAME"]

"-----------------------------------------------------------------------------------------------------------------------------" 

" Rearranging columns in the DATAF_ALL_HIST_PROJ dataframe "

" This also includes the SUBCAT, CHANNEL, FORMAT names added before "

column_names = ['UL_GEO_ID', 'UL_REGION_NAME', 'CATG_CD', 'CHNL_CD', 'FMT_CD', 'SUBCAT_CD',
       'PERIOD_BEGIN_DATE', 'YEAR', 'MONTH', 'HOLIDAY_CD', 'SALES_VALUE',
       'SALES_VOLUME', 'SALES_VOLUME_PREDICTED', 'PREF_VALUE', 'VOL_UNIT_CD',
       'PRICE_PER_VOL', 'TDP', 'AVG(MAX_TEMP_CELSIUS)',
       'AVG(MIN_TEMP_CELSIUS)', 'AVG(AVG_TEMP_CELSIUS)', 'AVG(HUMID_PCT)',
       'AVG(FEELS_LIKE_CELSIUS)', 'AVG(AVG_WIND_MPH)', 'AVG(PRECIP_MM)',
       'AVG(SNOW_CM)', 'SUM(NEW_CASES)', 'SUM(NEW_DEATHS)', 'SUM(CUMU_CASES)',
       'SUM(CUMU_DEATHS)', 'AVG(RETAIL_AND_RECREATION_PCT_CHANGE)',
       'AVG(GROCERY_AND_PHARMACY_PCT_CHANGE)', 'AVG(PARKS_PCT_CHANGE)',
       'AVG(TRANSIT_STATIONS_PCT_CHANGE)', 'AVG(WORKPLACES_PCT_CHANGE)',
       'AVG(RESIDENTIAL_PCT_CHANGE)', 'STRINGENCY_INDEX', 'SECTOR_SCENARIO_CD',
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

DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(MAX_TEMP_CELSIUS)':'MAX_TEMP_CELSIUS'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(AVG_TEMP_CELSIUS)':'AVG_TEMP_CELSIUS'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(MIN_TEMP_CELSIUS)':'MIN_TEMP_CELSIUS'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(FEELS_LIKE_CELSIUS)':'FEELS_LIKE_CELSIUS'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(AVG_WIND_MPH)':'AVG_WIND_MPH'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(HUMID_PCT)':'HUMID_PCT'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(PRECIP_MM)':'PRECIP_MM'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(SNOW_CM)':'SNOW_CM'}, inplace = True)

DATAF_ALL_HIST_PROJ.rename(columns = {'SUM(NEW_CASES)':'NEW_CASES'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'SUM(NEW_DEATHS)':'NEW_DEATHS'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'SUM(CUMU_CASES)':'CUMU_CASES'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'SUM(CUMU_DEATHS)':'CUMU_DEATHS'}, inplace = True)

DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(RETAIL_AND_RECREATION_PCT_CHANGE)':'RETAIL_AND_RECREATION_PCT_CHANGE'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(GROCERY_AND_PHARMACY_PCT_CHANGE)':'GROCERY_AND_PHARMACY_PCT_CHANGE'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(PARKS_PCT_CHANGE)':'PARKS_PCT_CHANGE'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(TRANSIT_STATIONS_PCT_CHANGE)':'TRANSIT_STATIONS_PCT_CHANGE'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(WORKPLACES_PCT_CHANGE)':'WORKPLACES_PCT_CHANGE'}, inplace = True)
DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(RESIDENTIAL_PCT_CHANGE)':'RESIDENTIAL_PCT_CHANGE'}, inplace = True)

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

" Adding RECORD_TYPE column "

s = DATAF_ALL_HIST_PROJ.shape[0]

RECORD_TYPE = ["" for i in range(s)]

for i in range(s):
    SALES = DATAF_ALL_HIST_PROJ.SALES_VOLUME.iloc[i]
    if SALES==0:
        RECORD_TYPE[i] = "FORECAST"
    else:
        RECORD_TYPE[i] = "ACTUAL"
        

DATAF_ALL_HIST_PROJ.insert(13, "RECORD_TYPE", RECORD_TYPE)       


" Exporting the file "

file = 'DATAF_ALL_HIST_PROJ_SALES_SCENARIO_'+str(SCENARIO_NO)+'.csv'

DATAF_ALL_HIST_PROJ.to_csv(file,index=False)



