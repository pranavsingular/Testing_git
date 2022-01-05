






" REPORT GENERATION "
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
from sklearn.linear_model import LinearRegression
from sklearn.inspection import plot_partial_dependence
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.svm import SVR

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


"Generating sales growth numbers"

DATAF_ALL_HIST_PROJ_SALES = pd.read_csv('Washing_Powder_Revised.csv')

" Adding SALES VALUE FORECAST "

DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ_SALES[(DATAF_ALL_HIST_PROJ_SALES['SALES_VOLUME']>0)]
DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ_SALES[(DATAF_ALL_HIST_PROJ_SALES['SALES_VOLUME']==0)]


DATAF_ALL_HIST.SALES_VALUE = DATAF_ALL_HIST.SALES_VOLUME*DATAF_ALL_HIST.PRICE_PER_VOL

DATAF_ALL_PROJ.SALES_VALUE = DATAF_ALL_PROJ.SALES_VOLUME_PREDICTED*DATAF_ALL_PROJ.PRICE_PER_VOL

DATAF_ALL_HIST_PROJ_SALES = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])

DATAF_ALL_HIST_PROJ_SALES_COPY = DATAF_ALL_HIST_PROJ_SALES.copy()


s = DATAF_ALL_HIST_PROJ_SALES.shape[0]
QUARTER = np.zeros((s), dtype=float)

for i in range(s):
    MONTH = DATAF_ALL_HIST_PROJ_SALES.MONTH.iloc[i]
    
    if MONTH<4:
        QUARTER[i] = 1
    elif MONTH>3 and MONTH<7:
        QUARTER[i] = 2
    elif MONTH>6 and MONTH<10:
        QUARTER[i] = 3
    elif MONTH>9:
        QUARTER[i] = 4
        
        
        
DATAF_ALL_HIST_PROJ_SALES.insert(DATAF_ALL_HIST_PROJ_SALES.columns.get_loc("MONTH")+1, "QUARTER", QUARTER)


DATAF_ALL_START = DATAF_ALL_HIST_PROJ_SALES[(DATAF_ALL_HIST_PROJ_SALES.YEAR==Start_year) & (DATAF_ALL_HIST_PROJ_SALES.MONTH==Start_month)]

s = DATAF_ALL_START.shape[0]

DATAF_ALL_SALES_QTR = pd.DataFrame(columns = ['SUBCAT_CD','UL_GEO_ID', 'UL_REGION_NAME', 'CHNL_CD','FMT_CD', 'YEAR', 'QUARTER', 'ACTUAL_SALES','PREDICTED_SALES','SALES_VALUE','SEASONALITY_INDEX','PRICE_PER_VOL','TDP','PREF_VALUE','AVG(AVG_TEMP_CELSIUS)','AVG(HUMID_PCT)','AVG(RETAIL_AND_RECREATION_PCT_CHANGE)','AVG(RESIDENTIAL_PCT_CHANGE)','AVG(GDP_REAL_LCU)','AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)','AVG(UNEMP_RATE)'])

for i in range(s):
    SUBCAT = DATAF_ALL_START.SUBCAT_CD.iloc[i]
    #SUBCAT_NAME = DATAF_ALL_START.SUBCAT_NAME.iloc[i]
    CHNL= DATAF_ALL_START.CHNL_CD.iloc[i]
    #CHNL_NAME= DATAF_ALL_START.CHANNEL_NAME.iloc[i]
    REG = DATAF_ALL_START.UL_GEO_ID.iloc[i]
    REG_NAME = DATAF_ALL_START.UL_REGION_NAME.iloc[i]
    FMT = DATAF_ALL_START.FMT_CD.iloc[i]
    #FMT_NAME = DATAF_ALL_START.FORMAT_NAME.iloc[i]
    
    DATAF_FIL_COMBO = DATAF_ALL_HIST_PROJ_SALES[(DATAF_ALL_HIST_PROJ_SALES.SUBCAT_CD==SUBCAT)&(DATAF_ALL_HIST_PROJ_SALES.CHNL_CD==CHNL)&(DATAF_ALL_HIST_PROJ_SALES.UL_GEO_ID==REG)&(DATAF_ALL_HIST_PROJ_SALES.FMT_CD==FMT)]
    
    for j in Year_Range:
        for k in Qtr_Range:
            DATAF_FIL_QTR_YR = DATAF_FIL_COMBO[(DATAF_FIL_COMBO.YEAR==j) & (DATAF_FIL_COMBO.QUARTER==k)]
            if DATAF_FIL_QTR_YR.shape[0]>0:
                Actual_Sales = np.sum(DATAF_FIL_QTR_YR.SALES_VOLUME)
                Predicted_Sales = np.sum(DATAF_FIL_QTR_YR.SALES_VOLUME_PREDICTED)
                Sales_value = np.sum(DATAF_FIL_QTR_YR.SALES_VALUE)
                Seasonality = np.sum(DATAF_FIL_QTR_YR.SEASONALITY_INDEX_SALES)
                Price = np.sum(DATAF_FIL_QTR_YR.PRICE_PER_VOL)
                tdp = np.sum(DATAF_FIL_QTR_YR.TDP)
                Pref = np.sum(DATAF_FIL_QTR_YR.PREF_VALUE)
                #Min_Temp = np.sum(DATAF_FIL_QTR_YR['AVG(MIN_TEMP_CELSIUS)'])
                Avg_Temp = np.sum(DATAF_FIL_QTR_YR['AVG(AVG_TEMP_CELSIUS)'])
                Humid = np.sum(DATAF_FIL_QTR_YR['AVG(HUMID_PCT)'])
                Precip = np.sum(DATAF_FIL_QTR_YR['AVG(PRECIP_MM)'])
                Retail_Mob = np.sum(DATAF_FIL_QTR_YR['AVG(RETAIL_AND_RECREATION_PCT_CHANGE)'])
                Resid_Mob = np.sum(DATAF_FIL_QTR_YR['AVG(RESIDENTIAL_PCT_CHANGE)'])
                gdp = np.sum(DATAF_FIL_QTR_YR['AVG(GDP_REAL_LCU)'])
                pdi = np.sum(DATAF_FIL_QTR_YR['AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)'])
                unemp = np.sum(DATAF_FIL_QTR_YR['AVG(UNEMP_RATE)'])
                DATAF_ALL_SALES_QTR = DATAF_ALL_SALES_QTR.append({'SUBCAT_CD' : SUBCAT, 'UL_GEO_ID' : REG,'UL_REGION_NAME' : REG_NAME,'CHNL_CD' : CHNL, 'FMT_CD' : FMT, 'YEAR' : j, 'QUARTER' : k, 'ACTUAL_SALES' : Actual_Sales, 'PREDICTED_SALES' : Predicted_Sales,'SALES_VALUE' : Sales_value,'SEASONALITY_INDEX' : Seasonality,'PRICE_PER_VOL' : Price,'TDP' : tdp,'PREF_VALUE' : Pref,'AVG(AVG_TEMP_CELSIUS)' : Avg_Temp,'AVG(HUMID_PCT)' : Humid,'AVG(RETAIL_AND_RECREATION_PCT_CHANGE)' : Retail_Mob,'AVG(RESIDENTIAL_PCT_CHANGE)' : Resid_Mob,'AVG(GDP_REAL_LCU)' : gdp,'AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)' : pdi,'AVG(UNEMP_RATE)' : unemp}, ignore_index = True)
            
        
        

DATAF_ALL_SALES_QTR_GROWTH = pd.DataFrame(columns = ['SUBCAT_CD','UL_GEO_ID', 'UL_REGION_NAME', 'CHNL_CD','FMT_CD', 'PREV_YEAR','CURR_YEAR', 'QUARTER', 'SALES_VOLUME_GROWTH','SALES_VALUE_GROWTH','SEASONALITY_INDEX','PRICE_PER_VOL','TDP','PREF_VALUE','AVG(AVG_TEMP_CELSIUS)','AVG(HUMID_PCT)','AVG(RETAIL_AND_RECREATION_PCT_CHANGE)','AVG(RESIDENTIAL_PCT_CHANGE)','AVG(GDP_REAL_LCU)','AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)','AVG(UNEMP_RATE)'])
s = DATAF_ALL_START.shape[0]

for x in range(s):
    SUBCAT = DATAF_ALL_START.SUBCAT_CD.iloc[x]
    #SUBCAT_NAME = DATAF_ALL_START.SUBCAT_NAME.iloc[x]
    CHNL= DATAF_ALL_START.CHNL_CD.iloc[x]
    #CHNL_NAME= DATAF_ALL_START.CHANNEL_NAME.iloc[x]
    REG = DATAF_ALL_START.UL_GEO_ID.iloc[x]
    REG_NAME = DATAF_ALL_START.UL_REGION_NAME.iloc[x]
    FMT = DATAF_ALL_START.FMT_CD.iloc[x]
    #FMT_NAME = DATAF_ALL_START.FORMAT_NAME.iloc[x]
    
    DATAF_FIL_COMBO_QTR = DATAF_ALL_SALES_QTR[(DATAF_ALL_SALES_QTR.SUBCAT_CD==SUBCAT)&(DATAF_ALL_SALES_QTR.CHNL_CD==CHNL)&(DATAF_ALL_SALES_QTR.UL_GEO_ID==REG)&(DATAF_ALL_SALES_QTR.FMT_CD==FMT)]
    
    for i in Prev_Year_Range:
        for j in Qtr_Range:
                DATAF_FIL_QTR_PREV = DATAF_FIL_COMBO_QTR[(DATAF_FIL_COMBO_QTR.YEAR==i) & (DATAF_FIL_COMBO_QTR.QUARTER==j)]
                DATAF_FIL_QTR_CURR = DATAF_FIL_COMBO_QTR[(DATAF_FIL_COMBO_QTR.YEAR==i+1) & (DATAF_FIL_COMBO_QTR.QUARTER==j)] 
                
                if DATAF_FIL_QTR_PREV.shape[0]>0 and DATAF_FIL_QTR_CURR.shape[0]>0:
                    if DATAF_FIL_QTR_PREV.ACTUAL_SALES.iloc[0]>0 and DATAF_FIL_QTR_CURR.ACTUAL_SALES.iloc[0]>0:
                        Sales_Vol_Growth = (DATAF_FIL_QTR_CURR.ACTUAL_SALES.iloc[0]-DATAF_FIL_QTR_PREV.ACTUAL_SALES.iloc[0])/DATAF_FIL_QTR_PREV.ACTUAL_SALES.iloc[0]
                        Sales_Vol_Growth = np.round(Sales_Vol_Growth*100,2)
                    elif DATAF_FIL_QTR_PREV.ACTUAL_SALES.iloc[0]>0 and DATAF_FIL_QTR_CURR.ACTUAL_SALES.iloc[0]==0:
                        Sales_Vol_Growth = (DATAF_FIL_QTR_CURR.PREDICTED_SALES.iloc[0]-DATAF_FIL_QTR_PREV.ACTUAL_SALES.iloc[0])/DATAF_FIL_QTR_PREV.ACTUAL_SALES.iloc[0]
                        Sales_Vol_Growth = np.round(Sales_Vol_Growth*100,2)
                    else:
                        Sales_Vol_Growth = (DATAF_FIL_QTR_CURR.PREDICTED_SALES.iloc[0]-DATAF_FIL_QTR_PREV.PREDICTED_SALES.iloc[0])/DATAF_FIL_QTR_PREV.PREDICTED_SALES.iloc[0]
                        Sales_Vol_Growth = np.round(Sales_Vol_Growth*100,2)
                        
                        
                    Sales_Val_Growth = (DATAF_FIL_QTR_CURR.SALES_VALUE.iloc[0]-DATAF_FIL_QTR_PREV.SALES_VALUE.iloc[0])/DATAF_FIL_QTR_PREV.SALES_VALUE.iloc[0]
                    Sales_Val_Growth = np.round(Sales_Val_Growth*100,2)
                    Seasonality_Change = (DATAF_FIL_QTR_CURR.SEASONALITY_INDEX.iloc[0]-DATAF_FIL_QTR_PREV.SEASONALITY_INDEX.iloc[0])/DATAF_FIL_QTR_PREV.SEASONALITY_INDEX.iloc[0]
                    Seasonality_Change = np.round(Seasonality_Change*100,2)
                    Price_Change = (DATAF_FIL_QTR_CURR.PRICE_PER_VOL.iloc[0]-DATAF_FIL_QTR_PREV.PRICE_PER_VOL.iloc[0])/DATAF_FIL_QTR_PREV.PRICE_PER_VOL.iloc[0]
                    Price_Change = np.round(Price_Change*100,2)
                    TDP_Change = (DATAF_FIL_QTR_CURR.TDP.iloc[0]-DATAF_FIL_QTR_PREV.TDP.iloc[0])/DATAF_FIL_QTR_PREV.TDP.iloc[0]
                    TDP_Change = np.round(TDP_Change*100,2)
                    Pref_Change = (DATAF_FIL_QTR_CURR.PREF_VALUE.iloc[0]-DATAF_FIL_QTR_PREV.PREF_VALUE.iloc[0])/DATAF_FIL_QTR_PREV.PREF_VALUE.iloc[0]
                    Pref_Change = np.round(Pref_Change*100,2)
                    #Min_Temp_Change = (DATAF_FIL_QTR_CURR['AVG(MIN_TEMP_CELSIUS)'].iloc[0]-DATAF_FIL_QTR_PREV['AVG(MIN_TEMP_CELSIUS)'].iloc[0])/DATAF_FIL_QTR_PREV['AVG(MIN_TEMP_CELSIUS)'].iloc[0]
                    #Min_Temp_Change = np.round(Min_Temp_Change*100,2)
                    Avg_Temp_Change = (DATAF_FIL_QTR_CURR['AVG(AVG_TEMP_CELSIUS)'].iloc[0]-DATAF_FIL_QTR_PREV['AVG(AVG_TEMP_CELSIUS)'].iloc[0])/DATAF_FIL_QTR_PREV['AVG(AVG_TEMP_CELSIUS)'].iloc[0]
                    Avg_Temp_Change = np.round(Avg_Temp_Change*100,2)
                    Humid_Change = (DATAF_FIL_QTR_CURR['AVG(HUMID_PCT)'].iloc[0]-DATAF_FIL_QTR_PREV['AVG(HUMID_PCT)'].iloc[0])/DATAF_FIL_QTR_PREV['AVG(HUMID_PCT)'].iloc[0]
                    Humid_Change = np.round(Humid_Change*100,2)
                    #Precip_Change = (DATAF_FIL_QTR_CURR['AVG(PRECIP_MM)'].iloc[0]-DATAF_FIL_QTR_PREV['AVG(PRECIP_MM)'].iloc[0])/DATAF_FIL_QTR_PREV['AVG(PRECIP_MM)'].iloc[0]
                    #Precip_Change = np.round(Precip_Change*100,2)
                    Retail_Change = (DATAF_FIL_QTR_CURR['AVG(RETAIL_AND_RECREATION_PCT_CHANGE)'].iloc[0]-DATAF_FIL_QTR_PREV['AVG(RETAIL_AND_RECREATION_PCT_CHANGE)'].iloc[0])/DATAF_FIL_QTR_PREV['AVG(RETAIL_AND_RECREATION_PCT_CHANGE)'].iloc[0]
                    Retail_Change = np.round(Retail_Change*100,2)
                    Resid_Change = (DATAF_FIL_QTR_CURR['AVG(RESIDENTIAL_PCT_CHANGE)'].iloc[0]-DATAF_FIL_QTR_PREV['AVG(RESIDENTIAL_PCT_CHANGE)'].iloc[0])/DATAF_FIL_QTR_PREV['AVG(RESIDENTIAL_PCT_CHANGE)'].iloc[0]
                    Resid_Change = np.round(Resid_Change*100,2)
                    GDP_Change = (DATAF_FIL_QTR_CURR['AVG(GDP_REAL_LCU)'].iloc[0]-DATAF_FIL_QTR_PREV['AVG(GDP_REAL_LCU)'].iloc[0])/DATAF_FIL_QTR_PREV['AVG(GDP_REAL_LCU)'].iloc[0]
                    GDP_Change = np.round(GDP_Change*100,2)
                    PDI_Change = (DATAF_FIL_QTR_CURR['AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)'].iloc[0]-DATAF_FIL_QTR_PREV['AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)'].iloc[0])/DATAF_FIL_QTR_PREV['AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)'].iloc[0]
                    PDI_Change = np.round(PDI_Change*100,2)
                    UNEMP_Change = (DATAF_FIL_QTR_CURR['AVG(UNEMP_RATE)'].iloc[0]-DATAF_FIL_QTR_PREV['AVG(UNEMP_RATE)'].iloc[0])/DATAF_FIL_QTR_PREV['AVG(UNEMP_RATE)'].iloc[0]
                    UNEMP_Change = np.round(UNEMP_Change*100,2)
                    DATAF_ALL_SALES_QTR_GROWTH = DATAF_ALL_SALES_QTR_GROWTH.append({'SUBCAT_CD' : SUBCAT, 'UL_GEO_ID' : REG,'UL_REGION_NAME' : REG_NAME,'CHNL_CD' : CHNL, 'FMT_CD' : FMT, 'PREV_YEAR' : i,'CURR_YEAR' : i+1, 'QUARTER' : j, 'SALES_VOLUME_GROWTH' : Sales_Vol_Growth ,'SALES_VALUE_GROWTH' : Sales_Val_Growth ,'SEASONALITY_INDEX' : Seasonality_Change,'PRICE_PER_VOL' : Price_Change,'TDP' : TDP_Change,'PREF_VALUE' : Pref_Change,'AVG(AVG_TEMP_CELSIUS)' : Avg_Temp_Change,'AVG(HUMID_PCT)' : Humid_Change,'AVG(RETAIL_AND_RECREATION_PCT_CHANGE)' : Retail_Change,'AVG(RESIDENTIAL_PCT_CHANGE)' : Resid_Change,'AVG(GDP_REAL_LCU)' : GDP_Change,'AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)':PDI_Change,'AVG(UNEMP_RATE)' : UNEMP_Change}, ignore_index = True)
                else:
                    DATAF_ALL_SALES_QTR_GROWTH = DATAF_ALL_SALES_QTR_GROWTH.append({'SUBCAT_CD' : SUBCAT, 'UL_GEO_ID' : REG,'UL_REGION_NAME' : REG_NAME,'CHNL_CD' : CHNL, 'FMT_CD' : FMT,  'PREV_YEAR' : i,'CURR_YEAR' : i+1, 'QUARTER' : j, 'SALES_VOLUME_GROWTH' : 'NA' ,'SALES_VALUE_GROWTH' : 'NA' ,'SEASONALITY_INDEX' : 'NA','PRICE_PER_VOL' : 'NA','TDP' : 'NA','PREF_VALUE' : 'NA','AVG(AVG_TEMP_CELSIUS)' : 'NA','AVG(HUMID_PCT)' : 'NA','AVG(RETAIL_AND_RECREATION_PCT_CHANGE)' : 'NA','AVG(RESIDENTIAL_PCT_CHANGE)' : 'NA','AVG(GDP_REAL_LCU)' : 'NA','AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)':'NA','AVG(UNEMP_RATE)' : 'NA'}, ignore_index = True)



" Adding absolute Sales and Value "

s = DATAF_ALL_SALES_QTR_GROWTH.shape[0]

SALES_VOL_PREV_YEAR_ACTUAL = np.zeros((s), dtype=float)
SALES_VOL_CURR_YEAR_ACTUAL = np.zeros((s), dtype=float)

SALES_VOL_PREV_YEAR_PRED = np.zeros((s), dtype=float)
SALES_VOL_CURR_YEAR_PRED = np.zeros((s), dtype=float)

SALES_VALUE_PREV_YEAR = np.zeros((s), dtype=float)
SALES_VALUE_CURR_YEAR = np.zeros((s), dtype=float)

for i in range(s):
    PREV_YEAR = DATAF_ALL_SALES_QTR_GROWTH.PREV_YEAR.iloc[i]
    CURR_YEAR = DATAF_ALL_SALES_QTR_GROWTH.CURR_YEAR.iloc[i]
    QTR = DATAF_ALL_SALES_QTR_GROWTH.QUARTER.iloc[i]
    SUBCAT = DATAF_ALL_SALES_QTR_GROWTH.SUBCAT_CD.iloc[i]
    
    " Previous Year Sales and Value "
    DATAF_QTR_FIL = DATAF_ALL_SALES_QTR[(DATAF_ALL_SALES_QTR.YEAR==PREV_YEAR) & (DATAF_ALL_SALES_QTR.QUARTER==QTR) & (DATAF_ALL_SALES_QTR.SUBCAT_CD==SUBCAT)]
    if DATAF_QTR_FIL.shape[0]>0:
        SALES_VOL_PREV_YEAR_ACTUAL[i] = DATAF_QTR_FIL.ACTUAL_SALES.iloc[0]
        SALES_VOL_PREV_YEAR_PRED[i] = DATAF_QTR_FIL.PREDICTED_SALES.iloc[0]
        SALES_VALUE_PREV_YEAR[i] = DATAF_QTR_FIL.SALES_VALUE.iloc[0]
    else:
        SALES_VOL_PREV_YEAR_ACTUAL[i] = -1
        SALES_VOL_PREV_YEAR_PRED[i] = -1
        SALES_VALUE_PREV_YEAR[i] = -1
    
    " Current Year Sales and Value "
    DATAF_QTR_FIL = DATAF_ALL_SALES_QTR[(DATAF_ALL_SALES_QTR.YEAR==CURR_YEAR) & (DATAF_ALL_SALES_QTR.QUARTER==QTR) & (DATAF_ALL_SALES_QTR.SUBCAT_CD==SUBCAT)]
    
    if DATAF_QTR_FIL.shape[0]>0:
        SALES_VOL_CURR_YEAR_ACTUAL[i] = DATAF_QTR_FIL.ACTUAL_SALES.iloc[0]
        SALES_VOL_CURR_YEAR_PRED[i] = DATAF_QTR_FIL.PREDICTED_SALES.iloc[0]
        SALES_VALUE_CURR_YEAR[i] = DATAF_QTR_FIL.SALES_VALUE.iloc[0]
    else:
        SALES_VOL_CURR_YEAR_ACTUAL[i] = -1
        SALES_VOL_CURR_YEAR_PRED[i] = -1
        SALES_VALUE_CURR_YEAR[i] = -1
        
        
        
    
DATAF_ALL_SALES_QTR_GROWTH.insert(8, "SALES_VOL_PREV_YEAR_ACTUAL", SALES_VOL_PREV_YEAR_ACTUAL)
DATAF_ALL_SALES_QTR_GROWTH.insert(8, "SALES_VOL_CURR_YEAR_ACTUAL", SALES_VOL_CURR_YEAR_ACTUAL)   
DATAF_ALL_SALES_QTR_GROWTH.insert(8, "SALES_VOL_PREV_YEAR_PRED", SALES_VOL_PREV_YEAR_PRED)   
DATAF_ALL_SALES_QTR_GROWTH.insert(8, "SALES_VOL_CURR_YEAR_PRED", SALES_VOL_CURR_YEAR_PRED)   
DATAF_ALL_SALES_QTR_GROWTH.insert(8, "SALES_VALUE_PREV_YEAR", SALES_VALUE_PREV_YEAR)   
DATAF_ALL_SALES_QTR_GROWTH.insert(8, "SALES_VALUE_CURR_YEAR", SALES_VALUE_CURR_YEAR)       
    

" Export "

filename = 'DATAF_ALL_TOTALSALES_QTR_GROWTH_INDIA_RERUN_WASHINGPOWDER.csv'

DATAF_ALL_SALES_QTR_GROWTH.to_csv(filename,index=False)
#DATAF_ALL_SALES_QTR.to_csv('DATAF_ALL_TOTALSALES_QTR_INDIA_RERUN.csv',index=False)


