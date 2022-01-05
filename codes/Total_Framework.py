


" Data Processing and  Modelling Framework "

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


print(" Data Acquisition ")

" File contains both total and granular level data "

DATAF_ALL_GRANULAR_TOTAL_all = pd.read_csv('CN_Batch2_NoB2C.csv')
subcats = [9]
all_data = pd.DataFrame()
roi_prices = pd.DataFrame()
for sub_cat in subcats:
    
    DATAF_ALL_GRANULAR_TOTAL = pd.DataFrame()
    DATAF_ALL_HIST = pd.DataFrame()
    DATAF_ALL_HIST_PROJ = pd.DataFrame()
    DATAF_ALL_HIST_PROJ_SALES = pd.DataFrame()
    
    DATAF_ALL_GRANULAR_TOTAL = DATAF_ALL_GRANULAR_TOTAL_all[DATAF_ALL_GRANULAR_TOTAL_all.SUBCAT_CD == sub_cat]
    
    " Filtering Total level data "
    
    DATAF_ALL_HIST = DATAF_ALL_GRANULAR_TOTAL[(DATAF_ALL_GRANULAR_TOTAL.CHNL_CD==0) & (DATAF_ALL_GRANULAR_TOTAL.FMT_CD==0) & (DATAF_ALL_GRANULAR_TOTAL.UL_REGION_NAME=='Country')]
    
    " Selecting Baseline "
    
    DATAF_ALL_HIST = DATAF_ALL_HIST[(DATAF_ALL_HIST.SECTOR_SCENARIO_CD==1)]
    
    DATAF_SUBCAT_LIST = pd.read_csv('CatSubcatFormat.csv')
    
    DATAF_CHNL_LIST = pd.read_csv('Channel.csv')
    
    SCENARIO_NO = 1
    
    
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
    
    " Enter the directory path "
    
    path = os.getcwd()
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " Enter dates according to the data "
    
    Start_month = 1
    Start_year = 2019
    
    End_month = 6
    End_year = 2021
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    
    print(" Adding blank/zeros forecast dates rows ")
    
    " For combinations "
    
    DATAF_ALL_START = DATAF_ALL_HIST[(DATAF_ALL_HIST.YEAR==Start_year) & (DATAF_ALL_HIST.MONTH==Start_month)]
    
    s = DATAF_ALL_START.shape[0]
    
    DATAF_PROJ_EMPTY = pd.read_csv('DATAF_PROJ_EMPTY.csv')
    
    for i in range(s):
        REG = DATAF_ALL_START.UL_GEO_ID.iloc[i]
        REG_NAME = DATAF_ALL_START.UL_REGION_NAME.iloc[i]
        CATG = DATAF_ALL_START.CATG_CD.iloc[i]
        CHNL = DATAF_ALL_START.CHNL_CD.iloc[i]
        FMT = DATAF_ALL_START.FMT_CD.iloc[i]
        SUBCAT = DATAF_ALL_START.SUBCAT_CD.iloc[i]
        
        DATAF_PROJ_EMPTY_COPY = DATAF_PROJ_EMPTY.copy()
        DATAF_PROJ_EMPTY_COPY.UL_GEO_ID = REG
        DATAF_PROJ_EMPTY_COPY.UL_REGION_NAME = REG_NAME
        DATAF_PROJ_EMPTY_COPY.CATG_CD = CATG
        DATAF_PROJ_EMPTY_COPY.CHNL_CD = CHNL
        DATAF_PROJ_EMPTY_COPY.FMT_CD = FMT
        DATAF_PROJ_EMPTY_COPY.SUBCAT_CD = SUBCAT
        
        if i==0:
            DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_PROJ_EMPTY_COPY])
        else:
            DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST_PROJ,DATAF_PROJ_EMPTY_COPY])
    
    
    
    DATAF_ALL_HIST_PROJ = DATAF_ALL_HIST_PROJ.reset_index(drop=True)
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    print(" Adding Channel/Format names ")
    
    s = DATAF_ALL_HIST_PROJ.shape[0]
    
    CHNL_NAME = ["" for i in range(s)]
    
    FMT_NAME = ["" for i in range(s)]
    
    SUBCAT_NAME = ["" for i in range(s)]
    
    for i in range(s):
        CHNL = DATAF_ALL_HIST_PROJ['CHNL_CD'].iloc[i]
        FMT = DATAF_ALL_HIST_PROJ['FMT_CD'].iloc[i]
        
        if CHNL == 0:
            CHNL_NAME[i] = "NO CHANNEL"
        else:
            DATAF_FIL = DATAF_CHNL_LIST[(DATAF_CHNL_LIST.CHNL_CD==CHNL)]
            CHNL_NAME[i] = DATAF_FIL.CHNL_DESC.iloc[0]
            
        if FMT == 0:
            FMT_NAME[i] = "NO FORMAT"
        else:
            DATAF_FIL = DATAF_SUBCAT_LIST[(DATAF_SUBCAT_LIST.FMT_CD==FMT)]
            FMT_NAME[i] = DATAF_FIL.FMT_DESC.iloc[0]    
        
        SUBCAT = DATAF_ALL_HIST_PROJ['SUBCAT_CD'].iloc[i]
        DATAF_FIL = DATAF_SUBCAT_LIST[(DATAF_SUBCAT_LIST.SUBCAT_CD==SUBCAT)]
        SUBCAT_NAME[i] = DATAF_FIL.iloc[0,3]
        
    
    
    DATAF_ALL_HIST_PROJ.insert(2, "SUBCAT_NAME",SUBCAT_NAME)
    DATAF_ALL_HIST_PROJ.insert(3, "CHANNEL_NAME",CHNL_NAME)
    DATAF_ALL_HIST_PROJ.insert(4, "FORMAT_NAME",FMT_NAME)
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " Precovid sales trend calculation and additional input creation "
    " This section is a new one to calculate an extra input related to precovid sales trend"
    
    SUBCAT_LIST = DATAF_ALL_HIST_PROJ.SUBCAT_CD.unique()
    
    for d in SUBCAT_LIST:
        
        DATAF_ALL_FIL = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.SUBCAT_CD==d)]
        
        s = DATAF_ALL_FIL.shape[0]
        
        COUNT_CAL = np.zeros((s), dtype=int)
    
        for i in range(s):
            COUNT_CAL[i] = i+1
        
        
        DATAF_ALL_FIL.insert(2, "COUNT_CAL",COUNT_CAL)
        
        DATAF_PRECOVID = DATAF_ALL_FIL[(DATAF_ALL_FIL['YEAR']<2020)]
        
        x = DATAF_PRECOVID['COUNT_CAL'].values
        y = DATAF_PRECOVID['SALES_VOLUME'].values
        
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        
        trend_model = LinearRegression().fit(x, y)
        C = trend_model.coef_
        I = trend_model.intercept_
        SALES_TREND_CAL = np.zeros((s), dtype=float)
    
        for i in range(s):
            SALES_TREND_CAL[i] = (C*COUNT_CAL[i])+I
        
        
        DATAF_ALL_FIL.insert(2, "SALES_TREND_CAL",SALES_TREND_CAL)
        if d==SUBCAT_LIST[0]:
            DATAF_ALL_HIST_PROJ_UPD = DATAF_ALL_FIL
        else:
            DATAF_ALL_HIST_PROJ_UPD = pd.concat([DATAF_ALL_HIST_PROJ_UPD,DATAF_ALL_FIL])
        
    
    
    DATAF_ALL_HIST_PROJ = DATAF_ALL_HIST_PROJ_UPD.copy()
    DATAF_ALL_HIST_PROJ.drop(columns = 'COUNT_CAL',inplace = True)
    "-----------------------------------------------------------------------------------------------------------------------------"   
    
    
    
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " Precovid tdp trend calculation and additional input creation "
    " This section is a new one to calculate an extra input related to precovid tdp trend"
    
    SUBCAT_LIST = DATAF_ALL_HIST_PROJ.SUBCAT_CD.unique()
    
    for d in SUBCAT_LIST:
        
        DATAF_ALL_FIL = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.SUBCAT_CD==d)]
        
        s = DATAF_ALL_FIL.shape[0]
        
        COUNT_CAL = np.zeros((s), dtype=int)
    
        for i in range(s):
            COUNT_CAL[i] = i+1
        
        
        DATAF_ALL_FIL.insert(2, "COUNT_CAL",COUNT_CAL)
        
        DATAF_PRECOVID = DATAF_ALL_FIL[(DATAF_ALL_FIL['YEAR']<2020)]
        
        x = DATAF_PRECOVID['COUNT_CAL'].values
        y = DATAF_PRECOVID['TDP'].values
        
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        
        trend_model = LinearRegression().fit(x, y)
        C = trend_model.coef_
        I = trend_model.intercept_
        TDP_TREND_CAL = np.zeros((s), dtype=float)
    
        for i in range(s):
            TDP_TREND_CAL[i] = (C*COUNT_CAL[i])+I
        
        
        DATAF_ALL_FIL.insert(2, "TDP_TREND_CAL",TDP_TREND_CAL)
        if d==SUBCAT_LIST[0]:
            DATAF_ALL_HIST_PROJ_UPD = DATAF_ALL_FIL
        else:
            DATAF_ALL_HIST_PROJ_UPD = pd.concat([DATAF_ALL_HIST_PROJ_UPD,DATAF_ALL_FIL])
        
    
    
    DATAF_ALL_HIST_PROJ = DATAF_ALL_HIST_PROJ_UPD.copy()
    
    "-----------------------------------------------------------------------------------------------------------------------------"   
    
    
    
    
    
    print(" Price Seasonality Index Computation and Addition Method New")
    
    DATAF_ALL_HIST_PRECOVID = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.YEAR==2018) | (DATAF_ALL_HIST_PROJ.YEAR==2019)]
    
    DATAF_ALL_START = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.YEAR==Start_year) & (DATAF_ALL_HIST_PROJ.MONTH==Start_month)]
    
    s = DATAF_ALL_START.shape[0]
    
    for i in range(s):
        REG = DATAF_ALL_START.UL_GEO_ID.iloc[i]
        CHNL = DATAF_ALL_START.CHNL_CD.iloc[i]
        FMT = DATAF_ALL_START.FMT_CD.iloc[i]
        SUBCAT = DATAF_ALL_START.SUBCAT_CD.iloc[i]
        
        DATAF = DATAF_ALL_HIST_PRECOVID[(DATAF_ALL_HIST_PRECOVID.UL_GEO_ID==REG) & (DATAF_ALL_HIST_PRECOVID.CHNL_CD==CHNL) & (DATAF_ALL_HIST_PRECOVID.FMT_CD==FMT) & (DATAF_ALL_HIST_PRECOVID.SUBCAT_CD==SUBCAT)]
        
        DATAF_2018 = DATAF[(DATAF.YEAR==2018)]
    
        DATAF_2019 = DATAF[(DATAF.YEAR==2019)]
        
        AVG_PRICE_2018 = np.mean(DATAF_2018.PRICE_PER_VOL)
        
        AVG_PRICE_2019 = np.mean(DATAF_2019.PRICE_PER_VOL)
        
        p = DATAF.shape[0]
    
        SEASONALITY_INDEX_PRICE_TEMP = np.zeros((p), dtype=float)
        
        for j in range(p):
            YEAR = DATAF.YEAR.iloc[j]
            
            if YEAR==2018:
                SEASONALITY_INDEX_PRICE_TEMP[j] = DATAF.PRICE_PER_VOL.iloc[j]/AVG_PRICE_2018
            elif YEAR==2019:
                SEASONALITY_INDEX_PRICE_TEMP[j] = DATAF.PRICE_PER_VOL.iloc[j]/AVG_PRICE_2019
                
                
        DATAF.insert(2, "SEASONALITY_INDEX_PRICE_TEMP",SEASONALITY_INDEX_PRICE_TEMP)
        SEASONALITY_INDEX_PRICE = np.zeros((12), dtype=float)  
        q = 12
        
        for k in range(q):
            DATAF_FILT = DATAF[(DATAF.MONTH==k+1)]
            if DATAF_FILT.shape[0]==1:
                SEASONALITY_INDEX_PRICE[k] = DATAF_FILT.SEASONALITY_INDEX_PRICE_TEMP
            elif DATAF_FILT.shape[0]==2:
                SEASONALITY_INDEX_PRICE[k] = np.mean(DATAF_FILT.SEASONALITY_INDEX_PRICE_TEMP)
        
        
        
        DATAF_SEASONALITY = DATAF[(DATAF.YEAR==2019)]
        
        COL = ['UL_GEO_ID','CHNL_CD','FMT_CD','SUBCAT_CD','MONTH','SEASONALITY_INDEX_PRICE_TEMP']
        
        DATAF_SEASONALITY = DATAF_SEASONALITY[COL]
        
        del DATAF_SEASONALITY["SEASONALITY_INDEX_PRICE_TEMP"]
        
        DATAF_SEASONALITY.insert(DATAF_SEASONALITY.shape[1], "SEASONALITY_INDEX_PRICE",SEASONALITY_INDEX_PRICE)
        
        if i==0:
            DATAF_ALL_SEASONALITY = DATAF_SEASONALITY
        else:
            DATAF_ALL_SEASONALITY = pd.concat([DATAF_ALL_SEASONALITY,DATAF_SEASONALITY])
            
    
    "-----------------------------------------------------------------------------------------------------------------------------"        
           
    " Incorporating Seasonality in the Dataframe "
    
    s = DATAF_ALL_HIST_PROJ.shape[0]
    
    SEASONALITY_INDEX_PRICE = np.zeros((s), dtype=float)
    
    DATAF_ALL_HIST_PROJ.insert(2, "SEASONALITY_INDEX_PRICE", SEASONALITY_INDEX_PRICE)
    
    for i in range(s):
        REG = DATAF_ALL_HIST_PROJ.UL_GEO_ID.iloc[i]
        CHNL = DATAF_ALL_HIST_PROJ.CHNL_CD.iloc[i]
        FMT = DATAF_ALL_HIST_PROJ.FMT_CD.iloc[i]
        SUBCAT = DATAF_ALL_HIST_PROJ.SUBCAT_CD.iloc[i]
        MONTH = DATAF_ALL_HIST_PROJ.MONTH.iloc[i]
        
        Seasonality_Value = DATAF_ALL_SEASONALITY[(DATAF_ALL_SEASONALITY.UL_GEO_ID==REG) & (DATAF_ALL_SEASONALITY.CHNL_CD==CHNL) 
        & (DATAF_ALL_SEASONALITY.FMT_CD==FMT) & (DATAF_ALL_SEASONALITY.SUBCAT_CD==SUBCAT) & (DATAF_ALL_SEASONALITY.MONTH==MONTH)].SEASONALITY_INDEX_PRICE.values
        
        DATAF_ALL_HIST_PROJ.SEASONALITY_INDEX_PRICE.iloc[i] = Seasonality_Value
        
        
    "-----------------------------------------------------------------------------------------------------------------------------"   
          
    print(" TDP Seasonality Index Computation and Addition Method New")
    
    DATAF_ALL_HIST_PRECOVID = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.YEAR==2018) | (DATAF_ALL_HIST_PROJ.YEAR==2019)]
    
    DATAF_ALL_START = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.YEAR==Start_year) & (DATAF_ALL_HIST_PROJ.MONTH==Start_month)]
    
    s = DATAF_ALL_START.shape[0]
    
    for i in range(s):
        REG = DATAF_ALL_START.UL_GEO_ID.iloc[i]
        CHNL = DATAF_ALL_START.CHNL_CD.iloc[i]
        FMT = DATAF_ALL_START.FMT_CD.iloc[i]
        SUBCAT = DATAF_ALL_START.SUBCAT_CD.iloc[i]
        
        DATAF = DATAF_ALL_HIST_PRECOVID[(DATAF_ALL_HIST_PRECOVID.UL_GEO_ID==REG) & (DATAF_ALL_HIST_PRECOVID.CHNL_CD==CHNL) & (DATAF_ALL_HIST_PRECOVID.FMT_CD==FMT) & (DATAF_ALL_HIST_PRECOVID.SUBCAT_CD==SUBCAT)]
        
        DATAF_2018 = DATAF[(DATAF.YEAR==2018)]
    
        DATAF_2019 = DATAF[(DATAF.YEAR==2019)]
        
        AVG_TDP_2018 = np.mean(DATAF_2018.TDP)
        
        AVG_TDP_2019 = np.mean(DATAF_2019.TDP)
        
        p = DATAF.shape[0]
    
        SEASONALITY_INDEX_TDP_TEMP = np.zeros((p), dtype=float)
        
        for j in range(p):
            YEAR = DATAF.YEAR.iloc[j]
            
            if YEAR==2018:
                SEASONALITY_INDEX_TDP_TEMP[j] = DATAF.TDP.iloc[j]/AVG_TDP_2018
            elif YEAR==2019:
                SEASONALITY_INDEX_TDP_TEMP[j] = DATAF.TDP.iloc[j]/AVG_TDP_2019
                
                
        DATAF.insert(2, "SEASONALITY_INDEX_TDP_TEMP",SEASONALITY_INDEX_TDP_TEMP)
        SEASONALITY_INDEX_TDP = np.zeros((12), dtype=float)  
        q = 12
        
        for k in range(q):
            DATAF_FILT = DATAF[(DATAF.MONTH==k+1)]
            if DATAF_FILT.shape[0]==1:
                SEASONALITY_INDEX_TDP[k] = DATAF_FILT.SEASONALITY_INDEX_TDP_TEMP
            elif DATAF_FILT.shape[0]==2:
                SEASONALITY_INDEX_TDP[k] = np.mean(DATAF_FILT.SEASONALITY_INDEX_TDP_TEMP)
        
        
        
        DATAF_SEASONALITY = DATAF[(DATAF.YEAR==2019)]
        
        COL = ['UL_GEO_ID','CHNL_CD','FMT_CD','SUBCAT_CD','MONTH','SEASONALITY_INDEX_TDP_TEMP']
        
        DATAF_SEASONALITY = DATAF_SEASONALITY[COL]
        
        del DATAF_SEASONALITY["SEASONALITY_INDEX_TDP_TEMP"]
        
        DATAF_SEASONALITY.insert(DATAF_SEASONALITY.shape[1], "SEASONALITY_INDEX_TDP",SEASONALITY_INDEX_TDP)
        
        if i==0:
            DATAF_ALL_SEASONALITY = DATAF_SEASONALITY
        else:
            DATAF_ALL_SEASONALITY = pd.concat([DATAF_ALL_SEASONALITY,DATAF_SEASONALITY])
            
    
    "-----------------------------------------------------------------------------------------------------------------------------"        
    
           
    " Incorporating Seasonality in the Dataframe "
    
    s = DATAF_ALL_HIST_PROJ.shape[0]
    
    SEASONALITY_INDEX_TDP = np.zeros((s), dtype=float)
    
    DATAF_ALL_HIST_PROJ.insert(2, "SEASONALITY_INDEX_TDP", SEASONALITY_INDEX_TDP)
    
    for i in range(s):
        REG = DATAF_ALL_HIST_PROJ.UL_GEO_ID.iloc[i]
        CHNL = DATAF_ALL_HIST_PROJ.CHNL_CD.iloc[i]
        FMT = DATAF_ALL_HIST_PROJ.FMT_CD.iloc[i]
        SUBCAT = DATAF_ALL_HIST_PROJ.SUBCAT_CD.iloc[i]
        MONTH = DATAF_ALL_HIST_PROJ.MONTH.iloc[i]
        
        Seasonality_Value = DATAF_ALL_SEASONALITY[(DATAF_ALL_SEASONALITY.UL_GEO_ID==REG) & (DATAF_ALL_SEASONALITY.CHNL_CD==CHNL) 
        & (DATAF_ALL_SEASONALITY.FMT_CD==FMT) & (DATAF_ALL_SEASONALITY.SUBCAT_CD==SUBCAT) & (DATAF_ALL_SEASONALITY.MONTH==MONTH)].SEASONALITY_INDEX_TDP.values
        
        DATAF_ALL_HIST_PROJ.SEASONALITY_INDEX_TDP.iloc[i] = Seasonality_Value
    
    "-----------------------------------------------------------------------------------------------------------------------------"   
    
    print(" Sales Seasonality Index Computation and Addition Method New")
    
    DATAF_ALL_HIST_PRECOVID = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.YEAR==2018) | (DATAF_ALL_HIST_PROJ.YEAR==2019)]
    
    DATAF_ALL_START = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.YEAR==Start_year) & (DATAF_ALL_HIST_PROJ.MONTH==Start_month)]
    
    s = DATAF_ALL_START.shape[0]
    
    for i in range(s):
        REG = DATAF_ALL_START.UL_GEO_ID.iloc[i]
        CHNL = DATAF_ALL_START.CHNL_CD.iloc[i]
        FMT = DATAF_ALL_START.FMT_CD.iloc[i]
        SUBCAT = DATAF_ALL_START.SUBCAT_CD.iloc[i]
        
        DATAF = DATAF_ALL_HIST_PRECOVID[(DATAF_ALL_HIST_PRECOVID.UL_GEO_ID==REG) & (DATAF_ALL_HIST_PRECOVID.CHNL_CD==CHNL) & (DATAF_ALL_HIST_PRECOVID.FMT_CD==FMT) & (DATAF_ALL_HIST_PRECOVID.SUBCAT_CD==SUBCAT)]
        
        DATAF_2018 = DATAF[(DATAF.YEAR==2018)]
    
        DATAF_2019 = DATAF[(DATAF.YEAR==2019)]
        
        AVG_SALES_2018 = np.mean(DATAF_2018.SALES_VOLUME)
        
        AVG_SALES_2019 = np.mean(DATAF_2019.SALES_VOLUME)
        
        p = DATAF.shape[0]
    
        SEASONALITY_INDEX_SALES_TEMP = np.zeros((p), dtype=float)
        
        for j in range(p):
            YEAR = DATAF.YEAR.iloc[j]
            
            if YEAR==2018:
                SEASONALITY_INDEX_SALES_TEMP[j] = DATAF.SALES_VOLUME.iloc[j]/AVG_SALES_2018
            elif YEAR==2019:
                SEASONALITY_INDEX_SALES_TEMP[j] = DATAF.SALES_VOLUME.iloc[j]/AVG_SALES_2019
                
                
        DATAF.insert(2, "SEASONALITY_INDEX_SALES_TEMP",SEASONALITY_INDEX_SALES_TEMP)
        SEASONALITY_INDEX_SALES = np.zeros((12), dtype=float)  
        q = 12
        
        for k in range(q):
            DATAF_FILT = DATAF[(DATAF.MONTH==k+1)]
            if DATAF_FILT.shape[0]==1:
                SEASONALITY_INDEX_SALES[k] = DATAF_FILT.SEASONALITY_INDEX_SALES_TEMP
            elif DATAF_FILT.shape[0]==2:
                SEASONALITY_INDEX_SALES[k] = np.mean(DATAF_FILT.SEASONALITY_INDEX_SALES_TEMP)
        
        
        
        DATAF_SEASONALITY = DATAF[(DATAF.YEAR==2019)]
        
        COL = ['UL_GEO_ID','CHNL_CD','FMT_CD','SUBCAT_CD','MONTH','SEASONALITY_INDEX_SALES_TEMP']
        
        DATAF_SEASONALITY = DATAF_SEASONALITY[COL]
        
        del DATAF_SEASONALITY["SEASONALITY_INDEX_SALES_TEMP"]
        
        DATAF_SEASONALITY.insert(DATAF_SEASONALITY.shape[1], "SEASONALITY_INDEX_SALES",SEASONALITY_INDEX_SALES)
        
        if i==0:
            DATAF_ALL_SEASONALITY = DATAF_SEASONALITY
        else:
            DATAF_ALL_SEASONALITY = pd.concat([DATAF_ALL_SEASONALITY,DATAF_SEASONALITY])
            
    
    "-----------------------------------------------------------------------------------------------------------------------------"        
    
    " Incorporating Seasonality in the Dataframe "
    
    s = DATAF_ALL_HIST_PROJ.shape[0]
    
    SEASONALITY_INDEX_SALES = np.zeros((s), dtype=float)
    
    DATAF_ALL_HIST_PROJ.insert(2, "SEASONALITY_INDEX_SALES", SEASONALITY_INDEX_SALES)
    
    for i in range(s):
        REG = DATAF_ALL_HIST_PROJ.UL_GEO_ID.iloc[i]
        CHNL = DATAF_ALL_HIST_PROJ.CHNL_CD.iloc[i]
        FMT = DATAF_ALL_HIST_PROJ.FMT_CD.iloc[i]
        SUBCAT = DATAF_ALL_HIST_PROJ.SUBCAT_CD.iloc[i]
        MONTH = DATAF_ALL_HIST_PROJ.MONTH.iloc[i]
        
        Seasonality_Value = DATAF_ALL_SEASONALITY[(DATAF_ALL_SEASONALITY.UL_GEO_ID==REG) & (DATAF_ALL_SEASONALITY.CHNL_CD==CHNL) 
        & (DATAF_ALL_SEASONALITY.FMT_CD==FMT) & (DATAF_ALL_SEASONALITY.SUBCAT_CD==SUBCAT) & (DATAF_ALL_SEASONALITY.MONTH==MONTH)].SEASONALITY_INDEX_SALES.values
        
        DATAF_ALL_HIST_PROJ.SEASONALITY_INDEX_SALES.iloc[i] = Seasonality_Value
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    " Adding future numbers for COVID cases and deaths "
    
    DATAF_COVID_NOS = pd.read_excel('china covid.xlsx')
    
    " Filtering Historic & Forecast Period "
    DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
    DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]
    
    s = DATAF_ALL_PROJ.shape[0]
    
    " Row wise addition using the for loop "
    
    for i in range(s):
        Year = DATAF_ALL_PROJ.YEAR.iloc[i]
        Month = DATAF_ALL_PROJ.MONTH.iloc[i]
        Reg = DATAF_ALL_PROJ.UL_GEO_ID.iloc[i]
        
        DATAF_COVID_FIL = DATAF_COVID_NOS[(DATAF_COVID_NOS['YEAR']==Year) & (DATAF_COVID_NOS['MONTH']==Month) & (DATAF_COVID_NOS['UL_GEO_ID']==Reg)]
        
        DATAF_ALL_PROJ['SUM(NEW_CASES)'].iloc[i] = DATAF_COVID_FIL.NEW_CASES_BASELINE.iloc[0]
        DATAF_ALL_PROJ['SUM(NEW_DEATHS)'].iloc[i] = DATAF_COVID_FIL.NEW_DEATHS_BASELINE.iloc[0]
        
      
        
    DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])  
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    print(" Adding baseline forecasts for Macro-economic variables ")
    
    DATAF_ALL_MACRO = pd.read_csv('Macro.csv')
    
    DATAF_ALL_MACRO = DATAF_ALL_MACRO[(DATAF_ALL_MACRO['SECTOR_SCENARIO_CD']==1)]
    
    s = DATAF_ALL_HIST_PROJ.shape[0]
    
    for i in range(s):
        YEAR = DATAF_ALL_HIST_PROJ.YEAR.iloc[i]
        MONTH = DATAF_ALL_HIST_PROJ.MONTH.iloc[i]
        
        DATAF_MACRO_TEMP = DATAF_ALL_MACRO[(DATAF_ALL_MACRO.YEAR==YEAR) & (DATAF_ALL_MACRO.MONTH==MONTH) & (DATAF_ALL_MACRO.SECTOR_SCENARIO_CD==1)]
        DATAF_ALL_HIST_PROJ['AVG(CONSUMER_PRICE_INDEX)'].iloc[i] = DATAF_MACRO_TEMP.CONSUMER_PRICE_INDEX.iloc[0]
        DATAF_ALL_HIST_PROJ['AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)'].iloc[i] = DATAF_MACRO_TEMP.PERSONAL_DISPOSABLE_INCOME_REAL_LCU.iloc[0]
        DATAF_ALL_HIST_PROJ['AVG(GDP_NOMINAL_LCU)'].iloc[i] = DATAF_MACRO_TEMP.GDP_NOMINAL_LCU.iloc[0]
        DATAF_ALL_HIST_PROJ['AVG(GDP_REAL_LCU)'].iloc[i] = DATAF_MACRO_TEMP.GDP_REAL_LCU.iloc[0]
        DATAF_ALL_HIST_PROJ['AVG(RETAIL_PRICES_INDEX)'].iloc[i] = DATAF_MACRO_TEMP.RETAIL_PRICES_INDEX.iloc[0]
        DATAF_ALL_HIST_PROJ['AVG(SHARE_PRICE_INDEX)'].iloc[i] = DATAF_MACRO_TEMP.SHARE_PRICE_INDEX.iloc[0]
        DATAF_ALL_HIST_PROJ['AVG(UNEMP_RATE)'].iloc[i] = DATAF_MACRO_TEMP.UNEMP_RATE.iloc[0]
        
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    print(" Adding forecasts for weather variables ")
    
    DATAF_ALL_weather = pd.read_csv('china_weather_data.csv')
    
    s = DATAF_ALL_HIST_PROJ.shape[0]
    
    for i in range(s):
        YEAR = DATAF_ALL_HIST_PROJ.YEAR.iloc[i]
        MONTH = DATAF_ALL_HIST_PROJ.MONTH.iloc[i]
        
        DATAF_weather_TEMP = DATAF_ALL_weather[(DATAF_ALL_weather.YEAR==YEAR) & (DATAF_ALL_weather.MONTH==MONTH)]
        DATAF_ALL_HIST_PROJ['AVG(AVG_TEMP_CELSIUS)'].iloc[i] = DATAF_weather_TEMP["AVG(AVG_TEMP_CELSIUS)"].iloc[0]
        DATAF_ALL_HIST_PROJ['AVG(HUMID_PCT)'].iloc[i] = DATAF_weather_TEMP["AVG(HUMID_PCT)"].iloc[0]
        
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    print(" Adding forecasts for baseline mobility variables ")
    
    DATAF_ALL_mobility = pd.read_csv('china_traffic_mobility_baseline.csv')
    
    s = DATAF_ALL_HIST_PROJ.shape[0]
    
    for i in range(s):
        YEAR = DATAF_ALL_HIST_PROJ.YEAR.iloc[i]
        MONTH = DATAF_ALL_HIST_PROJ.MONTH.iloc[i]
        
        DATAF_mobility_TEMP = DATAF_ALL_mobility[(DATAF_ALL_mobility.YEAR==YEAR) & (DATAF_ALL_mobility.MONTH==MONTH)]
        DATAF_ALL_HIST_PROJ['AVG(TRAFFIC_WEIGHT)'].iloc[i] = DATAF_mobility_TEMP["TRAFFIC_WEIGHT"].iloc[0]
        
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    
    
    " Price Predictive Modelling "
    " Filtering Historic & Forecast Period "
    DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
    DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]
    
    runfile(path+'/PRICE_FORECASTING_MODELLING_TOTAL.py', wdir=path)
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " Price Forecast "
    
    " Filtering Historic & Forecast Period "
    DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
    DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]
    
    runfile(path+'/PRICE_PREDICTION_FUNCTION_TOTAL.py', wdir=path)
    
    DATAF_ALL_PROJ = Price_Prediction(DATAF_ALL_PROJ,price_input_col)
    
    DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])
    
    DATAF_ALL_HIST_PROJ.to_csv('DATAF_ALL_HIST_PROJ_PRICE.csv',index=False)
    
    
    
    
    
    
    "-----------------------------------------------------------------------------------------------------------------------------"  
    import pandas as pd
    import openpyxl
    import win32com.client
    #df1 = pd.read_excel(r'C:\Users\91900\Downloads\China_Price_Assumption_bc_1.xlsx',engine='openpyxl')
    df_price_assum = pd.read_csv('price_assumption_input.csv')
    df_baseline_granular = DATAF_ALL_HIST_PROJ
    
    def input_price_assumption(df=df_price_assum,SUBCAT_CD=2, CATG_CD=1):
        year = df[(df['CATG_CD']==CATG_CD) & (df['SUBCAT_CD']==SUBCAT_CD)]['Year'].values
        half_year = df[(df['CATG_CD']==CATG_CD) & (df['SUBCAT_CD']==SUBCAT_CD)]['H1'].values
        return year, half_year
    
    
    price_assump = []
    
    for i in df_baseline_granular.CATG_CD.unique():
        for j in df_baseline_granular.SUBCAT_CD.unique():
            for k in df_baseline_granular.CHNL_CD.unique():
                for l in df_baseline_granular.FMT_CD.unique():
                    
                    test = input_price_assumption(df_price_assum,j,i)
                    
                    wb = openpyxl.load_workbook('China_Price_Assumption_bc_1.xlsx')
                    sheet = wb.active
                    df_use = df_baseline_granular[(df_baseline_granular['CATG_CD']==i) & (df_baseline_granular['SUBCAT_CD']==j) & (df_baseline_granular['CHNL_CD']==k) & (df_baseline_granular['FMT_CD']==l)].reset_index(drop=False)[['PERIOD_BEGIN_DATE', 'YEAR', 'MONTH','PRICE_PER_VOL','SUBCAT_CD','UL_GEO_ID','CHNL_CD', 'FMT_CD']]
                    for m in range(15,69):
                        sheet.cell(row=m,column=5).value = df_use['PRICE_PER_VOL'][m-15]
        
                    sheet['J2'].value = test[1][0]
                    sheet['K2'].value = test[1][0]
                    sheet['L2'].value = 2*test[0][0]-test[1][0]
                    sheet['M2'].value = 2*test[0][0]-test[1][0]
                    wb.save(filename= 'China_Price_Assumption_bc_1.xlsx')
                    
                    # Start an instance of Excel
                    xlapp = win32com.client.DispatchEx("Excel.Application")
    
                    # Open the workbook in said instance of Excel
                    wb = xlapp.workbooks.open(path +'\China_Price_Assumption_bc_1.xlsx')
    
                    # Optional, e.g. if you want to debug
                    # xlapp.Visible = True
    
                    # Refresh all data connections.
                    wb.RefreshAll()
                    wb.Save()
    
                    # Quit
                    xlapp.Quit()
                    
                    df_use['predicted_PRICE_PER_VOL_adjusted'] = pd.read_excel('China_Price_Assumption_bc_1.xlsx',engine='openpyxl')['Price Assumptions'][13:67].to_list()
                    xlapp.Quit()
                    price_assump.append(df_use[['PERIOD_BEGIN_DATE', 'YEAR', 'MONTH','predicted_PRICE_PER_VOL_adjusted','SUBCAT_CD','UL_GEO_ID','CHNL_CD', 'FMT_CD']])
                    
                    
    df_final_pa = pd.concat(price_assump)
    file_name_price = 'roi_prices_for_'+str(DATAF_ALL_HIST_PROJ.SUBCAT_NAME[0])+'.csv'
    df_final_pa.to_csv(file_name_price,index =False)
    
    roi_prices = pd.concat([roi_prices,df_final_pa])
        
    "-----------------------------------------------------------------------------------------------------------------------------"  
        
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " TDP Predictive Modelling "
    
    " Filtering Historic & Forecast Period "
    DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
    DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]
    
    runfile(path+'/TDP_FORECASTING_MODELLING_TOTAL.py', wdir=path)
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " TDP Forecast "
    
    " Filtering Historic & Forecast Period "
    DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
    DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]
    
    
    runfile(path+'/TDP_PREDICTION_FUNCTION_TOTAL.py', wdir=path)
    
    DATAF_ALL_PROJ = TDP_Prediction(DATAF_ALL_PROJ,tdp_input_col)
    
    DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ])
    
    DATAF_ALL_HIST_PROJ.to_csv('DATAF_ALL_HIST_PROJ_TDP.csv',index=False)
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " Preference Predictive Modelling "
    " Preference value not applicable for total level modelling. Hence, adding constant value of 1 "
    
    DATAF_ALL_HIST_PROJ.insert(DATAF_ALL_HIST_PROJ.columns.get_loc("SALES_VOLUME")+1, "PREF_VALUE", 1)
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " Sales Modelling and Forecasting " 
    
    w1 = 0.95
    w2 = 0.05
    
    " Filtering Historic & Forecast Period "
    DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
    DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]
    
    " Sales Model with Trend as additional input "
    
    runfile(path+'/SALES_FORECASTING_MODELLING_3.py', wdir=path)
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " Shapley values calculation and export"
    
    runfile(path+'/SHAP_VALUES_TOTAL.py', wdir=path)
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " Rearranging columns in the DATAF_ALL_HIST_PROJ_SALES dataframe "
    
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
           'SEASONALITY_INDEX_PRICE', 'SALES_TREND_CAL','TDP_TREND_CAL']
    
    DATAF_ALL_HIST_PROJ_SALES = DATAF_ALL_HIST_PROJ_SALES.reindex(columns=column_names)
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " Renaming some of the variables names i.e. removing AVG() and SUM() from the names "
    " Following code may have to be modified if the variables changes "
    
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(MAX_TEMP_CELSIUS)':'MAX_TEMP_CELSIUS'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(AVG_TEMP_CELSIUS)':'AVG_TEMP_CELSIUS'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(MIN_TEMP_CELSIUS)':'MIN_TEMP_CELSIUS'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(FEELS_LIKE_CELSIUS)':'FEELS_LIKE_CELSIUS'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(AVG_WIND_MPH)':'AVG_WIND_MPH'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(HUMID_PCT)':'HUMID_PCT'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(PRECIP_MM)':'PRECIP_MM'}, inplace = True)
    #DATAF_ALL_HIST_PROJ.rename(columns = {'AVG(SNOW_CM)':'SNOW_CM'}, inplace = True)
    #
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'SUM(NEW_CASES)':'NEW_CASES'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'SUM(NEW_DEATHS)':'NEW_DEATHS'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'SUM(CUMU_CASES)':'CUMU_CASES'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'SUM(CUMU_DEATHS)':'CUMU_DEATHS'}, inplace = True)
    #
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(RETAIL_AND_RECREATION_PCT_CHANGE)':'RETAIL_AND_RECREATION_PCT_CHANGE'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(GROCERY_AND_PHARMACY_PCT_CHANGE)':'GROCERY_AND_PHARMACY_PCT_CHANGE'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(PARKS_PCT_CHANGE)':'PARKS_PCT_CHANGE'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(TRANSIT_STATIONS_PCT_CHANGE)':'TRANSIT_STATIONS_PCT_CHANGE'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(WORKPLACES_PCT_CHANGE)':'WORKPLACES_PCT_CHANGE'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(RESIDENTIAL_PCT_CHANGE)':'RESIDENTIAL_PCT_CHANGE'}, inplace = True)
    #
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(CONSUMER_PRICE_INDEX)':'CONSUMER_PRICE_INDEX'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)':'PERSONAL_DISPOSABLE_INCOME_REAL_LCU'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(GDP_NOMINAL_LCU)':'GDP_NOMINAL_LCU'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(GDP_REAL_LCU)':'GDP_REAL_LCU'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(RETAIL_PRICES_INDEX)':'RETAIL_PRICES_INDEX'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(SHARE_PRICE_INDEX)':'SHARE_PRICE_INDEX'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(UNEMP_RATE)':'UNEMP_RATE'}, inplace = True)
    #
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(ANXIETY_CONCERNED_PCT)':'ANXIETY_CONCERNED_PCT'}, inplace = True)
    #DATAF_ALL_HIST_PROJ_SALES.rename(columns = {'AVG(CHANGE_SINCE_PREV_FORTNIGHT)':'CHANGE_SINCE_PREV_FORTNIGHT'}, inplace = True)
    
    "-----------------------------------------------------------------------------------------------------------------------------" 
    
    " Exporting the prediction and metrics file "
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
    DATAF_ALL_HIST_PROJ_SALES.to_csv('CHN_'+str(DATAF_ALL_HIST_PROJ.SUBCAT_NAME[0])+'_total_sales.csv',index=False)
    
    DATAF_METRICS.to_csv('CHN_'+str(DATAF_ALL_HIST_PROJ.SUBCAT_NAME[0])+'_total_TOTALMETRICS.csv',index=False)
    all_data = pd.concat([all_data,DATAF_ALL_HIST_PROJ_SALES])

    "-----------------------------------------------------------------------------------------------------------------------------" 

all_data.to_csv("china_baseline_rerun_for_seperate_online.csv")
roi_prices.to_csv("roi_prices_for_all_subcats.csv")