


import shap

SUBCAT_LIST = DATAF_ALL_HIST_PROJ.SUBCAT_CD.unique()

#def prediction(X):
#    ypred = loaded_model2(X)
#    return ypred

  
for c in SUBCAT_LIST:
    print("---------------------------------------------------------------------")
    print(c)
    
    
    DATAF = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.SUBCAT_CD==c)]
    
    INP_COL = sales_input_col
    X = DATAF[INP_COL].values
    
    filename1 = 'SALES_TOTAL_Prediction_XGB_'+str(c)
    
    filename2 = 'SALES_TOTAL_Prediction_LIN_'+str(c)
    
    #load the model from disk
    loaded_model1 = joblib.load(filename1)
    loaded_model2 = joblib.load(filename2)
    
    explainer1 = shap.TreeExplainer(loaded_model1, X)
    shap_values1 = explainer1.shap_values(X)
    
    # If Model 2 is Tree Based
    # check_additivity = False is required for Random Forest
    #explainer2 = shap.TreeExplainer(loaded_model2, X)
    #shap_values2 = explainer2.shap_values(X,check_additivity = False)
    
    explainer2 = shap.LinearExplainer(loaded_model2, X)
    shap_values2 = explainer2.shap_values(X)
    
    shap_values = (w1*shap_values1)+(w2*shap_values2)
    
    exp1 = explainer1.expected_value
    exp2 = explainer2.expected_value

    exp = (w1*exp1)+(w2*exp2)
    
    DATAF_SHAP = pd.DataFrame(data = shap_values, columns = INP_COL)
    
    DATAF_SHAP.insert(DATAF_SHAP.shape[1], "EXP_VALUE",exp)
    
    DATAF_SHAP.insert(0, "UL_GEO_ID",DATAF.UL_GEO_ID.values)
    DATAF_SHAP.insert(1, "UL_REGION_NAME",DATAF.UL_REGION_NAME.values)
    DATAF_SHAP.insert(2, "CHNL_CD",DATAF.CHNL_CD.values)
    DATAF_SHAP.insert(3, "FMT_CD",DATAF.FMT_CD.values)
    DATAF_SHAP.insert(4, "SUBCAT_CD",DATAF.SUBCAT_CD.values)
    DATAF_SHAP.insert(5, "CATG_CD",DATAF.CATG_CD.values)
    DATAF_SHAP.insert(6, "SECTOR_SCENARIO_CD",SCENARIO_NO)
    DATAF_SHAP.insert(7, "PERIOD_BEGIN_DATE",DATAF.PERIOD_BEGIN_DATE.values)
    DATAF_SHAP.insert(8, "YEAR",DATAF.YEAR.values)
    DATAF_SHAP.insert(9, "MONTH",DATAF.MONTH.values)
    

    if c==SUBCAT_LIST[0]:
        DATAF_SHAP_ALL = DATAF_SHAP
    else:
        DATAF_SHAP_ALL = pd.concat([DATAF_SHAP_ALL,DATAF_SHAP])
        

" Adding zeros column for additional variables not used in the model but to maintain the seauence "

s = DATAF_SHAP_ALL.shape[0]

" Following code may have to be modified if the input list variables changes "
#DATAF_SHAP_ALL.insert(DATAF_SHAP_ALL.columns.get_loc("TDP")+1, "PREF_VALUE",0)
#DATAF_SHAP_ALL.insert(DATAF_SHAP_ALL.columns.get_loc("AVG(AVG_TEMP_CELSIUS)")+1, "AVG(MIN_TEMP_CELSIUS)",0)
#DATAF_SHAP_ALL.insert(DATAF_SHAP_ALL.columns.get_loc("AVG(HUMID_PCT)")+1, "AVG(PRECIP_MM)",0)

" Renaming some of the variables names i.e. removing AVG() from the names "
" Following code may have to be modified if the input list variables changes "

#DATAF_SHAP_ALL.rename(columns = {'AVG(AVG_TEMP_CELSIUS)':'AVG_TEMP_CELSIUS'}, inplace = True)
#DATAF_SHAP_ALL.rename(columns = {'AVG(MIN_TEMP_CELSIUS)':'MIN_TEMP_CELSIUS'}, inplace = True)
#DATAF_SHAP_ALL.rename(columns = {'AVG(HUMID_PCT)':'HUMID_PCT'}, inplace = True)
#DATAF_SHAP_ALL.rename(columns = {'AVG(PRECIP_MM)':'PRECIP_MM'}, inplace = True)
#DATAF_SHAP_ALL.rename(columns = {'AVG(RETAIL_AND_RECREATION_PCT_CHANGE)':'RETAIL_AND_RECREATION_PCT_CHANGE'}, inplace = True)
#DATAF_SHAP_ALL.rename(columns = {'AVG(RESIDENTIAL_PCT_CHANGE)':'RESIDENTIAL_PCT_CHANGE'}, inplace = True)
#DATAF_SHAP_ALL.rename(columns = {'AVG(PERSONAL_DISPOSABLE_INCOME_REAL_LCU)':'PERSONAL_DISPOSABLE_INCOME_REAL_LCU'}, inplace = True)
#DATAF_SHAP_ALL.rename(columns = {'AVG(UNEMP_RATE)':'UNEMP_RATE'}, inplace = True)


shap_file = 'CHN_'+str(DATAF_ALL_HIST_PROJ.SUBCAT_NAME[0])+'_TOTAL_SHAP_ALL'+'.csv'
DATAF_SHAP_ALL.to_csv(shap_file,index=False)
