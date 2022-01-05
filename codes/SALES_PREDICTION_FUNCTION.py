




" Sales Prediction function "


def Sales_Prediction(DATAF,INP_COL):
    s = DATAF.shape[0]
    
    for i in range(s):
        SUBCAT = DATAF.SUBCAT_CD.iloc[i]
        CHNL= DATAF.CHNL_CD.iloc[i]
        FMT = DATAF.FMT_CD.iloc[i]
        REG = DATAF.UL_GEO_ID.iloc[i]
        
        #INP_COL = ['SEASONALITY_INDEX_SALES','SALES_TREND_CAL','PRICE_PER_VOL','TDP','AVG(TRAFFIC_WEIGHT)','SUM(NEW_CASES)','AVG(GDP_REAL_LCU)','AVG(AVG_TEMP_CELSIUS)','AVG(HUMID_PCT)']
    
        X = DATAF[INP_COL].values
        
        filename1 = 'SALES_TOTAL_Prediction_XGB_'+str(SUBCAT)
    
        filename2 = 'SALES_TOTAL_Prediction_LIN_'+str(SUBCAT)
      
        #load the model from disk
        loaded_model1 = joblib.load(filename1)
        loaded_model2 = joblib.load(filename2)
        
        #YPred = (0.8*loaded_model1.predict(X))+(0.2*loaded_model2.predict(X))
        YPred = (w1*loaded_model1.predict(X).reshape(-1,1))+(w2*loaded_model2.predict(X))
        
        DATAF.SALES_VOLUME_PREDICTED.iloc[i] = YPred[i]
        
        
    return DATAF    
        
        



