




" TDP Prediction function "


def TDP_Prediction(DATAF,INP_COL):
    s = DATAF.shape[0]
    
    for i in range(s):
        SUBCAT = DATAF.SUBCAT_CD.iloc[i]
        CHNL= DATAF.CHNL_CD.iloc[i]
        FMT = DATAF.FMT_CD.iloc[i]
        REG = DATAF.UL_GEO_ID.iloc[i]
        
        #INP_COL = ['SEASONALITY_INDEX_TDP','AVG(TRAFFIC_WEIGHT)','YEAR','SALES_TREND_CAL']
    
        X = DATAF[INP_COL]
        
        filename = 'TDP_Total_Prediction_'+str(SUBCAT)+"_"+str(CHNL)+"_"+str(FMT)+"_"+str(REG)
      
        #load the model from disk
        loaded_model = joblib.load(filename)
        
        YPred = loaded_model.predict(X)
        
        DATAF.TDP.iloc[i] = YPred[i]
        
        
    return DATAF    
        
        


