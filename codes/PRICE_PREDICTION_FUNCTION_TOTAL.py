



" Price Prediction function "


def Price_Prediction(DATAF,INP_COL):
    s = DATAF.shape[0]
    
    for i in range(s):
        SUBCAT = DATAF.SUBCAT_CD.iloc[i]
        CHNL= DATAF.CHNL_CD.iloc[i]
        FMT = DATAF.FMT_CD.iloc[i]
        REG = DATAF.UL_GEO_ID.iloc[i]
        
        #INP_COL = ['SEASONALITY_INDEX_PRICE','AVG(CONSUMER_PRICE_INDEX)','AVG(TRAFFIC_WEIGHT)']
    
        X = DATAF[INP_COL]
        
        filename = 'Price_Total_Prediction_'+str(SUBCAT)+"_"+str(CHNL)+"_"+str(FMT)+"_"+str(REG)
      
        #load the model from disk
        loaded_model = joblib.load(filename)
        
        YPred = loaded_model.predict(X)
        
        DATAF.PRICE_PER_VOL.iloc[i] = YPred[i]
        
        
    return DATAF    
        
        

