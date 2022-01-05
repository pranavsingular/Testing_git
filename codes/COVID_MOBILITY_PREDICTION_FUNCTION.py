



" COVID Mobility Prediction function "



def COVID_Mobility_Prediction(DATAF_ALL_HIST_PROJ):
  
    " Filtering Historic & Forecast Period "
    DATAF_ALL_HIST = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']>0)]
    DATAF_ALL_PROJ = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ['SALES_VOLUME']==0)]
    
    " Forecasting "
    OUTPUT_VAR_LIST = ['AVG(RETAIL_AND_RECREATION_PCT_CHANGE)','AVG(RESIDENTIAL_PCT_CHANGE)']  
    REG_LIST = DATAF_ALL_PROJ.UL_GEO_ID.unique()
    
    for i in REG_LIST:
      
      DATAF = DATAF_ALL_PROJ[(DATAF_ALL_PROJ.UL_GEO_ID==i)]
      
      "RETAIL_AND_RECREATION_PCT_CHANGE PREDICTION"
      
      filename = 'COVID_Mobility_Prediction_'+'AVG(RETAIL_AND_RECREATION_PCT_CHANGE)'+"_REGION_"+str(i)
      
      #load the model from disk
      loaded_model = joblib.load(filename)
      
      INP_COL = ['SUM(NEW_CASES)','SUM(NEW_DEATHS)','TIME_COVID']
      
      X = DATAF[INP_COL]
      
      Y_PRED = loaded_model.predict(X)
      
      DATAF['AVG(RETAIL_AND_RECREATION_PCT_CHANGE)'] = Y_PRED
      
      "RESIDENTIAL_PCT_CHANGE PREDICTION"
      
      filename = 'COVID_Mobility_Prediction_'+'AVG(RESIDENTIAL_PCT_CHANGE)'+"_REGION_"+str(i)
      
      #load the model from disk
      loaded_model = joblib.load(filename)
      
      X = DATAF[INP_COL]
      
      Y_PRED = loaded_model.predict(X)
      
      DATAF['AVG(RESIDENTIAL_PCT_CHANGE)'] = Y_PRED
      
      if i==REG_LIST[0]:
          DATAF_ALL_PROJ_UPDATED = DATAF
      else:
          DATAF_ALL_PROJ_UPDATED = pd.concat([DATAF_ALL_PROJ_UPDATED,DATAF])
          
    
    s = DATAF_ALL_PROJ_UPDATED.shape[0] 
    
    " Applying saturation to the prediction "
    
    for i in range(s):
        if DATAF_ALL_PROJ_UPDATED['AVG(RETAIL_AND_RECREATION_PCT_CHANGE)'].iloc[i]>0:
            DATAF_ALL_PROJ_UPDATED['AVG(RETAIL_AND_RECREATION_PCT_CHANGE)'].iloc[i] = 0
            
    for i in range(s):
        if DATAF_ALL_PROJ_UPDATED['AVG(RESIDENTIAL_PCT_CHANGE)'].iloc[i]<0:
            DATAF_ALL_PROJ_UPDATED['AVG(RESIDENTIAL_PCT_CHANGE)'].iloc[i] = 0        
            
            
            
    DATAF_ALL_HIST_PROJ = pd.concat([DATAF_ALL_HIST,DATAF_ALL_PROJ_UPDATED])  

    return DATAF_ALL_HIST_PROJ
      
