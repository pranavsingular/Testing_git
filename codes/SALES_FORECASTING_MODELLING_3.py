





" SALES FORECASTING MODELLING "


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
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


" To calculate the number of combinations "

DATAF_ALL_START = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.YEAR==Start_year) & (DATAF_ALL_HIST_PROJ.MONTH==Start_month)]

s = DATAF_ALL_START.shape[0]

" METRICS DATAFRAME INITIALIZATION "

metrics_data = {'SUBCAT NAME':[],'SUBCAT_CD':[],'RMSE_Train':[],'RMSE_Test':[],'MAPE_Train':[],'MAPE_Test':[],'RSQ_Train':[], 'RSQ_Test':[]}
    
DATAF_METRICS = pd.DataFrame(metrics_data)

SUBCAT_LIST = DATAF_ALL_HIST_PROJ.SUBCAT_CD.unique()

" Weights for the 2 models "




for c in SUBCAT_LIST:
    
    
    cell_str = "SUBCAT_"+str(c)
    print("-----------------------------------------------------------------------------------")
    print(cell_str)

    DATAF = DATAF_ALL_HIST[(DATAF_ALL_HIST.SUBCAT_CD==c)]
    
    SUBCAT_NAME = 'SUBCAT'
    
    #DATAF = DATAF.sample(frac=1)
    '''
    SEASONALITY_INDEX_SALES 1
    SALES_TREND_CAL 1
    PRICE_PER_VOL -1
    TDP 1
    TRAFFIC_WEIGHT 0
    NEW_CASES 0
    GDP_REAL_LCU 1
    AVG_TEMP_CELSIUS 0
    HUMID_PCT 0
    
    '''
    INP_COL = sales_input_col
    OUT_COL = sales_output_col
    
    X = DATAF[INP_COL]
    Y = pd.DataFrame(DATAF[OUT_COL])
        
    
    " Train Test Split "
    
    X_train_test = X.iloc[0:42,:]
    #X_hold = X.iloc[33:36,:]
    
    Y_train_test = Y.iloc[0:42,:]
    #Y_test = Y.iloc[33:36,:]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_test, Y_train_test, test_size=0.2, shuffle=True, random_state=10, stratify = None)
    
    model_XGB = XGBRegressor(monotone_constraints = sales_cons)
    
    param_grid = {'objective':['reg:squarederror'],
                      'learning_rate': [0.02,0.05,0.07,0.09,0.1], #`eta` value
                      #'gamma': range(1,6),
                      'max_depth': range(2,5),
                      #'min_child_weight': range(1,10),
                      #'subsample': [i/10. for i in range(6,9)],
                      #'colsample_bytree': [i/10. for i in range(6,9)],
                      #'reg_alpha':  [i/100. for i in range(1,100)],
                      #'reg_lambda':  [0.05, 0.1, 1.0],
                      #'random_state':range(60,90,5),
                      'n_estimators': range(50,100,5)}
    
    
    grid = RandomizedSearchCV(model_XGB,
                            param_grid,
                            cv = 5,
                            n_iter=200,
                            n_jobs = 12,
                            verbose=True,
                            scoring='neg_mean_absolute_percentage_error')
        
        
    eval_set = [(X_train, Y_train)]
        
    grid.fit(X_train,Y_train, early_stopping_rounds=2, eval_metric="mae", eval_set=eval_set, verbose=False)
        
    
    print("-------------------------------------------------------------------------")
    print("Best Score XGB",np.round(grid.best_score_,2)*100)
    print("-------------------------------------------------------------------------")
    print("Best Parameters XGB",grid.best_params_)
    print("-------------------------------------------------------------------------")
    
    model_XGB = grid.best_estimator_
    
#    model_RF = RandomForestRegressor()
#    
#    param_grid = {'max_depth': range(4,7),'n_estimators': range(50,100,5),'max_features':[i/10. for i in range(6,9)]}
#    
#    
#    grid = RandomizedSearchCV(model_RF,
#                            param_grid,
#                            cv = 5,
#                            n_iter=200,
#                            n_jobs = 12,
#                            verbose=True,
#                            scoring='neg_mean_absolute_percentage_error')
#        
#        
#
#    grid.fit(X_train,Y_train)
#        
#    
#    print("-------------------------------------------------------------------------")
#    print("Best Score RF",np.round(grid.best_score_,2)*100)
#    print("-------------------------------------------------------------------------")
#    print("Best Parameters RF",grid.best_params_)
#    print("-------------------------------------------------------------------------")
#    
#    model_RF = grid.best_estimator_
    
    
    model_LIN = LinearRegression()
    modelfwd = VotingRegressor([('XGB', model_XGB), ('LIN', model_LIN)],weights=[w1, w2])
    
    modelfwd = modelfwd.fit(X_train,Y_train)
    
    from sklearn.model_selection import cross_val_score
    print(100*cross_val_score(modelfwd, X, Y, cv=5,scoring='neg_mean_absolute_percentage_error'))
    
    " Model validation "

    YPred_Train = modelfwd.predict(X_train) 
        
    RSQ_Train = np.round(r2_score(Y_train, YPred_Train),2)
    
    RMSE_Train= np.round(np.sqrt(mean_squared_error(Y_train, YPred_Train)),2)
    
    MAPE_Train= np.round(mean_absolute_percentage_error(Y_train, YPred_Train)*100,2)
    
    YPred_Test = modelfwd.predict(X_test)
        
    RSQ_Test = np.round(r2_score(Y_test, YPred_Test),2)
    
    RMSE_Test = np.round(np.sqrt(mean_squared_error(Y_test, YPred_Test)),2) 
    
    MAPE_Test= np.round(mean_absolute_percentage_error(Y_test, YPred_Test)*100,2)
    
    YPred_Full = modelfwd.predict(X) 
    
    print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
    print("Model training and validation metrics ")
    
    print("RMSE Train:",RMSE_Train)
    
    print("RMSE Test:",RMSE_Test)
    
    print("MAPE Train:",MAPE_Train)
    
    print("MAPE Test:",MAPE_Test)
    
    print("RSQ Train:",RSQ_Train)
    
    print("RSQ Test:",RSQ_Test)
    
    
    print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
    print("Model Interpretability ")
    
    features = INP_COL
    print("Feature list", features)
    
    sns.set(rc={'figure.figsize':(15,10)})
    sns.set_style("whitegrid")
#    
    print("Partial Dependence Plots")
    display = plot_partial_dependence(
           modelfwd, X, features, kind="both", subsample=100,
           n_jobs=1, grid_resolution=100, random_state=10
    )
    display.figure_.suptitle(
        'Partial dependence of features'
    )
    display.figure_.subplots_adjust(hspace=0.5)
    
    "-------------------------------------------------------------------------------------------------"
    
    X = DATAF[INP_COL].values
    Y = pd.DataFrame(DATAF[OUT_COL]).values
    
    X_train_test = X[0:42,:]
    #X_hold = X[33:36,:]
    
    Y_train_test = Y[0:42,:]
    #Y_hold = Y[33:36,:]
    
    " Train Test Split "
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_test, Y_train_test, test_size=0.2, shuffle=True, random_state=10, stratify = None)

    
    model_XGB = XGBRegressor(monotone_constraints = sales_cons)
    
    param_grid = {'objective':['reg:squarederror'],
                      'learning_rate': [0.02,0.05,0.07,0.09,0.1], #`eta` value
                      #'gamma': range(1,6),
                      'max_depth': range(2,5),
                      #'min_child_weight': range(1,10),
                      #'subsample': [i/10. for i in range(6,9)],
                      #'colsample_bytree': [i/10. for i in range(6,9)],
                      #'reg_alpha':  [i/100. for i in range(1,100)],
                      #'reg_lambda':  [0.05, 0.1, 1.0],
                      #'random_state':range(60,90,5),
                      'n_estimators': range(50,100,5)}
    
    
    grid = RandomizedSearchCV(model_XGB,
                            param_grid,
                            cv = 5,
                            n_iter=200,
                            n_jobs = 12,
                            verbose=True,
                            scoring='neg_mean_absolute_percentage_error')
        
        
    eval_set = [(X_train, Y_train)]
        
    grid.fit(X_train,Y_train, early_stopping_rounds=2, eval_metric="mae", eval_set=eval_set, verbose=False)
        
    
    print("-------------------------------------------------------------------------")
    print("Best Score XGB",np.round(grid.best_score_,2)*100)
    print("-------------------------------------------------------------------------")
    print("Best Parameters XGB",grid.best_params_)
    print("-------------------------------------------------------------------------")
    
    model_XGB = grid.best_estimator_
    
#    model_RF = RandomForestRegressor()
#    
#    param_grid = {'max_depth': range(4,7),'n_estimators': range(50,100,5),'max_features':[i/10. for i in range(6,9)]}
#    
#    
#    grid = RandomizedSearchCV(model_RF,
#                            param_grid,
#                            cv = 5,
#                            n_iter=200,
#                            n_jobs = 12,
#                            verbose=True,
#                            scoring='neg_mean_absolute_percentage_error')
#        
#        
#
#    grid.fit(X_train,Y_train)
#        
#    
#    print("-------------------------------------------------------------------------")
#    print("Best Score RF",np.round(grid.best_score_,2)*100)
#    print("-------------------------------------------------------------------------")
#    print("Best Parameters RF",grid.best_params_)
#    print("-------------------------------------------------------------------------")
#    
#    model_RF = grid.best_estimator_
    
    
    model_LIN = LinearRegression()
    model_LIN = model_LIN.fit(X_train,Y_train)

    
    " Model validation "
    
#    YPred_Train = (w1*model_XGB.predict(X_train))+(w2*model_RF.predict(X_train))
#
#    RSQ_Train = np.round(r2_score(Y_train, YPred_Train),2)
#    
#    RMSE_Train= np.round(np.sqrt(mean_squared_error(Y_train, YPred_Train)),2)
#    
#    MAPE_Train= np.round(mean_absolute_percentage_error(Y_train, YPred_Train)*100,2)
#    
#    YPred_Test = (w1*model_XGB.predict(X_test))+(w2*model_RF.predict(X_test))
#        
#    RSQ_Test = np.round(r2_score(Y_test, YPred_Test),2)
#    
#    RMSE_Test = np.round(np.sqrt(mean_squared_error(Y_test, YPred_Test)),2) 
#    
#    MAPE_Test= np.round(mean_absolute_percentage_error(Y_test, YPred_Test)*100,2)
#    
#    YPred_Full = (w1*model_XGB.predict(X))+(w2*model_RF.predict(X))
    
    YPred_Train = (w1*model_XGB.predict(X_train).reshape(-1,1))+(w2*model_LIN.predict(X_train))

    RSQ_Train = np.round(r2_score(Y_train, YPred_Train),2)
    
    RMSE_Train= np.round(np.sqrt(mean_squared_error(Y_train, YPred_Train)),2)
    
    MAPE_Train= np.round(mean_absolute_percentage_error(Y_train, YPred_Train)*100,2)
    
    YPred_Test = (w1*model_XGB.predict(X_test).reshape(-1,1))+(w2*model_LIN.predict(X_test))
        
    RSQ_Test = np.round(r2_score(Y_test, YPred_Test),2)
    
    RMSE_Test = np.round(np.sqrt(mean_squared_error(Y_test, YPred_Test)),2) 
    
    MAPE_Test= np.round(mean_absolute_percentage_error(Y_test, YPred_Test)*100,2)
    
    YPred_Full = (w1*model_XGB.predict(X).reshape(-1,1))+(w2*model_LIN.predict(X))
    
    print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
    print("Model training and validation metrics ")
    
    print("RMSE Train:",RMSE_Train)
    
    print("RMSE Test:",RMSE_Test)
    
    print("MAPE Train:",MAPE_Train)
    
    print("MAPE Test:",MAPE_Test)
    
    print("RSQ Train:",RSQ_Train)
    
    print("RSQ Test:",RSQ_Test)
    
    " UPDATING DATAF_METRICS "
    
    DATAF_METRICS.loc[len(DATAF_METRICS.index)] = [SUBCAT_NAME, c, RMSE_Train, RMSE_Test, MAPE_Train, MAPE_Test, RSQ_Train, RSQ_Test] 
    
    " Saving model file "
        
    filename = 'SALES_TOTAL_Prediction_XGB_'+str(c)
    joblib.dump(model_XGB, filename)
    
    filename = 'SALES_TOTAL_Prediction_LIN_'+str(c)
    joblib.dump(model_LIN, filename)
    
    print("Actual & Prediction ")
    DATAF = DATAF.sort_index()
    X = DATAF[INP_COL].values
    Y = DATAF[OUT_COL].values
    YPred_Full = (w1*model_XGB.predict(X).reshape(-1,1))+(w2*model_LIN.predict(X))
    
    DATAF.insert(2, "SALES_VOLUME_PREDICTED",YPred_Full)
    
    plt.figure(figsize=(20,6))
    plt.plot(Y)
    plt.plot(YPred_Full)
    plt.show()
    
    print(" Historic & Forecast Predictions")
          
    DATAF = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.SUBCAT_CD==c)]
    
    X = DATAF[INP_COL].values
    Y = DATAF[OUT_COL].values
    YPred_Full = (w1*model_XGB.predict(X).reshape(-1,1))+(w2*model_LIN.predict(X))
    
    DATAF.insert(2, "SALES_VOLUME_PREDICTED",YPred_Full)
    
    plt.figure(figsize=(20,6))
    plt.plot(Y)
    plt.plot(YPred_Full)
    plt.show()
    
    if c == SUBCAT_LIST[0]:
        DATAF_ALL_HIST_PROJ_SALES = DATAF
    else:
        DATAF_ALL_HIST_PROJ_SALES = pd.concat([DATAF_ALL_HIST_PROJ_SALES,DATAF])
        
        
    