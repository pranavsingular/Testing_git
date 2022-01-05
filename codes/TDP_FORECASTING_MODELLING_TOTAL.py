





" TDP PREDICTIVE MODELLING "

print("TDP Predictive Modelling in Progress.....................")

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
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

DATAF_ALL_START = DATAF_ALL_HIST[(DATAF_ALL_HIST.YEAR==Start_year) & (DATAF_ALL_HIST.MONTH==Start_month)]

s = DATAF_ALL_START.shape[0]

for i in range(s):
    SUBCAT = DATAF_ALL_START.SUBCAT_CD.iloc[i]
    CHNL= DATAF_ALL_START.CHNL_CD.iloc[i]
    FMT = DATAF_ALL_START.FMT_CD.iloc[i]
    REG = DATAF_ALL_START.UL_GEO_ID.iloc[i]
    
    print("TDP")
    print("SUBCAT ",SUBCAT)
    print("CHANNEL ",CHNL)
    print("FORMAT ",FMT)
    print("REGION ",REG)
    
    DATAF = DATAF_ALL_HIST[(DATAF_ALL_HIST.SUBCAT_CD==SUBCAT)&(DATAF_ALL_HIST.CHNL_CD==CHNL)&(DATAF_ALL_HIST.FMT_CD==FMT)&(DATAF_ALL_HIST.UL_GEO_ID==REG)]
    
    DATAF = DATAF.sample(frac=1)
    
    '''
    SEASONALITY_INDEX_TDP 1
    TRAFFIC_WEIGHT
    YEAR
    SALES_TREND_CAL
    
    
    '''
    INP_COL = tdp_input_col
    OUT_COL = tdp_output_col
    
    X = DATAF[INP_COL]
    Y = DATAF[OUT_COL]
    
    " Train Test Split "
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10, stratify = None)
    
    
    model_XGB = XGBRegressor(monotone_constraints = tdp_cons)
    
    param_grid = {'objective':['reg:squarederror'],
                      'learning_rate': [0.02,0.05,0.07,0.09,0.1], #`eta` value
                      #'gamma': range(1,6),
                      'max_depth': range(4,6),
                      #'min_child_weight': range(1,10),
                      #'subsample': [i/10. for i in range(6,9)],
                      #'colsample_bytree': [i/10. for i in range(6,9)],
                      #'reg_alpha':  [i/100. for i in range(1,100)],
                      #'reg_lambda':  [0.05, 0.1, 1.0],
                      #'random_state':range(60,90,5),
                      'n_estimators': range(50,80,5)}
    
    
    grid = RandomizedSearchCV(model_XGB,
                            param_grid,
                            cv = 5,
                            n_iter=100,
                            n_jobs = 2,
                            verbose=True,
                            scoring='neg_mean_absolute_error')
        
        
    eval_set = [(X_train, Y_train)]
        
    grid.fit(X_train,Y_train, early_stopping_rounds=2, eval_metric="mae", eval_set=eval_set, verbose=False)
        
    
    print("-------------------------------------------------------------------------")
    print("Best Score XGB",np.round(grid.best_score_,2))
    print("-------------------------------------------------------------------------")
    print("Best Parameters XGB",grid.best_params_)
    print("-------------------------------------------------------------------------")
    
    model_XGB = grid.best_estimator_
    
#    model_RF = RandomForestRegressor()
#    
#    param_grid = {'max_depth': range(4,6),'n_estimators': range(50,80,5)}
#    
#    
#    grid = RandomizedSearchCV(model_RF,
#                            param_grid,
#                            cv = 5,
#                            n_iter=200,
#                            n_jobs = 12,
#                            verbose=True,
#                            scoring='neg_mean_absolute_error')
#        
#        
#
#    grid.fit(X_train,Y_train)
#        
#    
#    print("-------------------------------------------------------------------------")
#    print("Best Score RF",np.round(grid.best_score_,2))
#    print("-------------------------------------------------------------------------")
#    print("Best Parameters RF",grid.best_params_)
#    print("-------------------------------------------------------------------------")
#    
#    model_RF = grid.best_estimator_
    
    
    model_RF = LinearRegression()
    modelfwd = VotingRegressor([('XGB', model_XGB), ('RF', model_RF)],weights=[0.9, 0.1])
    
    #modelfwd = model_LINEAR
    modelfwd = modelfwd.fit(X_train,Y_train)
    
    " Saving model file "
        
    filename = 'TDP_Total_Prediction_'+str(SUBCAT)+"_"+str(CHNL)+"_"+str(FMT)+"_"+str(REG)
    joblib.dump(modelfwd, filename)
    
    from sklearn.model_selection import cross_val_score
    print(cross_val_score(modelfwd, X, Y, cv=5,scoring='neg_mean_absolute_error'))
    
    " Model validation "

    YPred_Train = modelfwd.predict(X_train) 
        
    RSQ_Train_FWD = np.round(r2_score(Y_train, YPred_Train),2)
    
    RMSE_Train_FWD = np.round(np.sqrt(mean_squared_error(Y_train, YPred_Train)),2)
    
    YPred_Test = modelfwd.predict(X_test)
        
    RSQ_Test_FWD = np.round(r2_score(Y_test, YPred_Test),2)
    
    RMSE_Test_FWD = np.round(np.sqrt(mean_squared_error(Y_test, YPred_Test)),2) 
    
    YPred_Full = modelfwd.predict(X) 
    
    print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
    print("Model training and validation metrics ")
    
    print("RMSE Train FWD:",RMSE_Train_FWD)
    
    print("RMSE Test FWD:",RMSE_Test_FWD)
    
    print("RSQ Train FWD:",RSQ_Train_FWD)
    
    print("RSQ Test FWD:",RSQ_Test_FWD)
    
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
    
    print("Actual & Prediction ")
    DATAF = DATAF.sort_index()
    X = DATAF[INP_COL]
    Y = DATAF[OUT_COL]
    YPred_Full = modelfwd.predict(X)
    
    plt.figure(figsize=(20,6))
    plt.plot(Y.values)
    plt.plot(YPred_Full)
    plt.show()
    
    print(" Historic & Forecast Predictions")
          
    DATAF = DATAF_ALL_HIST_PROJ[(DATAF_ALL_HIST_PROJ.SUBCAT_CD==SUBCAT) & (DATAF_ALL_HIST_PROJ.CHNL_CD==CHNL)]
    
    DATAF = DATAF[(DATAF.FMT_CD==FMT) & (DATAF.UL_GEO_ID==REG)]
    
    X = DATAF[INP_COL]
    Y = DATAF[OUT_COL]
    YPred_Full = modelfwd.predict(X)
    
    plt.figure(figsize=(20,6))
    plt.plot(Y.values)
    plt.plot(YPred_Full)
    plt.show()
    

print("TDP Predictive Modelling Completed")    
