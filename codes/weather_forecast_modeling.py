# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 06:36:26 2021

@author: Administrator
"""



" WEATHER FORECASTING "

print("----------------------------------------------------------------------------------------------------------------")
print(" Weather predictive modelling and forecast started.........................                                     ")

import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster, ExponentialSmoothing
from sktime.forecasting.ets import AutoETS
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.arima import AutoARIMA
import pmdarima as pm
from pmdarima import model_selection
from statsmodels.tsa.holtwinters import ExponentialSmoothing


DATAF_FILTER = pd.read_excel("China_data_for_weather_prediction.xlsx")

UL_GEOID = DATAF_FILTER.UL_GEO_ID.unique()

for i in UL_GEOID:
    print("-----------------------------------------------------")
    print("UL_GEO_ID", i)
    print("-----------------------------------------------------")
    
    DATAF = DATAF_FILTER[(DATAF_FILTER.UL_GEO_ID==i)]
    
    p = DATAF.shape[0]
    
    
    "========================================================================"
    
    print("AVG TEMP CELSIUS")
    print("-----------------------------------------------------")
    
    Output_cols = DATAF['AVG(AVG_TEMP_CELSIUS)']
    
    y_all = Output_cols.iloc[:p]
    y_all.reset_index(drop=True,inplace =True)
    y = y_all.iloc[:p-19]
    y_forecast = y_all.iloc[p-19:p]
    y_train, y_test = temporal_train_test_split(y)
    
    fh_train = ForecastingHorizon(y_train.index, is_relative=False)
    forecaster = ThetaForecaster(sp=12) 
    forecaster.fit(y_train)
    y_pred_train = forecaster.predict(fh_train)
    
    #model = ExponentialSmoothing(history, trend="add", seasonal="add", seasonal_periods=12)
    #model_fit = model.fit(optimized=True,use_boxcox=False, remove_bias=False)

    print("Train Actual & Prediction ")
    plt.figure(figsize=(15,5))
    plt.plot(y_train)
    plt.plot(y_pred_train)
    plt.show()
    
    fh_test = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred_test = forecaster.predict(fh_test)
    
    print("Test Actual & Prediction ")
    plt.figure(figsize=(15,5))
    plt.plot(y_test)
    plt.plot(y_pred_test)
    plt.show()
    
    fh_forecast = ForecastingHorizon(y_forecast.index, is_relative=False)
    y_pred_forecast = forecaster.predict(fh_forecast)
    
    y_pred = y_pred_train.append(y_pred_test,ignore_index = True)
    y_forecast = y.append(y_pred_forecast,ignore_index = True)
    print("Historic (Actual, Predicted) & Forecast")
    plt.figure(figsize=(15,5))
    plt.plot(y_forecast)
    plt.plot(y_pred)
    plt.show()
    
    DATAF_TEMP = pd.DataFrame(y.append(y_pred_forecast,ignore_index = True))
    DATAF['AVG(AVG_TEMP_CELSIUS)'] = DATAF_TEMP.values
    
    "========================================================================"
    
    
    "========================================================================"
    
    print("HUMID_PCT")
    print("-----------------------------------------------------")
    
    Output_cols = DATAF['AVG(HUMID_PCT)']
    
    y_all = Output_cols.iloc[:p]
    y_all.reset_index(drop=True,inplace =True)
    y = y_all.iloc[:p-19]
    y_forecast = y_all.iloc[p-19:p]
    y_train, y_test = temporal_train_test_split(y)
    
    fh_train = ForecastingHorizon(y_train.index, is_relative=False)
    forecaster = ThetaForecaster(sp=12) 
    forecaster.fit(y_train)
    y_pred_train = forecaster.predict(fh_train)
    
    #model = ExponentialSmoothing(history, trend="add", seasonal="add", seasonal_periods=12)
    #model_fit = model.fit(optimized=True,use_boxcox=False, remove_bias=False)

    print("Train Actual & Prediction ")
    plt.figure(figsize=(15,5))
    plt.plot(y_train)
    plt.plot(y_pred_train)
    plt.show()
    
    fh_test = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred_test = forecaster.predict(fh_test)
    
    print("Test Actual & Prediction ")
    plt.figure(figsize=(15,5))
    plt.plot(y_test)
    plt.plot(y_pred_test)
    plt.show()
    
    fh_forecast = ForecastingHorizon(y_forecast.index, is_relative=False)
    y_pred_forecast = forecaster.predict(fh_forecast)
    
    y_pred = y_pred_train.append(y_pred_test,ignore_index = True)
    y_forecast = y.append(y_pred_forecast,ignore_index = True)
    print("Historic (Actual, Predicted) & Forecast")
    plt.figure(figsize=(15,5))
    plt.plot(y_forecast)
    plt.plot(y_pred)
    plt.show()
    
    DATAF_TEMP = pd.DataFrame(y.append(y_pred_forecast,ignore_index = True))
    DATAF['AVG(HUMID_PCT)'] = DATAF_TEMP.values
    
    "========================================================================"
    
    if i==UL_GEOID[0]:
        DATAF_FILTER_UPDATED = DATAF
    else:
        DATAF_FILTER_UPDATED = pd.concat([DATAF_FILTER_UPDATED,DATAF])
    

print(" Weather predictive modelling and forecast completed                                                            ")
print("----------------------------------------------------------------------------------------------------------------")
