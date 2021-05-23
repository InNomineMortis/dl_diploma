import itertools
import warnings
import tkinter as tk
from finta import TA
from tkinter import *
from IPython import display
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import statsmodels.api as sm
import datetime
import random
from statsmodels.tsa.arima_model import ARIMA
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from gym import spaces
import numpy as np
import pandas as pnd
from matplotlib import pyplot as plt

from gym_env.stock_trading_env import StockTradingEnv


def load_model(model, model_name):
    model = PPO2.load(model_name)


if __name__ == '__main__':
    df = pnd.read_csv('datasets/F.csv')
    df['date'] = pnd.to_datetime(df['date'], format='%Y-%m-%d')
    df.sort_values('date', ascending=True, inplace=True)
    df.date.dt.to_period('M')
    df.drop(columns=['adjclose'], inplace=True)
    df = df.groupby(pnd.DatetimeIndex(df.date).to_period('M')).nth(0)
    df.drop(columns=['date'], inplace=True)
    start_date = pnd.to_datetime('2015-01')
    print(type(start_date))
    end_date = pnd.to_datetime('2020-01')
    # CLOSE
    train_data_close = df['2000-01':'2015-01']
    train_data_close.drop(columns=['open', 'high', 'volume', 'low'], inplace=True)
    test_data_close = df['2015-01':'2020-01']
    test_data_close.drop(columns=['open', 'high', 'volume', 'low'], inplace=True)
    print('test_data_close length: ', len(test_data_close), 'train_data_close length:', len(train_data_close),
          'whole set:', len(df))
    train_data_close.tail()

    # LOW
    train_data_low = df['2000-01':'2015-01']
    train_data_low.drop(columns=['open', 'high', 'volume', 'close'], inplace=True)
    test_data_low = df['2015-01':'2020-01']
    test_data_low.drop(columns=['open', 'high', 'volume', 'close'], inplace=True)
    print('test_data_low length: ', len(test_data_low), 'train_data_low length:', len(train_data_low), 'whole set:',
          len(df))
    train_data_low.tail()

    # HIGH
    train_data_high = df['2000-01':'2015-01']
    train_data_high.drop(columns=['open', 'low', 'volume', 'close'], inplace=True)
    test_data_high = df['2015-01':'2020-01']
    test_data_high.drop(columns=['open', 'low', 'volume', 'close'], inplace=True)
    print('test_data_high length: ', len(test_data_high), 'train_data_high length:', len(train_data_high), 'whole set:',
          len(df))
    train_data_high.tail()

    # OPEN
    train_data_open = df['2000-01':'2015-01']
    train_data_open.drop(columns=['high', 'low', 'volume', 'close'], inplace=True)
    test_data_open = df['2015-01':'2020-01']
    test_data_open.drop(columns=['high', 'low', 'volume', 'close'], inplace=True)
    print('test_data_open length: ', len(test_data_open), 'train_data_open length:', len(train_data_open), 'whole set:',
          len(df))
    train_data_open.tail()

    # Define the d and q parameters to take any value between 0 and 1
    q = d = range(0, 2)
    # Define the p parameters to take any value between 0 and 3
    p = range(0, 4)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    warnings.filterwarnings("ignore")  # specify to ignore warning messages

    AIC = [(3, 1, 0)]
    SARIMAX_model = [(3, 1, 0), (3, 1, 1, 12)]
    # for param in pdq:
    #     for param_seasonal in seasonal_pdq:
    #         try:
    #             mod = sm.tsa.statespace.SARIMAX(train_data_close,
    #                                             order=param,
    #                                             seasonal_order=param_seasonal,
    #                                             enforce_stationarity=False,
    #                                             enforce_invertibility=False)
    #
    #             results = mod.fit()
    #
    #             print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
    #             AIC.append(results.aic)
    #             SARIMAX_model.append([param, param_seasonal])
    #         except:
    #             continue

    print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],
                                                                 SARIMAX_model[AIC.index(min(AIC))][1]))

    # mod_close = ARIMA(train_data_close,order=(3,2,1))
    # results_close = mod_close.fit(disp=-1)

    # Let's fit this model
    print(SARIMAX_model[0])
    mod_close = sm.tsa.statespace.SARIMAX(train_data_close,
                                          order=SARIMAX_model[0],
                                          seasonal_order=SARIMAX_model[1],
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)

    results_close = mod_close.fit()

    # mod_open = ARIMA(train_data_open,order=(3,1,1))
    # results_open = mod_open.fit(disp=-1)

    mod_open = sm.tsa.statespace.SARIMAX(train_data_open,
                                         order=SARIMAX_model[0],
                                         seasonal_order=SARIMAX_model[1],
                                         enforce_stationarity=False,
                                         enforce_invertibility=False)

    results_open = mod_open.fit()

    # mod_low = ARIMA(train_data_low,order=(3,1,1))
    # results_low = mod_low.fit(disp=-1)

    mod_low = sm.tsa.statespace.SARIMAX(train_data_low,
                                        order=SARIMAX_model[0],
                                        seasonal_order=SARIMAX_model[1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

    results_low = mod_low.fit()

    # mod_high = ARIMA(train_data_high,order=(3,2,1))
    # results_high = mod_high.fit(disp=-1)

    mod_high = sm.tsa.statespace.SARIMAX(train_data_high,
                                         order=SARIMAX_model[0],
                                         seasonal_order=SARIMAX_model[1],
                                         enforce_stationarity=False,
                                         enforce_invertibility=False)

    results_high = mod_high.fit()

    index = pnd.date_range(start_date, end_date, freq='M').strftime('%Y-%m')
    pred_close = results_close.get_forecast(end_date)
    pred_close_ci = pred_close.conf_int()
    fc_series_close = pred_close.predicted_mean
    # lower_series_close = pnd.Series(conf[:, 0], index=index)
    # upper_series_close = pnd.Series(conf[:, 1], index=index)

    index = pnd.date_range(start_date, end_date, freq='M').strftime('%Y-%m')
    pred_open = results_open.get_forecast(end_date)
    pred_open_ci = pred_open.conf_int()
    fc_series_open = pred_open.predicted_mean
    # lower_series_open = pnd.Series(conf[:, 0], index=index)
    # upper_series_open = pnd.Series(conf[:, 1], index=index)

    index = pnd.date_range(start_date, end_date, freq='M').strftime('%Y-%m')
    pred_low = results_low.get_forecast(end_date)
    pred_low_ci = pred_low.conf_int()
    fc_series_low = pred_low.predicted_mean
    # lower_series_low = pnd.Series(conf[:, 0], index=index)
    # upper_series_low = pnd.Series(conf[:, 1], index=index)

    index = pnd.date_range(start_date, end_date, freq='M').strftime('%Y-%m')
    pred_high = results_high.get_forecast(end_date)
    pred_high_ci = pred_high.conf_int()
    fc_series_high = pred_high.predicted_mean
    # lower_series_high = pnd.Series(conf[:, 0], index=index)
    # upper_series_chigh = pnd.Series(conf[:, 1], index=index)

    ax = test_data_close.plot(y='close', figsize=(20, 18))
    bx = test_data_open.plot(y='open', figsize=(20, 18))
    pred_close.predicted_mean.plot(ax=ax, label='Dynamic Forecast close (get_forecast)')
    pred_open.predicted_mean.plot(ax=bx, label='Dynamic Forecast open (get_forecast)')
    ax.fill_between(pred_close_ci.index, pred_close_ci.iloc[:, 0], pred_close_ci.iloc[:, 1], color='k', alpha=.1)
    bx.fill_between(pred_open_ci.index, pred_open_ci.iloc[:, 0], pred_open_ci.iloc[:, 1], color='k', alpha=.1)
    # plt.plot(fc_series_close, label='forecast_close')
    # plt.plot(fc_series_open, label='forecast_open')
    # plt.plot(fc_series_low, label='forecast_low')
    # plt.plot(fc_series_high, label='forecast_high')
    # plt.xlim(left=start_date, right=end_date)
    # plt.ylabel('close')
    # plt.fill_between(lower_series_close.index, lower_series_close, upper_series_close,
    #                  color='k', alpha=.15)
    # plt.fill_between(lower_series_low.index, lower_series_low, upper_series_low,
    #                  color='k', alpha=.15)
    plt.xlabel('date')
    plt.legend()
    plt.show()

    print(len(index), len(fc_series_close))
    frame = {'date': index, 'close': fc_series_close, 'low': fc_series_low, 'open': fc_series_open,
             'high': fc_series_high}
    new_df = pnd.DataFrame(frame)
    new_df = new_df.reset_index()
    new_df.head()

    new_df['SMA'] = TA.SMA(new_df, 12)
    new_df['RSI'] = TA.RSI(new_df)
    new_df['MACD'] = 0
    new_df['MOM'] = TA.MOM(new_df)
    new_df['MACD'] = TA.MACD(new_df)
    new_df['EMA'] = TA.EMA(new_df)
    new_df.fillna(0, inplace=True)

    env = DummyVecEnv([lambda: StockTradingEnv(df=new_df)])

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=50)

    # model.save('stock_traiding_model')

    # model = PPO2.load('stock_traiding_model')

    obs = env.reset()

    # obs_first = obs[np.newaxis, ...]
    # action, _states = model.predict(obs_first)
    # obs, rewards, done, info = env.step(action)
    # print(obs, done)
    # obs_second = obs[np.newaxis, ...]
    # action, _states = model.predict(obs_second)
    # obs, rewards, done, info = env.step(action)
    # print(obs)
    # print('info', info)

    root = tk.Tk()
    while True:
        #     action, _states = model.predict(obs)
        #     obs, rewards, done, info = env.step(action)
        # obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode='live')
        # plt.imshow(img)
        # plt.axis("off")

        if done:
            print('info', info)
            break

    # canvas = Canvas(root, width=300, height=300)
    # canvas.pack()
    # canvas.create_image(image=img)

    # figure1 = plt.figure(figsize=(20, 10))
    # ax1 = figure1.add_subplot(111)
    # bar1 = FigureCanvasTkAgg(figure1, root)
    # bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


    mainloop()
