import itertools
import warnings
import tkinter as tk
from finta import TA
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import statsmodels.api as sm
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
import pandas as pnd
from matplotlib import pyplot as plt
import random
from tkinter import messagebox
import gym
from gym import spaces
from tkinter import ttk
import numpy as np
from lib.render.render import StockTradingGraph

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 10
#MAX_SHARE_PRICE = 50
MAX_STEPS = 20000

MAX_OPEN_POSITIONS = 5
# INITIAL_ACCOUNT_BALANCE = 10000

LOOKBACK_WINDOW_SIZE = 20

global MAX_SHARE_PRICE, INITIAL_ACCOUNT_BALANCE

def ok_button_click():
    global MAX_SHARE_PRICE
    if len(max_share_price.get()) != 0:
        try:
            MAX_SHARE_PRICE = int(max_share_price.get())
        except ValueError:
            messagebox.showwarning('Ошибка', 'Введите целое число!')
            return
    else:
        MAX_SHARE_PRICE = 5000

    global INITIAL_ACCOUNT_BALANCE
    if len(account_balance.get()) != 0:
        try:
            INITIAL_ACCOUNT_BALANCE = int(account_balance.get())
        except ValueError:
            messagebox.showwarning('Ошибка', 'Введите целое число!')
            return
    else:
        INITIAL_ACCOUNT_BALANCE = 10000

    global RENDER_MODE
    if render_style.get() == render_style['values'][1]:
        RENDER_MODE = 'file'
    else:
        RENDER_MODE = 'live'
    print(RENDER_MODE)

    main()


def factor_pairs(val):
    return [(i, val / i) for i in range(1, int(val ** 0.5) + 1) if val % i == 0]


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, LOOKBACK_WINDOW_SIZE + 2), dtype=np.float16)

    def _next_observation(self):
        frame = np.zeros((4, LOOKBACK_WINDOW_SIZE + 1))

        # Get the stock data points for the last 5 days and scale to between 0-1
        np.put(frame, [0, 3], [
            self.df.loc[self.current_step: self.current_step + LOOKBACK_WINDOW_SIZE, 'open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step + LOOKBACK_WINDOW_SIZE, 'high'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step + LOOKBACK_WINDOW_SIZE, 'low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step + LOOKBACK_WINDOW_SIZE, 'close'].values / MAX_SHARE_PRICE,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [
            [self.balance / MAX_ACCOUNT_BALANCE],
            [self.max_net_worth / MAX_ACCOUNT_BALANCE],
            [self.shares_held / MAX_NUM_SHARES],
            [self.cost_basis / MAX_SHARE_PRICE],
        ], axis=1)

        return obs

    def _take_action(self, action):
        current_price = random.uniform(
            self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                                      prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

            if shares_bought > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': shares_bought, 'total': additional_cost,
                                    'type': "buy"})

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

            if shares_sold > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': shares_sold, 'total': shares_sold * current_price,
                                    'type': "sell"})

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier + self.current_step
        done = self.net_worth <= 0 or self.current_step >= len(
            self.df.loc[:, 'open'].values)

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0
        self.trades = []

        return self._next_observation()

    def _render_to_file(self, filename='render.txt'):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        file = open(filename, 'a+')

        file.write(f'Step: {self.current_step}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})\n')
        file.write(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})\n')
        file.write(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})\n')
        file.write(f'Profit: {profit}\n\n')

        file.close()

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'file':
            self._render_to_file(kwargs.get('filename', 'render.txt'))

        elif mode == 'live':
            if self.visualization == None:
                self.visualization = StockTradingGraph(
                    self.df, kwargs.get('title', None))

            if self.current_step > LOOKBACK_WINDOW_SIZE:
                self.visualization.render(
                    self.current_step, self.net_worth, self.trades, window_size=LOOKBACK_WINDOW_SIZE)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None


def main():
    df = pnd.read_csv('datasets/VZ.csv')
    df['date'] = pnd.to_datetime(df['date'], format='%Y-%m-%d')
    df.sort_values('date', ascending=True, inplace=True)
    df.date.dt.to_period('M')
    df.drop(columns=['adjclose'], inplace=True)
    df = df.groupby(pnd.DatetimeIndex(df.date).to_period('M')).nth(0)
    df.drop(columns=['date'], inplace=True)
    start_date = pnd.to_datetime('2015-01')
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
    AIC_temp = []
    SARIMAX_model_temp = []
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

    frame = {'date': index, 'close': fc_series_close, 'low': fc_series_low, 'open': fc_series_open,
             'high': fc_series_high}
    new_df = pnd.DataFrame(frame)
    new_df = new_df.reset_index()

    new_df['SMA'] = TA.SMA(new_df, 12)
    new_df['RSI'] = TA.RSI(new_df)
    new_df['MACD'] = 0
    new_df['MOM'] = TA.MOM(new_df)
    new_df['MACD'] = TA.MACD(new_df)
    new_df['EMA'] = TA.EMA(new_df)
    new_df.fillna(0, inplace=True)

    env = DummyVecEnv([lambda: StockTradingEnv(df=new_df)])

    # model = PPO2(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=75000)

    # model.save('stock_traiding_model_20')

    model = PPO2.load('stock_traiding_model_20')

    obs = env.reset()

    root = tk.Tk()

    while True:
        #     action, _states = model.predict(obs)
        #     obs, rewards, done, info = env.step(action)
        # obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode=RENDER_MODE)
        # plt.imshow(img)
        # plt.axis("off")

        if done:
            print('info', info)
            break

    # canvas = Canvas(root, width=300, height=300)
    # canvas.pack()
    # canvas.create_image(image=img)

    fig_open = plt.Figure(figsize=(20, 15), dpi=100)
    ax_open = fig_open.add_subplot(111)
    bar_open = FigureCanvasTkAgg(fig_open, root)
    bar_open.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    ax = test_data_open.plot(y='open', ax=ax_open, ylim=(35, 60))
    ax_open.set_title('Open and Close prediction')
    pred_open.predicted_mean.plot(ax=ax, label='Dynamic Forecast open (get_forecast)')

    fig_close = plt.Figure(figsize=(20, 15), dpi=100)
    ax_close = fig_close.add_subplot(111)
    bar_close = FigureCanvasTkAgg(fig_close, root)
    bar_close.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    ax = test_data_close.plot(y='close', ax=ax_open)
    ax_close.set_title('CLOSE')
    pred_close.predicted_mean.plot(ax=ax, label='Dynamic Forecast close (get_forecast)')

    # pred_close.predicted_mean.plot(ax=ax, label='Dynamic Forecast close (get_forecast)')
    # bx = test_data_close.plot(y='close', figsize=(20, 18))
    # ax.fill_between(pred_close_ci.index, pred_close_ci.iloc[:, 0], pred_close_ci.iloc[:, 1], color='k', alpha=.1)
    # bx.fill_between(pred_open_ci.index, pred_open_ci.iloc[:, 0], pred_open_ci.iloc[:, 1], color='k', alpha=.1)
    #
    # figure1 = plt.figure(figsize=(20, 10))
    # ax1 = figure1.add_subplot(111)
    # bar1 = FigureCanvasTkAgg(figure1, root)
    # bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    # plt.show(block=False)
    mainloop()


def exit_app():
    window.destroy()


if __name__ == '__main__':
    window = tk.Tk()
    window.title("Долгосрочное планирование активов")
    window.geometry("640x480")

    max_share_price_label = tk.Label(text="Ограничение суммы покупки")
    max_share_price_label.grid(row=1, column=1, padx=60, pady=30)
    account_balance_label = tk.Label(text="Баланс аккаунта")
    account_balance_label.grid(row=2, column=1, padx=60, pady=30)

    max_share_price = tk.Entry()
    max_share_price.grid(row=1, column=3, padx=20, pady=30)
    account_balance = tk.Entry()
    account_balance.grid(row=2, column=3, padx=20, pady=30)

    account_balance_label = tk.Label(text="Стиль представления данных")
    account_balance_label.grid(row=3, column=1, padx=60, pady=30)
    render_style = ttk.Combobox()
    render_style['values'] = ('В реальном времени', 'Вывод в файл')
    render_style.current(1)
    render_style.grid(row=3, column=3, padx=20, pady=30)

    button = tk.Button(window, text='Задать значения', command=ok_button_click)

    button.place(x=150, y=400)

    exit_button = tk.Button(window, text='Выйти из программы', command=exit_app)

    exit_button.place(x=330, y=400)
    window.mainloop()
