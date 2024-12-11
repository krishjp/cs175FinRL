# Import List
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
print(dir(StockTradingEnv))

from finrl.agents.stablebaselines3.models import DRLAgent

from pathlib import Path
import itertools
import os
import gc

def get_tickers(filepath: str, top_x: int):
    # all the tickers
    if (filepath == 'sp500.txt'):
        tickers = []
        with open(Path('sp500.txt'), 'r') as r:
            lines = r.readlines()
            for i, line in enumerate(lines):
                if i < 2:
                    continue
                tickers.append(line.split('\t')[0])
        return tickers[:top_x]
    
    elif (filepath == 'sp500test.txt'):
        tickers = []
        with open(Path('sp500test.txt'), 'r') as r:
            lines = r.readlines()
            for i, line in enumerate(lines):
                if i < 2:
                    continue
                tickers.append(line.split('\t')[0])
        return tickers[:top_x]
    else:
        print("Recognised Ticker File Not Found")


def download_yahoo_data(tickers: list):
    data = YahooDownloader(start_date = '2010-01-01', end_date = '2024-11-01', ticker_list = tickers).fetch_data()
    return data


def feature_engine(downloaded_yahoo_data):
    # Adding extra feature engineering

    fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list = INDICATORS, 
                        use_vix=True, use_turbulence=True, user_defined_feature = False)

    processed = fe.preprocess_data(downloaded_yahoo_data)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)

    return processed_full

def create_data_splits(processed_full):
    #Train/test splits
    #changed train start to 2010 to match data downloading changes
    train_start = '2010-01-01'
    train_end = '2023-07-01'
    trade_start = '2023-07-01'
    trade_end = '2024-11-01'

    train = data_split(processed_full, train_start,train_end)
    trade = data_split(processed_full, trade_start,trade_end)

    train = train.reset_index()

    train = train.set_index(train.columns[0])
    train.index.names = ['']
    return train, trade

def load_env(train):
    # Build the action and state spaces
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    # High risk, free reign:
    env_kwargs_highRisk_free = {
        "hmax": 1000,
        "initial_amount": 500000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e2,
        "model_name": "Mproc_highRisk_free",
        "mode" : "PPO_hmax:1k_cash:500k_steps:500k"
    }

    # Tiny reward scaling, free reign:
    env_kwargs_tiny_free = {
        "hmax": 1000,
        "initial_amount": 500000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-8,
        "model_name": "Mproc_tiny_free",
        "mode" : "PPO_hmax:1k_cash:500k_steps:100k"
    }

    # Tiny reward scaling, limited and must diversify:
    env_kwargs_limit_diversity = {
        "hmax": 10,
        "initial_amount": 500000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-8,
        "model_name": "MProc_limit_div",
        "mode" : "PPO_hmax:10_cash:500k_steps:100k"
    }

    # Buy only
    env_kwargs_buy = {
        "hmax": 75,
        "initial_amount": 500000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": [1.0] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "model_name": "SProc_buy",
        "mode" : "PPO_hmax:75_cash:500k_steps:500"
    }

    # Sell only
    env_kwargs_sell = {
        "hmax": 75,
        "initial_amount": 500000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": [1.0] * stock_dimension,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "model_name": "SProc_sell",
        "mode" : "PPO_hmax:75_cash:500k_steps:500"
    }

    choosenOne = env_kwargs_tiny_free

    e_train_gym = StockTradingEnv(df = train, **choosenOne)
    print(choosenOne['model_name'])
    print(choosenOne['mode'])

    #env_train, obs_trade = e_train_gym.get_sb_env()
    env_train, obs_trade = e_train_gym.get_multiproc_env()

    return env_train, choosenOne


if __name__ == "__main__":
    start_time = time.time()

    if not os.path.exists('results'):
        os.makedirs('results')
    
    if not os.path.exists('results/csvBank'):
        os.makedirs('results/csvBank')
    
    tickers = get_tickers('sp500.txt', 500)
    data = download_yahoo_data(tickers)

    done_download_time = time.time()
    done_download_elapse = (done_download_time - start_time) / 60
    print(f"Download time: {done_download_elapse:.2f} minutes")
    print()

    # TODO the downlaod time does not ouput until the deature engineering is done
    # tempting 4

    processed_full = feature_engine(data)

    done_feature_time = time.time()
    done_feature_elapse = (done_feature_time - start_time) / 60
    print(f"Feature time: {done_feature_elapse:.2f} minutes")

    train, trade = create_data_splits(processed_full)
    env_train, env_kwargs = load_env(train)
    
    agent = PPO('MlpPolicy', env_train, verbose=1)
    agent.learn(total_timesteps=100000)

    del train
    gc.collect()

    env_kwargs['model_name'] += "_evalMode"

    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
    env_trade, _ = e_trade_gym.get_sb_env() #use one proc for testing as multiprocs requires alterative methods
    
    #run test eval
    obs_trade = env_trade.reset()
    done = False
    while not done:
        action, _states = agent.predict(obs_trade)
        obs_trade, rewards, done, info = env_trade.step(action)
    
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=agent, environment = e_trade_gym)

    daily_returns = df_account_value['account_value'].pct_change()
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns)

    # Calculate maximum drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Maximum Drawdown: {max_drawdown}")
    print(f"Max Cumulative Returns: {cumulative_returns.max()}")

    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")

