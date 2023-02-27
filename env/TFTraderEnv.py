import process_data
import pandas as pd
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
#from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import deque
import os
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from ta.trend import EMAIndicator, ADXIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator

LONG = 0
SHORT = 1
FLAT = 2

BUY = 0
SELL = 1
HOLD = 2

class OhlcvEnv(gym.Env):

    def __init__(self, window_size, path, train=True, show_trade=True):
        self.train= train
        self.show_trade = show_trade
        self.df = path
        print('COMECA AQUI', self.df)
        self.actions = ["LONG", "SHORT", "FLAT"]        
        self.seed()
        self.file_list = []        
        
        self.window_size = window_size
        
        #self.df.dropna(inplace=True)
        self.closingPrices = self.df['close'].values
        #self.df = self.df[['open','high','low','close','volume', 'vwap', 'ema9', 'ema21', 'ema200','sma200','adx','-di', '+di','rsi', 'volume_21',"close_win",'ema9_win', 'ema21_win']].values
        self.df = self.df[['open','high','low','close','volume',"close_win", 'volume_win']].values
        #self.df = self.df[['pct_open','pct_high','pct_low','pct_close','pct_volume',"pct_close_win", 'pct_volume_win']].values
        #self.df = self.df[['bar_hc','bar_ho','bar_hl','bar_cl','bar_ol','bar_co','bar_mov','bar_mov_vol','bar_mov_close_win','bar_mov_volume_win']].values
        #self.df = self.df[['open/10000','high/10000','low/10000','close/10000','volume/10000000000','close_win/10000','volume_win/10000000000' ]].values
        #self.df = self.df[['open','high','low','close','volume/10000000000']].values
        #self.df = self.df[['close']].values
        #self.df = self.df[['open','high','low','close','volume', 'bb_bbh', 'bb_bbl', 'vwap', 'ema9', 'ema21', 'ema200','rsi','adx','-di', '+di', 'volume_21', 'estocastico', 'estocastico_signal']].values
        #self.df = self.df[['open','high','low','close','volume']].values
        print('N_FEATURES', self.df.shape)
        self.n_features = self.df.shape[1]
        #self.shape = (self.window_size, self.n_features +4, 1)
        self.shape = (self.window_size, self.n_features +4)
        # defines action space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
        print('second', self.df)

    #def load_from_csv(self):
        #if(len(self.file_list) == 0):
        #    self.file_list = [x.name for x in Path(self.path).iterdir() if x.is_file()]
        #    self.file_list.sort()
        #self.rand_episode = self.file_list.pop()
    #    names = ['Data','timestamp','open','high','low','close','volume']
    #    raw_df= pd.read_csv('WDO_15MIN.csv' , header = 0, names = names,  sep = ';', index_col=False)
    #    print(raw_df)
    #    #extractor = process_data.FeatureExtractor(raw_df)
    #    #self.df = extractor.add_bar_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features
    #    self.df = raw_df  
    #    self.df.dropna(inplace=True) # drops Nan rows
    #    self.closingPrices = self.df['close'].values
    #    indicator_bb = BollingerBands(close=self.df["close"], window=20, window_dev=2, fillna = True)
    #    vwap = VolumeWeightedAveragePrice(high=self.df["high"], low=self.df["low"], close=self.df["close"], volume=self.df["volume"], fillna=True )
    #    ema9 = EMAIndicator(close=self.df["close"], window=9, fillna=True )
    #    ema21 = EMAIndicator(close=self.df["close"], window=21, fillna=True )
    #   ema200 = EMAIndicator(close=self.df["close"], window=200, fillna=True )
    #    rsi = RSIIndicator(close=self.df["close"], fillna=True)
    #    adx = ADXIndicator(high=self.df["high"], low=self.df["low"], close=self.df["close"], window = 8, fillna=True)
    #   volume_21 = SMAIndicator(close=self.df["volume"], window=21, fillna=True )
    #    estocastico = StochasticOscillator(high=self.df["high"], low=self.df["low"], close=self.df["close"], fillna=True)
    #    self.df['bb_bbm'] = indicator_bb.bollinger_mavg()
    #    self.df['bb_bbh'] = indicator_bb.bollinger_hband()
    #    self.df['bb_bbl'] = indicator_bb.bollinger_lband()
    #    self.df['vwap'] = vwap.volume_weighted_average_price()
    #    self.df['ema9'] = ema9.ema_indicator()
    #    self.df['ema21'] = ema9.ema_indicator() 
    #    self.df['ema200'] = ema9.ema_indicator()
    #    self.df['rsi'] = rsi.rsi()
    #    self.df['adx'] = adx.adx() 
    #    self.df['-di'] = adx.adx_neg()
    #    self.df['+di'] = adx.adx_pos()
    #    self.df['volume_21'] = volume_21.sma_indicator()
    #    self.df['estocastico']  = estocastico.stoch()
    #    self.df['estocastico_signal'] = estocastico.stoch_signal()
    #    print(self.df)
    #    self.df = self.df[['open','high','low','close','volume', 'bb_bbh', 'bb_bbl', 'vwap', 'ema9', 'ema21', 'ema200','rsi','adx','-di', '+di', 'volume_21', 'estocastico', 'estocastico_signal']].values
        #self.df = self.df[['open','high','low','close','volume', 'vwap', 'ema9', 'ema21', 'ema200','rsi', 'volume_21']].values
        #self.df = self.df.values
    #    print('second', self.df)

    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normalize_frame(self, frame):
        offline_scaler = MinMaxScaler()
        observe = frame[..., :-3]
        #print(observe)
        observe = offline_scaler.fit_transform(observe)
        agent_state = frame[..., -3:]
        temp = np.concatenate((observe, agent_state), axis=1)
        #print('QUERO ESSE', temp.shape)
        #temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
        #print('QUERO ESSE', temp.shape)
        return temp

    def step(self, action):
        s, r, d, i = self._step(action)
        self.state_queue.append(s)

        #print(s)
        return self.normalize_frame(np.concatenate(tuple(self.state_queue))), r, d, i

    def _step(self, action):

        if self.done:
            return self.state, self.reward, self.done, {}
        self.reward = 0
        self.action = HOLD  
        if action == BUY: 
            if self.position == FLAT: 
                self.position = LONG 
                self.action = BUY 
                self.entry_price = self.closingPrice 
            elif self.position == SHORT:
                self.position = FLAT  
                self.action = BUY 
                self.exit_price = self.closingPrice
                fee = (self.closingPrice/1000)*0.4
                self.reward += (((self.entry_price - self.exit_price)*10)-fee)/10
                self.krw_balance = self.krw_balance + self.reward
                #self.reward += ((self.entry_price - self.exit_price)/self.exit_price + 1)*(1-self.fee)**2 - 1 # calculate reward
                #self.krw_balance = self.krw_balance * (1.0 + self.reward) # evaluate cumulative return in krw-won
                self.entry_price = 0 
                self.n_short += 1 
        elif action == 1: 
            if self.position == FLAT:
                self.position = SHORT
                self.action = 1
                self.entry_price = self.closingPrice
            elif self.position == LONG:
                self.position = FLAT
                self.action = 1
                self.exit_price = self.closingPrice
                fee = (self.closingPrice/1000)*0.4
                self.reward += (((self.exit_price - self.entry_price )*10)-fee)/10
                self.krw_balance = self.krw_balance + self.reward
                #self.reward += ((self.exit_price - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
                #self.krw_balance = self.krw_balance * (1.0 + self.reward)
                self.entry_price = 0
                self.n_long += 1
       
        if(self.position == LONG):
            fee = (self.closingPrice/1000)*0.4
            temp_reward = (((self.closingPrice - self.entry_price)*10)-fee)/10 
            new_portfolio = self.krw_balance + temp_reward
            #temp_reward = ((self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
            #new_portfolio = self.krw_balance * (1.0 + temp_reward)
        elif(self.position == SHORT):
            fee = (self.closingPrice/1000)*0.4
            temp_reward = (((self.entry_price - self.closingPrice)*10)-fee)/10
            new_portfolio = self.krw_balance + temp_reward
            #temp_reward = ((self.entry_price - self.closingPrice)/self.closingPrice + 1)*(1-self.fee)**2 - 1
            #new_portfolio = self.krw_balance * (1.0 + temp_reward)
        else:
            temp_reward = 0
            new_portfolio = self.krw_balance

        self.portfolio = new_portfolio
        self.current_tick += 1
        self.history.append((self.action, self.current_tick, self.closingPrice, self.portfolio, self.reward))
        if(self.show_trade and self.current_tick%10 == 0):
            #print("Tick: {2}/ Action: {0}/ Closing price: {1} ".format(self.action, self.closingPrice,self.current_tick))
            print("Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
            #print(self.history[-1])
            #print(self.krw_balance)
        
        self.state = self.updateState()
        info = {'portfolio':np.array([self.portfolio]),
                "history":self.history,
                "n_trades":{'long':self.n_long, 'short':self.n_short}}
        if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
            self.done = True
            self.reward = self.get_profit()
            if(self.train == False):
                np.array([info]).dump('./info/ppo_{0}_LS_{1}_{2}.info'.format(self.portfolio, self.n_long, self.n_short))
        return self.state, self.reward, self.done, info

    def get_profit(self):
        if(self.position == LONG):
            fee = (self.closingPrice/1000)*0.4
            profit = (((self.closingPrice - self.entry_price)*10)-fee)/10 
            #profit = ((self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
        elif(self.position == SHORT):
            fee = (self.closingPrice/1000)*0.4
            profit = (((self.entry_price - self.closingPrice)*10)-fee)/10            
            #profit = ((self.entry_price - self.closingPrice)/self.closingPrice + 1)*(1-self.fee)**2 - 1
        else:
            profit = 0
        return profit

    def reset(self):
        ## self.current_tick = random.randint(0, self.df.shape[0]-800)
        #if(self.train):
        #    self.current_tick = random.randint(0, self.df.shape[0] - 25)
        #else:
        self.current_tick = 0
        print("START ENV")
        #print("start episode ... {0} at {1}" .format(self.rand_episode, self.current_tick))
        self.n_long = 0
        self.n_short = 0
        self.history = [] 
        self.krw_balance = 0 
        self.portfolio = float(self.krw_balance) 
        self.profit = 0
        self.closingPrice = self.closingPrices[self.current_tick]

        self.action = HOLD
        self.position = FLAT
        self.done = False

        self.state_queue = deque(maxlen=self.window_size)
        self.state = self.preheat_queue()
        return self.state


    def preheat_queue(self):
        while(len(self.state_queue) < self.window_size):
            # rand_action = random.randint(0, len(self.actions)-1)
            rand_action = 2
            s, r, d, i= self._step(rand_action)
            self.state_queue.append(s)
        return self.normalize_frame(np.concatenate(tuple(self.state_queue)))

    def updateState(self):
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]
        self.closingPrice = float(self.closingPrices[self.current_tick])
        prev_position = self.position
        one_hot_position = one_hot_encode(prev_position,3)
        profit = self.get_profit()
        # append two        
        #state = self.df[self.current_tick]
        state = np.concatenate((self.df[self.current_tick],[profit], one_hot_position))
        return state.reshape(1,-1)
