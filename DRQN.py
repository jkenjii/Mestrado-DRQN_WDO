import numpy as np

from env.gymWrapper import create_btc_env
from dqn_agent import DQNAgent
import os

import pandas as pd
import numpy as np
import math
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice, MFIIndicator
from ta.trend import EMAIndicator, ADXIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator

import tensorflow as tf
from tensorflow import keras

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
#sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(
    tf.compat.v1.Session(config=config))

class DADOS():    

    def __init__(self):
        
        names = ['Data','time','open','high','low','close', 'volume','open_win','high_win','low_win','close_win', 'volume_win']
        #names = ['Data','open','high','low','close', 'volume','open_win','high_win','low_win','close_win', 'volume_win']
        
        #names = ['DATE','TIME','open','high','low','close','TICKVOL','volume','open_win','high_win','low_win','close_win','TICKVOL_WIN','volume_win']
        raw_df= pd.read_csv('DATASET_PARA_TREINO_FINAL_26_01_2022_15MIN.csv' , header = 0, names = names,  sep = ';', index_col=False)
        print(raw_df)
        #extractor = process_data.FeatureExtractor(raw_df)
        #self.df = extractor.add_bar_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features
        self.df = raw_df  
        #self.df.dropna(inplace=True) # drops Nan rows
        self.closingPrices = self.df['close'].values
        indicator_bb = BollingerBands(close=self.df["close"], window=20, window_dev=2)
        vwap = VolumeWeightedAveragePrice(high=self.df["high"], low=self.df["low"], close=self.df["close"], volume=self.df["volume"] )
        ema9 = EMAIndicator(close=self.df["close"], window=8 )
        ema21 = EMAIndicator(close=self.df["close"], window=17 )
        ema200 = EMAIndicator(close=self.df["close"], window=72 )
        sma200 = SMAIndicator(close=self.df["close"],window=200)
        ema9_win = EMAIndicator(close=self.df["close_win"], window=9 )
        ema21_win = EMAIndicator(close=self.df["close_win"], window=21 )
        #ema200_win = EMAIndicator(close=self.df["close_win"], window=9, fillna=True )
        rsi = RSIIndicator(close=self.df["close"])
        adx = ADXIndicator(high=self.df["high"], low=self.df["low"], close=self.df["close"], window = 8)
        volume_21 = SMAIndicator(close=self.df["volume"], window=21 )
        mfi = MFIIndicator(high=self.df["high"], low=self.df["low"], close=self.df["close"], volume=self.df["volume"])

        self.df['bar_hc'] = self.df["high"] - self.df["close"]
        self.df['bar_ho'] = self.df["high"] - self.df["open"]
        self.df['bar_hl'] = self.df["high"] - self.df["low"]
        self.df['bar_cl'] = self.df["close"] - self.df["low"]
        self.df['bar_ol'] = self.df["open"] - self.df["low"]
        self.df['bar_co'] = self.df["close"] - self.df["open"]
        self.df['bar_mov'] = self.df['close'] - self.df['close'].shift(1)
        self.df['bar_mov_vol'] = self.df['volume'] - self.df['volume'].shift(1)
        self.df['bar_mov_close_win'] = self.df['close_win'] - self.df['close_win'].shift(1)
        self.df['bar_mov_volume_win'] = self.df['volume_win'] - self.df['volume_win'].shift(1)
        
        
        self.df['pct_open'] = self.df['open'].pct_change()
        self.df['pct_high'] = self.df['high'].pct_change()
        self.df['pct_low'] = self.df['low'].pct_change()
        self.df['pct_close'] = self.df['close'].pct_change()
        self.df['pct_volume'] = self.df['volume'].pct_change()
        self.df['pct_open_win'] = self.df['open_win'].pct_change()
        self.df['pct_high_win'] = self.df['high_win'].pct_change()
        self.df['pct_low_win'] = self.df['low_win'].pct_change()
        self.df['pct_close_win'] = self.df['close_win'].pct_change()
        self.df['pct_volume_win'] = self.df['volume_win'].pct_change()
        
                    
       
        #self.df['bar_hc','bar_ho','bar_hl','bar_cl','bar_ol','bar_co','bar_mov']
        #estocastico = StochasticOscillator(high=self.df["high"], low=self.df["low"], close=self.df["close"], fillna=True)
        '''
        self.df['bb_bbm'] = indicator_bb.bollinger_mavg()
        self.df['bb_bbh'] = indicator_bb.bollinger_hband()
        self.df['bb_bbl'] = indicator_bb.bollinger_lband()
        self.df['vwap'] = vwap.volume_weighted_average_price()
        self.df['ema9'] = ema9.ema_indicator()
        self.df['ema21'] = ema21.ema_indicator() 
        self.df['ema200'] = ema200.ema_indicator()
        self.df['sma200'] = sma200.sma_indicator()
        self.df['ema9_win'] = ema9_win.ema_indicator()
        self.df['ema21_win'] = ema21_win.ema_indicator()
        
        self.df['rsi'] = rsi.rsi()
        self.df['adx'] = adx.adx() 
        self.df['-di'] = adx.adx_neg()
        self.df['+di'] = adx.adx_pos()
        self.df['volume_21'] = volume_21.sma_indicator()
        '''
        #self.df['open/10000'] = self.df['open']/100000
        #self.df['high/10000'] = self.df['high']/100000
        #self.df['low/10000'] = self.df['low']/100000
        #self.df['close/10000'] = self.df['close']/100000
        #self.df['volume/10000000000'] = self.df['volume']/1
        #self.df['volume/10000000000'] = self.df['volume'] #META
        #self.df['open_win/10000'] = self.df['open_win']/100000
        #self.df['high_win/10000'] = self.df['high_win']/100000
        #self.df['low_win/10000'] = self.df['low_win']/100000
        #self.df['close_win/10000'] = self.df['close_win']/100000
        #self.df['volume_win/10000000000'] = self.df['volume_win']/1
        #self.df['volume_win/10000000000'] = self.df['volume_win'] #META
        #vwap_norm = VolumeWeightedAveragePrice(high=self.df["high/10000"], low=self.df["low/10000"], close=self.df["close/10000"], volume=self.df["volume/10000000000"] )
        #self.df['vwapNorm'] = vwap_norm.volume_weighted_average_price()
        #self.df['mfi'] = mfi.money_flow_index()
        #self.df['bar_mov'] = self.df['close'] - self.df['close'].shift(1)
        #self.df['estocastico']  = estocastico.stoch()
        #self.df['estocastico_signal'] = estocastico.stoch_signal()
        #self.df = self.df.dropna()
        print(self.df)
        #self.df = self.df[['open','high','low','close', 'bb_bbh', 'bb_bbl', 'ema9', 'ema21', 'ema200','rsi','adx','-di', '+di', 'estocastico', 'estocastico_signal']].values
        #self.df = self.df[['open/10000','high/10000','low/10000','close/10000','volume/10000000000','close_win/10000','volume_win/10000000000' ]].values

    def separa_dados_train(self, index: int, gap_train: int, gap_test: int, time:int):
        #tem 36 candles 1 dia  - 1253 DIAS NA TABELA TODA
        train = self.df[index-time : index+gap_train+time] #0+720+30+30=780 750ticks
        test = self.df[index+gap_train-time : index+gap_train+gap_test+time]
        return train, test, index + gap_test

        
    
def main(): #+720+720+720+720+720+720
    #index = 32912+720+720+720+720+720 /// #32912#34609-780-12             #+720+720+720+720+720+720#+720+720 #34609-780-12(2 meses)#7663(60 min) #30+6 #34609 30+6 (34609-720+12 é de 2 meses de treino)
    #gap_train =36*5*4+780+12+905 /// #36*5*4+780+12(2 meses) #2322(60 min) #36*5*4 #36*5*4 #1mes #72*15 #2 dias 72 #15 MINUTOS (36*5*4+780+12 é de 2 meses de treino)
    #gap_test = 36*5*4 /// #1 dia 36 #15 MINUTOS

    index = 28006#28006 #26674 #26306-3 #26674#30704####25931 #13083+400-36-36-18-5-1#30704+720-36-92#29336+720-73+1#28006+684-38#25932-1+725+611-1#24524+27
    gap_train = 2338#2338 #2339 #3389-30-2#2339#2448####2356#1193+20#2412+73-1#2338+38#2415-30-30+1+37+1#2345
    gap_test = 684#684 #648 #684#720####725#468-156+30#720-36#684+36#754-30+1-70+30-1 #36*5*4+20
    #gap_train = 108*5*4 #1mes #5 MINUTOS
    #gap_test = 108*5*7 #1 dia 36 #5 MINUTOS
    TIMESTEP = 30 #30
    data = DADOS()
    filename = ''
    
    for N in range(1):
                
        train, test, index_pos_new = data.separa_dados_train(index=index, gap_train = gap_train, gap_test = gap_test, time = TIMESTEP)
        print(train)
        print(len(train))
        print(test)
        index = index_pos_new 
        train_env = create_btc_env(window_size=TIMESTEP, path=train,train=True)
        if len(filename)==0:
            agent = DQNAgent(train_env, window_size_from_env = TIMESTEP)
        else:
            agent = DQNAgent(train_env, window_size_from_env = TIMESTEP, policy_network = tf.keras.models.load_model("agents/"+filename))
        
        #agent = DQNAgent(train_env, window_size_from_env = TIMESTEP)    
        filename = agent.train(n_steps=5000, n_episodes=4500, save_path="agents/")
        print(filename)
        print(len(filename))
        test_env = create_btc_env(window_size=TIMESTEP, path=test, train=True)
        agent_test = DQNAgent(test_env, window_size_from_env = TIMESTEP, policy_network = tf.keras.models.load_model("agents/"+filename))
        memory = agent_test.test(gap_test, n_episodes=3)
        #print(memory)
        print(index_pos_new)
        print(test)
        train_env = None
        test_env = None
        agent = None
        agent_test = None

#def main():

    # create environment for train and test
    #PATH_TRAIN = "./data/train/"
    #PATH_TEST = "./data/test/"
    #TIMESTEP = 30 # window size
    #env = create_btc_env(window_size=TIMESTEP, path=PATH_TRAIN, train=True)
    ##test_environment = create_btc_env(window_size=TIMESTEP, path=PATH_TEST, train=False)
    #print("STEP", env.step)#

    #agent = DQNAgent(env, window_size_from_env = TIMESTEP)
    #1129
    #reward = agent.train(n_steps=5227, n_episodes=750, save_path="agents/")
    #print(reward)

    
if __name__ == '__main__':
    main()

