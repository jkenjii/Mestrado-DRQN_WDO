
from env.TFTraderEnv import OhlcvEnv

def create_btc_env(window_size, path, train):
    raw_env = OhlcvEnv(window_size=window_size, path=path, train=train)
    #env = OpenAIGym(raw_env, visualize=False)
    return raw_env

