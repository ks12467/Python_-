#analyzer
from ta.momentum import RSIIndicator
from ta.trend import MACD

def is_uptrend(df, window=5):
    df['ma'] = df['close'].rolling(window).mean()
    return df['close'].iloc[-1] > df['ma'].iloc[-1]

def get_rsi(df, period=14):
    return RSIIndicator(close=df['close'], window=period).rsi().iloc[-1]

def get_macd_signal(df):
    macd = MACD(close=df['close'])
    return macd.macd().iloc[-1], macd.macd_signal().iloc[-1]