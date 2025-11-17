# main.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import math
import plotly.graph_objs as go

# ---------------------------
# Helper indicator functions
# ---------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def true_range(df):
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df, period=14):
    tr = true_range(df)
    return tr.rolling(window=period, min_periods=1).mean()

def rsi(df, period=14):
    delta = df['close'].diff()
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    # Wilder smoothing
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------------------------
# Fetch OHLCV using ccxt
# ---------------------------
def get_exchange(binance_public=True):
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    # If you want to use API keys (for higher rate limits), set via env or streamlit secrets.
    return exchange

def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 500):
    # ccxt returns [ts, open, high, low, close, volume]
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

# ---------------------------
# Signal logic
# ---------------------------
def generate_signal(df, params):
    """Return dict: signal, entry, tp, sl, reason, confidence"""
    # require at least some rows
    if len(df) < max(params['ema_short'], params['ema_long'], params['rsi_period']):
        return None

    df = df.copy()
    df['ema_short'] = ema(df['close'], params['ema_short'])
    df['ema_long'] = ema(df['close'], params['ema_long'])
    df['rsi'] = rsi(df, params['rsi_period'])
    df['atr'] = atr(df, params['atr_period'])

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    ema_short_now = last['ema_short']
    ema_long_now = last['ema_long']
    ema_short_prev = prev['ema_short']
    ema_long_prev = prev['ema_long']

    rsi_now = last['rsi']
    atr_now = last['atr']
    close_now = last['close']
    high_now = last['high']
    low_now = last['low']

    # Basic crossover + RSI filter
    signal = 'neutral'
    reason = []
    confidence = 0.0

    # bullish crossover (short crossed above long)
    if (ema_short_prev <= ema_long_prev) and (ema_short_now > ema_long_now):
        # bullish potential
        if rsi_now < params['rsi_overbought']:
            signal = 'long'
            reason.append('EMA bullish crossover')
            confidence += 0.6
        else:
            reason.append('EMA bullish crossover but RSI too high')
    # bearish crossover
    elif (ema_short_prev >= ema_long_prev) and (ema_short_now < ema_long_now):
        if rsi_now > params['rsi_oversold']:
            signal = 'short'
            reason.append('EMA bearish crossover')
            confidence += 0.6
        else:
            reason.append('EMA bearish crossover but RSI too low')
    else:
        reason.append('No EMA crossover')

    # price momentum confirmation (close vs previous high/low breakout)
    if signal == 'long':
        if close_now > prev['high']:
            confidence += 0.2
            reason.append('Breakout above previous high')
    elif signal == 'short':
        if close_now < prev['low']:
            confidence += 0.2
            reason.append('Breakdown below previous low')

    # finalize entry, tp, sl using ATR multiples
    if signal in ('long', 'short'):
        atr_mul = params['atr_multiplier']
        if signal == 'long':
            entry = close_now
            sl = entry - atr_now * atr_mul
            tp = entry + atr_now * atr_mul * params['tp_atr_factor']
        else:
            entry = close_now
            sl = entry + atr_now * atr_mul
            tp = entry - atr_now * atr_mul * params['tp_atr_factor']

        # Safety: ensure tp/sl not equal to entry
        if abs(tp - entry) < 1e-8 or abs(sl - entry) < 1e-8:
            return None

        # Confidence normalization 0..1
        confidence = min(1.0, confidence)
        return {
            'signal': signal,
            'entry': float(round(entry, params['price_precision'])),
            'tp': float(round(tp, params['price_precision'])),
            'sl': float(round(sl, params['price_precision'])),
            'reason': '; '.join(reason),
            'confidence': round(float(confidence), 2),
            'rsi': round(float(rsi_now), 1),
            'atr': round(float(atr_now), 8)
        }
    else:
        return {
            'signal': 'neutral',
            'reason': '; '.join(reason),
            'confidence': round(float(confidence), 2),
            'rsi': round(float(rsi_now), 1),
            'atr': round(float(atr_now), 8)
        }

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Crypto Multi-TF Signal Bot", layout="wide")

st.title("AI-like Crypto Signal Bot (multi-timeframe) — Streamlit")
st.markdown("""
Generates entry / TP / SL signals across multiple timeframes using EMA crossover + RSI + ATR sizing.
**Not financial advice. Backtest before trading.**
""")

col1, col2 = st.columns([2,1])

with col1:
    symbol = st.text_input("Symbol (Binance format)", value="BTC/USDT")
    tf_options = ["1m","3m","5m","15m","30m","1h"]
    selected_tfs = st.multiselect("Select timeframes", tf_options, default=tf_options)
    update_interval = st.slider("Auto-refresh (seconds)", min_value=10, max_value=600, value=30, step=5)
    max_bars = st.number_input("History bars to fetch per timeframe", value=500, min_value=50, max_value=2000, step=50)
    price_precision = st.number_input("Price rounding decimals", value=2, min_value=0, max_value=8, step=1)

with col2:
    st.markdown("### Strategy parameters")
    ema_short = st.number_input("EMA short period", value=9, min_value=2, max_value=200)
    ema_long = st.number_input("EMA long period", value=21, min_value=2, max_value=400)
    rsi_period = st.number_input("RSI period", value=14, min_value=2, max_value=50)
    rsi_overbought = st.number_input("RSI overbought threshold (for short signals)", value=70)
    rsi_oversold = st.number_input("RSI oversold threshold (for long signals)", value=30)
    atr_period = st.number_input("ATR period", value=14)
    atr_multiplier = st.number_input("ATR multiplier for SL", value=1.5, min_value=0.1, step=0.1)
    tp_atr_factor = st.number_input("TP factor (TP = entry +/- ATR * multiplier * factor)", value=2.0, min_value=0.1, step=0.1)

params = {
    'ema_short': int(ema_short),
    'ema_long': int(ema_long),
    'rsi_period': int(rsi_period),
    'rsi_overbought': float(rsi_overbought),
    'rsi_oversold': float(rsi_oversold),
    'atr_period': int(atr_period),
    'atr_multiplier': float(atr_multiplier),
    'tp_atr_factor': float(tp_atr_factor),
    'price_precision': int(price_precision)
}

st.markdown("---")

exchange = get_exchange()

placeholder = st.empty()

# Auto refresh loop with Streamlit's autorefresh
from streamlit_autorefresh import st_autorefresh
count = st_autorefresh(interval=update_interval * 1000, limit=None, key="autoreload")

with placeholder.container():
    results = []
    errors = []
    for tf in selected_tfs:
        try:
            df = fetch_ohlcv(exchange, symbol, timeframe=tf, limit=max_bars)
            sig = generate_signal(df, params)
            row = {
                'timeframe': tf,
                'signal': sig['signal'] if sig else 'error',
                'entry': sig.get('entry') if sig and 'entry' in sig else None,
                'tp': sig.get('tp') if sig and 'tp' in sig else None,
                'sl': sig.get('sl') if sig and 'sl' in sig else None,
                'confidence': sig.get('confidence') if sig else None,
                'rsi': sig.get('rsi') if sig else None,
                'reason': sig.get('reason') if sig else 'no data'
            }
            results.append(row)
        except Exception as e:
            errors.append(f"{tf}: {e}")

    df_results = pd.DataFrame(results).set_index('timeframe')
    st.subheader(f"Signals for {symbol}")
    st.table(df_results)

    if errors:
        st.error("Errors fetching some timeframes. See console for details.")
        st.write(errors)

    # Plot last timeframe candle + indicators if any TF selected
    if len(selected_tfs) > 0:
        tf_plot = selected_tfs[0]
        try:
            df_plot = fetch_ohlcv(exchange, symbol, timeframe=tf_plot, limit=200)
            df_plot['ema_short'] = ema(df_plot['close'], params['ema_short'])
            df_plot['ema_long'] = ema(df_plot['close'], params['ema_long'])
            fig = go.Figure(data=[go.Candlestick(x=df_plot.index,
                                                 open=df_plot['open'],
                                                 high=df_plot['high'],
                                                 low=df_plot['low'],
                                                 close=df_plot['close'],
                                                 name='Price')])
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema_short'], name=f'EMA{params["ema_short"]}'))
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema_long'], name=f'EMA{params["ema_long"]}'))
            fig.update_layout(height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write("Could not plot chart:", e)

st.markdown("---")
st.caption("Built with ccxt + Streamlit. Edit strategy parameters above. This app polls public market data only.")
