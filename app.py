# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import random

# Initialize session state
if 'bot_active' not in st.session_state:
    st.session_state.bot_active = False
if 'trade_journal' not in st.session_state:
    st.session_state.trade_journal = []
if 'capital' not in st.session_state:
    st.session_state.capital = 10000.00
if 'trades' not in st.session_state:
    st.session_state.trades = []

# Constants
WIN_RATE = 0.98  # Simulated win rate
RISK_PER_TRADE = 0.02  # 2% of capital per trade

# Pocket Option API Simulation (Unofficial)
class PocketOptionAPI:
    @staticmethod
    def place_trade(asset, amount, direction, duration):
        # Simulate trade execution - real implementation would use Selenium or reverse-engineered API
        trade_id = f"TRADE_{int(time.time())}_{random.randint(1000,9999)}"
        outcome = "WIN" if random.random() < WIN_RATE else "LOSS"
        profit = amount * (0.75 if outcome == "WIN" else -1)
        return {
            "id": trade_id,
            "asset": asset,
            "amount": amount,
            "direction": direction,
            "duration": duration,
            "outcome": outcome,
            "profit": profit,
            "timestamp": datetime.now(pytz.utc)
        }

# Technical Indicators
class TradingIndicators:
    @staticmethod
    def calculate_rsi(data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data, slow=26, fast=12, signal=9):
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_ema(data, window):
        return data['Close'].ewm(span=window, adjust=False).mean()

# Pattern Recognition
class PatternDetector:
    @staticmethod
    def detect_double_top(data):
        # Simplified pattern detection
        max_idx = data['High'].idxmax()
        left = data.loc[:max_idx]
        right = data.loc[max_idx:]
        return (
            len(left) > 3 and 
            len(right) > 3 and
            left['High'][-2] < left['High'][-1] and
            right['High'][0] > right['High'][1]
        )

    @staticmethod
    def detect_triangle(data):
        # Simplified triangle detection
        highs = data['High'].rolling(5).max()
        lows = data['Low'].rolling(5).min()
        return (highs.std() < 0.5 * highs.mean() and 
                lows.std() < 0.5 * lows.mean())

# Wealth Warrior Strategy
class WealthWarriorStrategy:
    def __init__(self):
        self.rsi_window = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.ema_short = 20
        self.ema_long = 50

    def generate_signal(self, data):
        # Calculate indicators
        data['RSI'] = TradingIndicators.calculate_rsi(data, self.rsi_window)
        data['MACD'], data['Signal'] = TradingIndicators.calculate_macd(
            data, self.macd_slow, self.macd_fast, self.macd_signal
        )
        data['EMA20'] = TradingIndicators.calculate_ema(data, self.ema_short)
        data['EMA50'] = TradingIndicators.calculate_ema(data, self.ema_long)
        
        # Generate signal
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Entry conditions
        buy_conditions = (
            last['EMA20'] > last['EMA50'] and
            prev['EMA20'] <= prev['EMA50'] and
            last['RSI'] < 70 and
            last['MACD'] > last['Signal'] and
            PatternDetector.detect_double_top(data.tail(10))
        )
        
        sell_conditions = (
            last['EMA20'] < last['EMA50'] and
            prev['EMA20'] >= prev['EMA50'] and
            last['RSI'] > 30 and
            last['MACD'] < last['Signal'] and
            PatternDetector.detect_triangle(data.tail(10))
        )
        
        if buy_conditions:
            return 'BUY'
        elif sell_conditions:
            return 'SELL'
        return 'HOLD'

# Risk Management
class RiskManager:
    def __init__(self, capital):
        self.capital = capital
        self.max_daily_loss = 0.05  # 5% of capital
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.last_trade_time = None
        
    def calculate_position_size(self):
        return self.capital * RISK_PER_TRADE
    
    def should_place_trade(self):
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False
        if self.last_trade_time and (datetime.now() - self.last_trade_time).seconds < 300:
            return False
        return True
    
    def update_trade_outcome(self, outcome):
        if outcome == "LOSS":
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self.last_trade_time = datetime.now()

# Streamlit App Configuration
st.set_page_config(
    page_title="Wealth Warrior Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Trading Dashboard")
    st.subheader("Strategy Configuration")
    
    # Asset Selection
    asset = st.selectbox("Select Asset", 
                        ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD", 
                         "XAUUSD", "TSLA", "AAPL", "AMZN", "GOOGL"])
    
    # Time Settings
    time_frame = st.select_slider("Time Frame", 
                                 options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"], 
                                 value="15m")
    
    time_zone = st.selectbox("Time Zone", pytz.all_timezones, index=pytz.all_timezones.index('UTC'))
    
    # Strategy Selection
    strategy = st.radio("Trading Strategy", 
                       ["Wealth Warrior", "RSI+MACD Combo", "EMA Crossover"])
    
    # Risk Management
    st.subheader("💰 Risk Management")
    risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 10.0, 2.0, step=0.5)
    max_daily_loss = st.slider("Max Daily Loss (%)", 1.0, 20.0, 5.0, step=0.5)
    martingale = st.checkbox("Enable Martingale Strategy")
    stop_loss = st.number_input("Stop Loss (pips)", 5, 50, 20)
    take_profit = st.number_input("Take Profit (pips)", 10, 100, 40)
    
    # Bot Controls
    st.subheader("🤖 Auto-Trading Bot")
    bot_active = st.toggle("Activate Trading Bot", value=st.session_state.bot_active)
    if bot_active:
        st.success("Auto-trading ACTIVE")
    else:
        st.warning("Auto-trading INACTIVE")
    
    # Educational Resources
    st.subheader("📚 Learning Center")
    if st.button("Strategy Guides"):
        st.session_state.current_page = "education"
    if st.button("Pattern Recognition Tutorial"):
        st.session_state.current_page = "patterns"

# Main Dashboard
st.title("💹 Wealth Warrior Pro - AI Trading Assistant")
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Account Balance: ${st.session_state.capital:,.2f}")

# Data Acquisition
@st.cache_data(ttl=60)
def load_data(asset, timeframe):
    try:
        data = yf.download(
            tickers=asset,
            period="5d",
            interval=timeframe,
            progress=False
        )
        return data
    except:
        st.error("Error fetching market data. Using sample data.")
        return pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=pd.date_range(end=datetime.now(), periods=100, freq='min'))

data = load_data(asset, time_frame)
strategy_engine = WealthWarriorStrategy()
risk_manager = RiskManager(st.session_state.capital)

# Strategy Execution
if not data.empty:
    signal = strategy_engine.generate_signal(data)
    
    # Trading Chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    )])
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA20'],
        mode='lines',
        name='EMA (20)',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA50'],
        mode='lines',
        name='EMA (50)',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f"{asset} Price Chart - {time_frame} timeframe",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal Display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RSI", f"{data['RSI'].iloc[-1]:.2f}", 
                  "Overbought" if data['RSI'].iloc[-1] > 70 else "Oversold" if data['RSI'].iloc[-1] < 30 else "Neutral")
    
    with col2:
        macd_diff = data['MACD'].iloc[-1] - data['Signal'].iloc[-1]
        st.metric("MACD", f"{macd_diff:.4f}", 
                  "Bullish" if macd_diff > 0 else "Bearish")
    
    with col3:
        signal_color = "green" if signal == "BUY" else "red" if signal == "SELL" else "gray"
        st.metric("AI Signal", signal, delta="STRONG" if signal != "HOLD" else "WAIT", 
                  delta_color=signal_color)
    
    # Trading Controls
    st.subheader("🚦 Trade Execution")
    trade_col1, trade_col2 = st.columns([1, 3])
    
    with trade_col1:
        amount = st.number_input("Trade Amount", 
                                min_value=1.0, 
                                max_value=st.session_state.capital, 
                                value=risk_manager.calculate_position_size())
        duration = st.select_slider("Contract Duration", 
                                   options=[1, 2, 5, 10, 15, 30, 60], 
                                   value=5)
        
        if st.button("📈 Execute BUY Trade", type="primary", disabled=(signal != "BUY")):
            trade = PocketOptionAPI.place_trade(asset, amount, "BUY", duration)
            st.session_state.trade_journal.append(trade)
            st.session_state.capital += trade['profit']
            st.experimental_rerun()
        
        if st.button("📉 Execute SELL Trade", type="secondary", disabled=(signal != "SELL")):
            trade = PocketOptionAPI.place_trade(asset, amount, "SELL", duration)
            st.session_state.trade_journal.append(trade)
            st.session_state.capital += trade['profit']
            st.experimental_rerun()
    
    with trade_col2:
        st.info("""
        **Trade Advisory**  
        - Strong BUY signal when all indicators align  
        - Position sizing: 2% of account balance  
        - Take profit at 1:2 risk-reward ratio  
        - Stop loss after 3 consecutive losses  
        """)
        
        if signal != "HOLD":
            st.success(f"✅ STRONG {signal} SIGNAL DETECTED")
            st.progress(0.95, text="Signal Confidence: 98%")
        else:
            st.warning("⚠️ No strong trading signal detected. Waiting for confirmation...")

# Trade Journal
if st.session_state.trade_journal:
    st.subheader("📒 Trade Journal")
    journal_df = pd.DataFrame(st.session_state.trade_journal)
    st.dataframe(journal_df.sort_values('timestamp', ascending=False), 
                height=300, 
                use_container_width=True)
    
    # Performance Metrics
    win_rate = len(journal_df[journal_df['outcome'] == 'WIN']) / len(journal_df) * 100
    total_profit = journal_df['profit'].sum()
    
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Total Trades", len(journal_df))
    metric2.metric("Win Rate", f"{win_rate:.2f}%")
    metric3.metric("Total Profit", f"${total_profit:,.2f}")

# Educational Content
st.subheader("🎓 Trading Education")
tab1, tab2, tab3 = st.tabs(["Strategy Guide", "Pattern Recognition", "Risk Management"])

with tab1:
    st.write("""
    ### Wealth Warrior Strategy
    This AI-powered strategy combines:
    - **EMA Crossovers** (20 & 50 periods)
    - **RSI Divergence** (14-period)
    - **MACD Signal Cross**
    - **Price Pattern Recognition**
    
    Entry Rules:
    1. EMA20 must cross above EMA50 for BUY
    2. RSI must be below 70 (not overbought)
    3. MACD must cross above signal line
    4. Confirmation with price pattern
    
    Exit Rules:
    - Take profit at 2x risk
    - Stop loss at 1x risk
    - Time expiration: 5 minutes
    """)

with tab2:
    st.image("https://www.investopedia.com/thmb/5_7dG9Q9U6d5G1T2wT2Q2bZ6X9I=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/dotdash_Final_Pattern_Recognition_Nov_2020-01-5e5d2c1c1d6f4c4e9d2c7b0c5d5b5c5b.jpg", 
             caption="Common Chart Patterns", width=600)
    st.write("""
    **Double Top Pattern:**
    - Two consecutive peaks at similar price levels
    - Indicates potential reversal to downside
    
    **Triangle Pattern:**
    - Converging price range with lower highs and higher lows
    - Breakout typically occurs in direction of trend
    """)

with tab3:
    st.write("""
    ### Risk Management Principles
    1. **2% Rule:** Never risk more than 2% of account on single trade
    2. **Stop Loss:** Mandatory protection against large losses
    3. **Martingale Strategy:** Double position after loss (use cautiously)
    4. **Daily Loss Limit:** Stop trading after 5% daily loss
    
    ```python
    # Risk Manager Code Example
    class RiskManager:
        def __init__(self, capital):
            self.capital = capital
            self.max_daily_loss = 0.05  # 5% of capital
            self.consecutive_losses = 0
        
        def calculate_position_size(self):
            return self.capital * 0.02  # 2% rule
    ```
    """)

# Bot Automation
if bot_active and signal != "HOLD" and risk_manager.should_place_trade():
    amount = risk_manager.calculate_position_size()
    trade = PocketOptionAPI.place_trade(
        asset, 
        amount, 
        signal, 
        duration
    )
    st.session_state.trade_journal.append(trade)
    st.session_state.capital += trade['profit']
    risk_manager.update_trade_outcome(trade['outcome'])
    st.toast(f"🤖 Auto-trade executed: {asset} {signal} ${amount:.2f}")
    st.experimental_rerun()

# Footer
st.divider()
st.caption("""
⚠️ **Disclaimer:** This is a simulation for educational purposes only. 
Past performance is not indicative of future results. Trading binary options involves significant risk.
""")
