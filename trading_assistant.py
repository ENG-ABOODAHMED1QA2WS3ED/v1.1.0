"""
Binary Options Trading Assistant

Requires Python 3.9-3.12 and Windows 11
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import json
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import requests
from dataclasses import dataclass
import asyncio
import websocket
import ssl

# Import PocketOptionAPI from local file
from pocketoption_api import PocketOptionAPI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_assistant.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading Signal"""
    pair: str
    direction: str  # UP or DOWN
    confidence: float
    quality: str  # Premium, Standard, Basic
    price: float
    timeframe: int  # in minutes
    timestamp: datetime
    indicators_agreement: int  # Number of agreeing indicators
    note: str = "Execute this trade on PocketOption"

@dataclass
class AccountInfo:
    """Account Information"""
    uid: str
    demo_balance: float
    live_balance: float
    last_updated: datetime

class TechnicalAnalyzer:
    """Advanced Technical Analyzer"""
    
    def __init__(self):
        self.timeframes = ['15s', '1m', '3m', '5m', '15m', '1h', '4h']
        self.indicators = [
            'accelerator_oscillator', 'adx', 'alligator', 'aroon', 'atr',
            'awesome_oscillator', 'bears_power', 'bollinger_bands', 'bb_width',
            'bulls_power', 'cci', 'donchian_channels', 'demarker', 'envelopes',
            'fractal', 'fractal_chaos_bands', 'ichimoku', 'keltner_channel',
            'macd', 'momentum', 'moving_average', 'osma', 'parabolic_sar',
            'rsi', 'rate_of_change', 'schaff_trend_cycle', 'stochastic_oscillator',
            'supertrend', 'vortex', 'williams_r', 'zigzag'
        ]
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average (EMA)"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        if len(prices) > 0:
            ema[0] = prices[0]
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema

    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average (SMA)"""
        sma = np.convolve(prices, np.ones(period)/period, mode='valid')
        return np.concatenate((np.full(period-1, np.nan), sma)) # Pad with NaN for alignment

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index (RSI)"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate Moving Average Convergence Divergence (MACD)"""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        prices_array = np.array(prices)
        ema12 = self._calculate_ema(prices_array, 12)
        ema26 = self._calculate_ema(prices_array, 26)
        
        macd_line = ema12[-1] - ema26[-1]
        signal_line = self._calculate_ema(np.array([macd_line]), 9)[-1]
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return 0.0, 0.0, 0.0
        
        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
    
    def calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float], k_period: int = 14) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator"""
        if len(closes) < k_period:
            return 50.0, 50.0
        
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Simplified %D calculation
        d_percent = k_percent
        
        return k_percent, d_percent

    def calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)"""
        if len(highs) < period * 2:
            return 0.0

        df = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': closes
        })

        # Calculate True Range (TR)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate Directional Movement (DM)
        df['plus_dm'] = df['high'].diff().apply(lambda x: x if x > 0 else 0)
        df['minus_dm'] = -df['low'].diff().apply(lambda x: x if x < 0 else 0)

        # Adjust DM where -DM > +DM or +DM > -DM
        df.loc[(df['plus_dm'] > df['minus_dm']), 'minus_dm'] = 0
        df.loc[(df['minus_dm'] > df['plus_dm']), 'plus_dm'] = 0

        # Calculate Smoothed True Range (ATR) and Smoothed Directional Movement
        atr = df['tr'].ewm(span=period, adjust=False).mean()
        plus_di = (df['plus_dm'].ewm(span=period, adjust=False).mean() / atr) * 100
        minus_di = (df['minus_dm'].ewm(span=period, adjust=False).mean() / atr) * 100

        # Calculate DX
        df['dx'] = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        df['dx'] = df['dx'].fillna(0) # Handle division by zero

        # Calculate ADX
        adx = df['dx'].ewm(span=period, adjust=False).mean()
        return adx.iloc[-1] if not adx.empty else 0.0

    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Average True Range (ATR)"""
        if len(highs) < period:
            return 0.0

        df = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': closes
        })

        df['tr'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
        atr = df['tr'].rolling(window=period).mean()
        return atr.iloc[-1] if not atr.empty else 0.0

    def calculate_cci(self, highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> float:
        """Calculate Commodity Channel Index (CCI)"""
        if len(closes) < period:
            return 0.0

        df = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': closes
        })

        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = df['tp'].rolling(window=period).mean()
        mad_tp = df['tp'].rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)

        cci = (df['tp'] - sma_tp) / (0.015 * mad_tp)
        return cci.iloc[-1] if not cci.empty else 0.0

    def calculate_momentum(self, prices: List[float], period: int = 14) -> float:
        """Calculate Momentum Indicator"""
        if len(prices) < period:
            return 0.0
        return prices[-1] - prices[-period]

    def calculate_rate_of_change(self, prices: List[float], period: int = 14) -> float:
        """Calculate Rate of Change (ROC)"""
        if len(prices) < period:
            return 0.0
        return ((prices[-1] - prices[-period]) / prices[-period]) * 100

    def calculate_williams_r(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Williams %R"""
        if len(closes) < period:
            return 0.0

        highest_high = pd.Series(highs).rolling(window=period).max()
        lowest_low = pd.Series(lows).rolling(window=period).min()

        williams_r = ((highest_high - closes) / (highest_high - lowest_low)) * -100
        return williams_r.iloc[-1] if not williams_r.empty else 0.0

    def calculate_accelerator_oscillator(self, prices: List[float]) -> float:
        """Calculate Accelerator Oscillator (AO)"""
        if len(prices) < 34:
            return 0.0
        ao = self._calculate_sma(np.array(prices), 5) - self._calculate_sma(np.array(prices), 34)
        return ao[-1] if len(ao) > 0 else 0.0

    def calculate_alligator(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate Alligator Indicator (Jaw, Teeth, Lips)"""
        if len(prices) < 13:
            return 0.0, 0.0, 0.0
        jaw = self._calculate_sma(np.array(prices), 13)[-1] # Smoothed by 8 bars, shifted 5 bars
        teeth = self._calculate_sma(np.array(prices), 8)[-1] # Smoothed by 5 bars, shifted 3 bars
        lips = self._calculate_sma(np.array(prices), 5)[-1] # Smoothed by 3 bars, shifted 2 bars
        return jaw, teeth, lips

    def calculate_aroon(self, highs: List[float], lows: List[float], period: int = 25) -> Tuple[float, float]:
        """Calculate Aroon Indicator (Aroon Up, Aroon Down)"""
        if len(highs) < period:
            return 0.0, 0.0
        
        aroon_up = ((period - np.argmax(highs[-period:])) / period) * 100
        aroon_down = ((period - np.argmin(lows[-period:])) / period) * 100
        return aroon_up, aroon_down

    def calculate_awesome_oscillator(self, prices: List[float]) -> float:
        """Calculate Awesome Oscillator (AO)"""
        if len(prices) < 34:
            return 0.0
        median_prices = (np.array(prices) + np.array(prices)) / 2 # Assuming prices are close/median
        ao = self._calculate_sma(median_prices, 5) - self._calculate_sma(median_prices, 34)
        return ao[-1] if len(ao) > 0 else 0.0

    def calculate_bears_power(self, lows: List[float], closes: List[float], period: int = 13) -> float:
        """Calculate Bears Power"""
        if len(closes) < period:
            return 0.0
        ema = self._calculate_ema(np.array(closes), period)
        bears_power = lows[-1] - ema[-1]
        return bears_power

    def calculate_bulls_power(self, highs: List[float], closes: List[float], period: int = 13) -> float:
        """Calculate Bulls Power"""
        if len(closes) < period:
            return 0.0
        ema = self._calculate_ema(np.array(closes), period)
        bulls_power = highs[-1] - ema[-1]
        return bulls_power

    def calculate_bb_width(self, prices: List[float], period: int = 20, std_dev: int = 2) -> float:
        """Calculate Bollinger Band Width"""
        if len(prices) < period:
            return 0.0
        upper_bb, _, lower_bb = self.calculate_bollinger_bands(prices, period, std_dev)
        return upper_bb - lower_bb

    def calculate_donchian_channels(self, highs: List[float], lows: List[float], period: int = 20) -> Tuple[float, float, float]:
        """Calculate Donchian Channels (Upper, Middle, Lower)"""
        if len(highs) < period:
            return 0.0, 0.0, 0.0
        upper_channel = max(highs[-period:])
        lower_channel = min(lows[-period:])
        middle_channel = (upper_channel + lower_channel) / 2
        return upper_channel, middle_channel, lower_channel

    def calculate_demarker(self, highs: List[float], lows: List[float], period: int = 14) -> float:
        """Calculate DeMarker Indicator"""
        if len(highs) < period:
            return 0.0
        
        dm_up = []
        dm_down = []
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                dm_up.append(highs[i] - highs[i-1])
            else:
                dm_up.append(0)
            
            if lows[i] < lows[i-1]:
                dm_down.append(lows[i-1] - lows[i])
            else:
                dm_down.append(0)
        
        sum_dm_up = np.sum(dm_up[-period:])
        sum_dm_down = np.sum(dm_down[-period:])
        
        if (sum_dm_up + sum_dm_down) == 0:
            return 0.5 # Neutral
        
        demarker = sum_dm_up / (sum_dm_up + sum_dm_down)
        return demarker

    def calculate_envelopes(self, prices: List[float], period: int = 14, deviation: float = 0.1) -> Tuple[float, float]:
        """Calculate Envelopes (Upper, Lower)"""
        if len(prices) < period:
            return 0.0, 0.0
        sma = self._calculate_sma(np.array(prices), period)
        upper_envelope = sma[-1] * (1 + deviation)
        lower_envelope = sma[-1] * (1 - deviation)
        return upper_envelope, lower_envelope

    def calculate_ichimoku(self, highs: List[float], lows: List[float], closes: List[float]) -> Dict[str, float]:
        """Calculate Ichimoku Cloud components"""
        if len(highs) < 52:
            return {
                'tenkan_sen': 0.0,
                'kijun_sen': 0.0,
                'senkou_span_a': 0.0,
                'senkou_span_b': 0.0,
                'chikou_span': 0.0
            }

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        tenkan_sen = (max(highs[-9:]) + min(lows[-9:])) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        kijun_sen = (max(highs[-26:]) + min(lows[-26:])) / 2

        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 plotted 26 periods ahead
        senkou_span_a = (tenkan_sen + kijun_sen) / 2

        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
        senkou_span_b = (max(highs[-52:]) + min(lows[-52:])) / 2

        # Chikou Span (Lagging Span): Current closing price plotted 26 periods behind
        chikou_span = closes[-1]

        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }

    def calculate_keltner_channel(self, highs: List[float], lows: List[float], closes: List[float], period: int = 20, multiplier: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Keltner Channel (Upper, Middle, Lower)"""
        if len(closes) < period:
            return 0.0, 0.0, 0.0
        
        middle_line = self._calculate_ema(np.array(closes), period)[-1]
        atr = self.calculate_atr(highs, lows, closes, period)
        
        upper_band = middle_line + (multiplier * atr)
        lower_band = middle_line - (multiplier * atr)
        
        return upper_band, middle_line, lower_band

    def calculate_osma(self, prices: List[float]) -> float:
        """Calculate Oscillator of Moving Averages (OsMA)"""
        if len(prices) < 35: # MACD needs 26, Signal needs 9, so 26+9=35
            return 0.0
        macd_line, signal_line, _ = self.calculate_macd(prices)
        osma = macd_line - signal_line
        return osma

    def calculate_parabolic_sar(self, highs: List[float], lows: List[float], acceleration: float = 0.02, max_acceleration: float = 0.2) -> float:
        """Calculate Parabolic SAR"""
        # This is a simplified implementation and needs historical SAR values for proper calculation
        # For a full implementation, consider a dedicated TA library or more complex logic.
        if len(highs) < 2:
            return 0.0
        
        # Placeholder: In a real scenario, SAR would be calculated iteratively
        # For now, return a value based on recent price movement
        if highs[-1] > highs[-2]: # Uptrend
            return min(lows[-1], lows[-2]) # SAR below price
        else: # Downtrend
            return max(highs[-1], highs[-2]) # SAR above price

    def calculate_schaff_trend_cycle(self, prices: List[float], macd_fast: int = 23, macd_slow: int = 50, cycle: int = 10) -> float:
        """Calculate Schaff Trend Cycle (STC)"""
        if len(prices) < macd_slow + cycle:
            return 0.0
        
        # Step 1: Calculate MACD
        macd_line, _, _ = self.calculate_macd(prices)
        
        # Step 2: Calculate 2-period EMA of MACD
        ema_macd = self._calculate_ema(np.array([macd_line]), 2)
        
        # Step 3: Calculate Stochastic of EMA_MACD
        lowest_ema_macd = np.min(ema_macd[-cycle:])
        highest_ema_macd = np.max(ema_macd[-cycle:])
        
        if (highest_ema_macd - lowest_ema_macd) == 0:
            stoch_k = 0.0
        else:
            stoch_k = ((ema_macd[-1] - lowest_ema_macd) / (highest_ema_macd - lowest_ema_macd)) * 100
        
        # Step 4: Calculate 2-period EMA of Stoch_K
        stc = self._calculate_ema(np.array([stoch_k]), 2)
        
        return stc[-1] if len(stc) > 0 else 0.0

    def calculate_supertrend(self, highs: List[float], lows: List[float], closes: List[float], period: int = 10, multiplier: float = 3.0) -> Tuple[float, str]:
        """Calculate SuperTrend Indicator"""
        if len(highs) < period:
            return 0.0, "None"
        
        df = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': closes
        })

        # Calculate ATR
        df['tr'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
        df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()

        # Calculate Basic Upper and Lower Bands
        df['basic_upper_band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
        df['basic_lower_band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])

        # Calculate Final Upper and Lower Bands
        df['final_upper_band'] = df['basic_upper_band']
        df['final_lower_band'] = df['basic_lower_band']

        for i in range(1, len(df)):
            if df['close'].iloc[i-1] > df['final_upper_band'].iloc[i-1]:
                df['final_upper_band'].iloc[i] = max(df['basic_upper_band'].iloc[i], df['final_upper_band'].iloc[i-1])
            else:
                df['final_upper_band'].iloc[i] = df['basic_upper_band'].iloc[i]

            if df['close'].iloc[i-1] < df['final_lower_band'].iloc[i-1]:
                df['final_lower_band'].iloc[i] = min(df['basic_lower_band'].iloc[i], df['final_lower_band'].iloc[i-1])
            else:
                df['final_lower_band'].iloc[i] = df['basic_lower_band'].iloc[i]

        # Calculate SuperTrend
        df['supertrend'] = np.nan
        df['direction'] = np.nan

        for i in range(len(df)):
            if i == 0:
                df['direction'].iloc[i] = "up" if df['close'].iloc[i] > df['final_upper_band'].iloc[i] else "down"
                df['supertrend'].iloc[i] = df['final_lower_band'].iloc[i] if df['direction'].iloc[i] == "up" else df['final_upper_band'].iloc[i]
            else:
                if df['supertrend'].iloc[i-1] == df['final_lower_band'].iloc[i-1] and df['close'].iloc[i] <= df['final_lower_band'].iloc[i]:
                    df['direction'].iloc[i] = "down"
                elif df['supertrend'].iloc[i-1] == df['final_upper_band'].iloc[i-1] and df['close'].iloc[i] >= df['final_upper_band'].iloc[i]:
                    df['direction'].iloc[i] = "up"
                elif df['supertrend'].iloc[i-1] == df['final_lower_band'].iloc[i-1] and df['close'].iloc[i] >= df['final_lower_band'].iloc[i]:
                    df['direction'].iloc[i] = "up"
                elif df['supertrend'].iloc[i-1] == df['final_upper_band'].iloc[i-1] and df['close'].iloc[i] <= df['final_upper_band'].iloc[i]:
                    df['direction'].iloc[i] = "down"
                else:
                    df['direction'].iloc[i] = df['direction'].iloc[i-1]

                if df['direction'].iloc[i] == "up":
                    df['supertrend'].iloc[i] = df['final_lower_band'].iloc[i]
                else:
                    df['supertrend'].iloc[i] = df['final_upper_band'].iloc[i]
        
        return df['supertrend'].iloc[-1], df['direction'].iloc[-1]

    def calculate_vortex(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[float, float]:
        """Calculate Vortex Indicator (VI+, VI-)"""
        if len(highs) < period:
            return 0.0, 0.0
        
        df = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': closes
        })

        df['tr'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
        df['vm_plus'] = abs(df['high'] - df['low'].shift(1))
        df['vm_minus'] = abs(df['low'] - df['high'].shift(1))

        sum_tr = df['tr'].rolling(window=period).sum()
        sum_vm_plus = df['vm_plus'].rolling(window=period).sum()
        sum_vm_minus = df['vm_minus'].rolling(window=period).sum()

        vi_plus = sum_vm_plus / sum_tr
        vi_minus = sum_vm_minus / sum_tr
        
        return vi_plus.iloc[-1] if not vi_plus.empty else 0.0, vi_minus.iloc[-1] if not vi_minus.empty else 0.0

    def calculate_fractal(self, highs: List[float], lows: List[float]) -> Tuple[List[float], List[float]]:
        """Calculate Fractals (Up and Down)"""
        up_fractals = []
        down_fractals = []
        # A fractal requires at least 5 bars: two bars to the left and two to the right of the central bar.
        if len(highs) < 5:
            return up_fractals, down_fractals

        for i in range(2, len(highs) - 2):
            # Up Fractal: A high with two lower highs before and two lower highs after
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                up_fractals.append(highs[i])
            # Down Fractal: A low with two higher lows before and two higher lows after
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                down_fractals.append(lows[i])
        return up_fractals, down_fractals

    def calculate_fractal_chaos_bands(self, highs: List[float], lows: List[float], period: int = 10) -> Tuple[float, float]:
        """Calculate Fractal Chaos Bands (Upper and Lower)"""
        if len(highs) < period:
            return 0.0, 0.0
        
        upper_band = max(highs[-period:])
        lower_band = min(lows[-period:])
        return upper_band, lower_band

    def calculate_zigzag(self, prices: List[float], percentage: float = 5.0) -> List[float]:
        """Calculate ZigZag indicator (simplified - returns pivot points)"""
        if len(prices) < 2:
            return []

        zigzag_points = [prices[0]]
        last_pivot_idx = 0
        trend = None # 'up' or 'down'

        for i in range(1, len(prices)):
            current_price = prices[i]
            last_pivot_price = zigzag_points[-1]

            if trend is None:
                # Determine initial trend
                if current_price > last_pivot_price * (1 + percentage / 100):
                    trend = 'up'
                elif current_price < last_pivot_price * (1 - percentage / 100):
                    trend = 'down'
            elif trend == 'up':
                if current_price < last_pivot_price * (1 - percentage / 100):
                    # New down pivot
                    zigzag_points.append(current_price)
                    last_pivot_idx = i
                    trend = 'down'
                elif current_price > last_pivot_price: # Update existing pivot if higher
                    zigzag_points[-1] = current_price
            elif trend == 'down':
                if current_price > last_pivot_price * (1 + percentage / 100):
                    # New up pivot
                    zigzag_points.append(current_price)
                    last_pivot_idx = i
                    trend = 'up'
                elif current_price < last_pivot_price: # Update existing pivot if lower
                    zigzag_points[-1] = current_price
        return zigzag_points

    def analyze_candlestick_patterns(self, ohlc_data: List[Dict]) -> List[str]:
        """Analyze candlestick patterns"""
        if len(ohlc_data) < 3:
            return []
        
        current = ohlc_data[-1]
        previous = ohlc_data[-2]
        
        patterns = []
        
        # Doji pattern
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        
        if total_range > 0 and body_size / total_range < 0.1:
            patterns.append('Doji')
        
        # Engulfing pattern
        if (previous['close'] > previous['open'] and  # Previous bullish candle
            current['open'] > current['close'] and   # Current bearish candle
            current['open'] > previous['close'] and  # Open higher than previous close
            current['close'] < previous['open']):    # Close lower than previous open
            patterns.append('Bearish_Engulfing')
        
        elif (previous['open'] > previous['close'] and  # Previous bearish candle
              current['close'] > current['open'] and   # Current bullish candle
              current['open'] < previous['close'] and  # Open lower than previous close
              current['close'] > previous['open']):    # Close higher than previous open
            patterns.append('Bullish_Engulfing')
        
        # Hammer pattern
        lower_shadow = current['open'] - current['low'] if current['open'] < current['close'] else current['close'] - current['low']
        upper_shadow = current['high'] - current['close'] if current['close'] > current['open'] else current['high'] - current['open']
        
        if lower_shadow > 2 * body_size and upper_shadow < body_size:
            patterns.append('Hammer')
        
        return patterns
    
    def generate_signal(self, pair: str, ohlc_data: List[Dict]) -> Optional[TradingSignal]:
        """Generate a trading signal based on technical analysis"""
        if len(ohlc_data) < 52: # Minimum for Ichimoku
            return None
        
        closes = [candle['close'] for candle in ohlc_data]
        highs = [candle['high'] for candle in ohlc_data]
        lows = [candle['low'] for candle in ohlc_data]
        
        # Calculate indicators
        rsi = self.calculate_rsi(closes)
        macd_line, signal_line, histogram = self.calculate_macd(closes)
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(closes)
        k_percent, d_percent = self.calculate_stochastic(highs, lows, closes)
        adx = self.calculate_adx(highs, lows, closes)
        atr = self.calculate_atr(highs, lows, closes)
        cci = self.calculate_cci(highs, lows, closes)
        momentum = self.calculate_momentum(closes)
        roc = self.calculate_rate_of_change(closes)
        williams_r = self.calculate_williams_r(highs, lows, closes)
        ao_accel = self.calculate_accelerator_oscillator(closes)
        jaw, teeth, lips = self.calculate_alligator(closes)
        aroon_up, aroon_down = self.calculate_aroon(highs, lows)
        ao_awesome = self.calculate_awesome_oscillator(closes)
        bears_power = self.calculate_bears_power(lows, closes)
        bulls_power = self.calculate_bulls_power(highs, closes)
        bb_width = self.calculate_bb_width(closes)
        upper_dc, middle_dc, lower_dc = self.calculate_donchian_channels(highs, lows)
        demarker = self.calculate_demarker(highs, lows)
        upper_env, lower_env = self.calculate_envelopes(closes)
        ichimoku = self.calculate_ichimoku(highs, lows, closes)
        upper_kc, middle_kc, lower_kc = self.calculate_keltner_channel(highs, lows, closes)
        osma = self.calculate_osma(closes)
        parabolic_sar = self.calculate_parabolic_sar(highs, lows) # Simplified
        stc = self.calculate_schaff_trend_cycle(closes)
        supertrend_val, supertrend_dir = self.calculate_supertrend(highs, lows, closes)
        vi_plus, vi_minus = self.calculate_vortex(highs, lows, closes)
        up_fractals, down_fractals = self.calculate_fractal(highs, lows)
        upper_fcb, lower_fcb = self.calculate_fractal_chaos_bands(highs, lows)
        zigzag_points = self.calculate_zigzag(closes)

        # Analyze candlestick patterns
        patterns = self.analyze_candlestick_patterns(ohlc_data)
        
        # Determine direction based on indicators
        bullish_signals = 0
        bearish_signals = 0
        total_indicators = 0
        
        # RSI
        total_indicators += 1
        if rsi < 30:
            bullish_signals += 1
        elif rsi > 70:
            bearish_signals += 1
        
        # MACD
        total_indicators += 1
        if macd_line > signal_line and histogram > 0:
            bullish_signals += 1
        elif macd_line < signal_line and histogram < 0:
            bearish_signals += 1
        
        # Bollinger Bands
        current_price = closes[-1]
        total_indicators += 1
        if current_price < lower_bb:
            bullish_signals += 1
        elif current_price > upper_bb:
            bearish_signals += 1
        
        # Stochastic
        total_indicators += 1
        if k_percent < 20 and d_percent < 20:
            bullish_signals += 1
        elif k_percent > 80 and d_percent > 80:
            bearish_signals += 1

        # ADX (trend strength, not direction)
        # CCI
        total_indicators += 1
        if cci < -100:
            bullish_signals += 1
        elif cci > 100:
            bearish_signals += 1

        # Momentum
        total_indicators += 1
        if momentum > 0:
            bullish_signals += 1
        elif momentum < 0:
            bearish_signals += 1

        # ROC
        total_indicators += 1
        if roc > 0:
            bullish_signals += 1
        elif roc < 0:
            bearish_signals += 1

        # Williams %R
        total_indicators += 1
        if williams_r < -80:
            bullish_signals += 1
        elif williams_r > -20:
            bearish_signals += 1

        # Accelerator Oscillator
        total_indicators += 1
        if ao_accel > 0:
            bullish_signals += 1
        elif ao_accel < 0:
            bearish_signals += 1

        # Alligator
        total_indicators += 1
        if lips > teeth and teeth > jaw: # Mouth opening up
            bullish_signals += 1
        elif lips < teeth and teeth < jaw: # Mouth opening down
            bearish_signals += 1

        # Aroon
        total_indicators += 1
        if aroon_up > 70 and aroon_down < 30:
            bullish_signals += 1
        elif aroon_down > 70 and aroon_up < 30:
            bearish_signals += 1

        # Awesome Oscillator
        total_indicators += 1
        if ao_awesome > 0:
            bullish_signals += 1
        elif ao_awesome < 0:
            bearish_signals += 1

        # Bears Power
        total_indicators += 1
        if bears_power < 0: # Negative bears power indicates bullish
            bullish_signals += 1
        elif bears_power > 0: # Positive bears power indicates bearish
            bearish_signals += 1

        # Bulls Power
        total_indicators += 1
        if bulls_power > 0: # Positive bulls power indicates bullish
            bullish_signals += 1
        elif bulls_power < 0: # Negative bulls power indicates bearish
            bearish_signals += 1

        # Donchian Channels (price near upper/lower band)
        total_indicators += 1
        if current_price > upper_dc * 0.99: # Price near upper band
            bullish_signals += 1
        elif current_price < lower_dc * 1.01: # Price near lower band
            bearish_signals += 1

        # DeMarker
        total_indicators += 1
        if demarker > 0.7:
            bullish_signals += 1
        elif demarker < 0.3:
            bearish_signals += 1

        # Envelopes
        total_indicators += 1
        if current_price > upper_env:
            bearish_signals += 1 # Price above upper envelope, overbought
        elif current_price < lower_env:
            bullish_signals += 1 # Price below lower envelope, oversold

        # Ichimoku (simplified interpretation)
        total_indicators += 1
        if ichimoku['tenkan_sen'] > ichimoku['kijun_sen'] and current_price > ichimoku['senkou_span_a'] and current_price > ichimoku['senkou_span_b']:
            bullish_signals += 1
        elif ichimoku['tenkan_sen'] < ichimoku['kijun_sen'] and current_price < ichimoku['senkou_span_a'] and current_price < ichimoku['senkou_span_b']:
            bearish_signals += 1

        # Keltner Channel
        total_indicators += 1
        if current_price > upper_kc:
            bearish_signals += 1 # Price above upper Keltner, overbought
        elif current_price < lower_kc:
            bullish_signals += 1 # Price below lower Keltner, oversold

        # OsMA
        total_indicators += 1
        if osma > 0:
            bullish_signals += 1
        elif osma < 0:
            bearish_signals += 1

        # Parabolic SAR (simplified)
        total_indicators += 1
        if current_price > parabolic_sar: # Price above SAR
            bullish_signals += 1
        elif current_price < parabolic_sar: # Price below SAR
            bearish_signals += 1

        # Schaff Trend Cycle
        total_indicators += 1
        if stc > 75:
            bearish_signals += 1 # Overbought
        elif stc < 25:
            bullish_signals += 1 # Oversold

        # SuperTrend
        total_indicators += 1
        if supertrend_dir == "up":
            bullish_signals += 1
        elif supertrend_dir == "down":
            bearish_signals += 1

        # Vortex
        total_indicators += 1
        if vi_plus > vi_minus:
            bullish_signals += 1
        elif vi_minus > vi_plus:
            bearish_signals += 1

        # Fractals (presence of recent fractal indicates potential reversal)
        total_indicators += 1
        if len(up_fractals) > 0 and up_fractals[-1] == highs[-3]: # Check if last fractal is recent
            bearish_signals += 1 # Up fractal often indicates potential reversal to down
        if len(down_fractals) > 0 and down_fractals[-1] == lows[-3]: # Check if last fractal is recent
            bullish_signals += 1 # Down fractal often indicates potential reversal to up

        # Fractal Chaos Bands (price breaking bands)
        total_indicators += 1
        if current_price > upper_fcb:
            bullish_signals += 1 # Price breaking upper band
        elif current_price < lower_fcb:
            bearish_signals += 1 # Price breaking lower band

        # Zigzag (simplified - looking for recent pivot)
        total_indicators += 1
        if len(zigzag_points) >= 2:
            if zigzag_points[-1] > zigzag_points[-2]: # Last move was up
                bullish_signals += 1
            else:
                bearish_signals += 1

        # Candlestick patterns
        for pattern in patterns:
            total_indicators += 1
            if pattern in ['Bullish_Engulfing', 'Hammer']:
                bullish_signals += 1
            elif pattern in ['Bearish_Engulfing']:
                bearish_signals += 1
        
        # Determine direction and confidence
        if bullish_signals > bearish_signals and bullish_signals >= 5: # At least 5 indicators agree
            direction = "UP"
            confidence = (bullish_signals / total_indicators) * 100
            indicators_agreement = bullish_signals
        elif bearish_signals > bullish_signals and bearish_signals >= 5: # At least 5 indicators agree
            direction = "DOWN"
            confidence = (bearish_signals / total_indicators) * 100
            indicators_agreement = bearish_signals
        else:
            return None  # Unclear signal or not enough agreement
        
        # Determine signal quality
        if confidence >= 90:
            quality = "Premium"
        elif confidence >= 80:
            quality = "Standard"
        elif confidence >= 70:
            quality = "Basic"
        else:
            return None  # Very low confidence
        
        # Determine optimal timeframe
        timeframe = self._determine_optimal_timeframe(confidence)
        
        return TradingSignal(
            pair=pair,
            direction=direction,
            confidence=round(confidence, 1),
            quality=quality,
            price=current_price,
            timeframe=timeframe,
            timestamp=datetime.now(),
            indicators_agreement=indicators_agreement
        )
    
    def _determine_optimal_timeframe(self, confidence: float) -> int:
        """Determine optimal timeframe based on confidence level"""
        if confidence >= 90:
            return 1  # 1 minute for very high confidence (fast execution)
        elif confidence >= 80:
            return 3  # 3 minutes for high confidence
        elif confidence >= 70:
            return 5  # 5 minutes for medium-high confidence
        else:
            return 10 # 10 minutes for medium confidence

class TradingAssistantGUI:
    """Graphical User Interface"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Binary Options Trading Assistant")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Core components
        self.connector = PocketOptionAPI()
        self.analyzer = TechnicalAnalyzer()
        
        # GUI variables
        self.selected_market = tk.StringVar(value="OTC")
        self.selected_pair = tk.StringVar()
        self.analysis_running = False
        
        self.setup_ui()
        self.setup_styles()
        
    def setup_styles(self):
        """Setup GUI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom colors
        style.configure('Title.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffffff', 
                       font=('Arial', 16, 'bold'))
        
        style.configure('Info.TLabel', 
                       background='#2b2b2b', 
                       foreground='#cccccc', 
                       font=('Arial', 10))
        
        style.configure('Success.TLabel', 
                       background='#2b2b2b', 
                       foreground='#00ff00', 
                       font=('Arial', 12, 'bold'))
        
        style.configure('Warning.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffaa00', 
                       font=('Arial', 12, 'bold'))
        
        style.configure('Error.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ff0000', 
                       font=('Arial', 12, 'bold'))
        
        # New balance styles
        style.configure('Balance.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffffff', 
                       font=('Arial', 14, 'bold'))
        style.configure('DemoBalance.TLabel', 
                       background='#2b2b2b', 
                       foreground='#6495ED',  # CornflowerBlue
                       font=('Arial', 18, 'bold'))
        style.configure('LiveBalance.TLabel', 
                       background='#2b2b2b', 
                       foreground='#32CD32',  # LimeGreen
                       font=('Arial', 18, 'bold'))
        
        # Login button styles
        style.configure('Login.TFrame', background='#3c3c3c', relief='flat', borderwidth=0)
        style.configure('TButton', font=('Arial', 10, 'bold'), background='#555555', foreground='#ffffff')
        style.map('TButton', background=[('active', '#777777')])

    def setup_ui(self):
        """Setup user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="Binary Options Trading Assistant", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Login frame
        self.setup_login_frame(main_frame)
        
        # Account info frame (initially hidden)
        self.account_frame = ttk.Frame(main_frame, padding="10 10 10 10", style='Login.TFrame')
        
        self.demo_balance_label = ttk.Label(self.account_frame, text="Demo Balance: --", style='DemoBalance.TLabel')
        self.demo_balance_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.live_balance_label = ttk.Label(self.account_frame, text="Live Balance: --", style='LiveBalance.TLabel')
        self.live_balance_label.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Main content frame (initially hidden)
        self.main_content_frame = ttk.Frame(main_frame)
        
        # Trading frame
        self.setup_trading_frame(self.main_content_frame)
        
        # Log frame
        self.setup_log_frame(self.main_content_frame)
        
        # Hide main content until logged in
        self.account_frame.pack_forget()
        self.main_content_frame.pack_forget()
        
    def setup_login_frame(self, parent_frame):
        """Setup simplified login frame"""
        self.login_frame = ttk.Frame(parent_frame, padding="30 30 30 30", style='Login.TFrame')
        self.login_frame.pack(pady=50, padx=50, fill=tk.BOTH, expand=True)
        
        ttk.Label(self.login_frame, text="Login to PocketOption", style='Title.TLabel').pack(pady=20)
        
        # SSID field
        ssid_label = ttk.Label(self.login_frame, text="SSID:", style='Info.TLabel')
        ssid_label.pack(pady=5)
        self.ssid_entry = ttk.Entry(self.login_frame, width=50, font=('Arial', 12))
        self.ssid_entry.pack(pady=10)
        
        # Connect button
        connect_button = ttk.Button(self.login_frame, text="Connect", command=self.connect_to_pocketoption)
        connect_button.pack(pady=30, ipadx=20, ipady=10)
        
    def setup_account_frame(self, parent_frame):
        """Setup account info frame - simplified and merged into setup_ui"""
        pass # This function is no longer needed separately
            
    def setup_trading_frame(self, parent_frame):
        """Setup trading and analysis frame"""
        trading_frame = ttk.LabelFrame(parent_frame, text="Trading and Analysis", padding="10 10 10 10", style='Login.TFrame')
        trading_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Market type selection
        market_type_frame = ttk.Frame(trading_frame, style='Login.TFrame')
        market_type_frame.pack(pady=5)
        ttk.Radiobutton(market_type_frame, text="OTC", variable=self.selected_market, value="OTC", command=self.update_available_pairs, style='Info.TLabel').pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(market_type_frame, text="Real", variable=self.selected_market, value="Real", command=self.update_available_pairs, style='Info.TLabel').pack(side=tk.LEFT, padx=10)
        
        # Trading pair selection
        pair_label = ttk.Label(trading_frame, text="Trading Pair:", style='Info.TLabel')
        pair_label.pack(pady=5)
        self.pair_combobox = ttk.Combobox(trading_frame, textvariable=self.selected_pair, state="readonly", width=30, font=('Arial', 10))
        self.pair_combobox.pack(pady=5)
        
        # Start analysis button
        self.start_analysis_button = ttk.Button(trading_frame, text="Start Auto Analysis", command=self.perform_analysis)
        self.start_analysis_button.pack(pady=10)
        
        # Stop analysis button
        self.stop_analysis_button = ttk.Button(trading_frame, text="Stop Analysis", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_analysis_button.pack(pady=5)
        
        # Analysis interval (minutes)
        analysis_interval_label = ttk.Label(trading_frame, text="Analysis Interval (minutes):", style='Info.TLabel')
        analysis_interval_label.pack(pady=5)
        self.analysis_interval = tk.IntVar(value=1) # Default 1 minute
        ttk.Spinbox(trading_frame, from_=1, to=60, textvariable=self.analysis_interval, width=5, font=('Arial', 10)).pack(pady=5)
        
        # Signal display
        self.signal_display = scrolledtext.ScrolledText(trading_frame, width=80, height=10, bg='#3c3c3c', fg='#ffffff', font=('Arial', 10), relief=tk.FLAT, borderwidth=0)
        self.signal_display.pack(pady=10, fill=tk.BOTH, expand=True)
        
    def setup_log_frame(self, parent_frame):
        """Setup log frame"""
        log_frame = ttk.LabelFrame(parent_frame, text="Logs", padding="10 10 10 10", style='Login.TFrame')
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=100, height=15, bg='#3c3c3c', fg='#ffffff', font=('Arial', 9), relief=tk.FLAT, borderwidth=0)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def log_message(self, message: str, level: str = "info"):
        """Log messages to GUI and log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] [{level.upper()}] {message}\n"
        
        self.log_text.insert(tk.END, full_message)
        self.log_text.see(tk.END) # Auto-scroll to bottom
        
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "success":
            logger.info(message) # Can use INFO for success logs too
            
    def connect_to_pocketoption(self):
        ssid = self.ssid_entry.get()
        if not ssid:
            messagebox.showerror("Error", "Please enter SSID.")
            return
        
        self.log_message("Connecting to PocketOption...", "info")
        
        # Run connection in a separate thread to avoid freezing the GUI
        threading.Thread(target=self._perform_connection, args=(ssid,)).start()

    def _perform_connection(self, ssid):
        if self.connector.connect(ssid):
            self.log_message("Successfully connected!", "success")
            self.is_connected = True
            self.update_account_info()
            self.update_available_pairs()
            self.login_frame.pack_forget() # Hide login frame
            
            # Show account info frame and main content
            self.account_frame.pack(fill=tk.X, pady=10)
            self.main_content_frame.pack(fill=tk.BOTH, expand=True)
            
            # Start periodic balance update
            self.root.after(5000, self._periodic_balance_update) # Update every 5 seconds
            self.root.after(60000, self._periodic_connection_check) # Check connection every minute
        else:
            self.log_message("Connection failed. Please check SSID.", "error")
            self.is_connected = False
            
    def _periodic_connection_check(self):
        if self.is_connected:
            if not self.connector.check_connection_status():
                self.log_message("Connection to platform lost. SSID might be expired.", "error")
                self.disconnect_and_reprompt_ssid()
            else:
                self.root.after(60000, self._periodic_connection_check) # Schedule next check

    def disconnect_and_reprompt_ssid(self):
        self.connector.disconnect()
        self.is_connected = False
        self.analysis_running = False
        self.start_analysis_button.config(state=tk.NORMAL)
        self.stop_analysis_button.config(state=tk.DISABLED)
        
        self.account_frame.pack_forget()
        self.main_content_frame.pack_forget()
        self.login_frame.pack(pady=50, padx=50, fill=tk.BOTH, expand=True)
        self.ssid_entry.delete(0, tk.END) # Clear SSID field
        messagebox.showinfo("Connection Lost", "Please re-enter your SSID to connect again.")

    def update_account_info(self):
        if self.connector.is_connected:
            demo_balance, live_balance = self.connector.get_balance()
            self.demo_balance_label.config(text=f"Demo Balance: {demo_balance:.2f} $")
            self.live_balance_label.config(text=f"Live Balance: {live_balance:.2f} $")
            self.log_message(f"Account info updated: Demo: {demo_balance:.2f}$, Live: {live_balance:.2f}$", "info")
        else:
            self.demo_balance_label.config(text="Demo Balance: --")
            self.live_balance_label.config(text="Live Balance: --")
            
    def _periodic_balance_update(self):
        if self.is_connected:
            self.update_account_info()
            self.root.after(5000, self._periodic_balance_update) # Schedule next update
            
    def update_available_pairs(self):
        market_type = self.selected_market.get()
        pairs = self.connector.get_available_assets(market_type)
        self.pair_combobox["values"] = pairs
        if pairs:
            self.selected_pair.set(pairs[0])
        self.log_message(f"Available trading pairs updated ({market_type} Market).", "info")
        
    def perform_analysis(self):
        if not self.connector.is_connected:
            self.log_message("Please connect to the platform first.", "error")
            return
        
        if self.analysis_running:
            self.log_message("Analysis is already running.", "warning")
            return
        
        self.analysis_running = True
        self.start_analysis_button.config(state=tk.DISABLED)
        self.stop_analysis_button.config(state=tk.NORMAL)
        self.log_message("Starting auto analysis...", "info")
        
        threading.Thread(target=self._run_analysis_loop).start()

    def _run_analysis_loop(self):
        while self.analysis_running:
            pair = self.selected_pair.get()
            if not pair:
                self.log_message("Please select a trading pair.", "error")
                self.stop_analysis()
                break
            
            try:
                # Get candlestick data from API
                # Changed get_candle_data to get_candles and timeframe to 60 seconds
                candles_data = self.connector.get_candles(pair, timeframe=60, count=100)
                
                if not candles_data:
                    self.log_message(f"No candlestick data for {pair}.", "warning")
                    time.sleep(5) # Wait before trying again
                    continue
                
                signal = self.analyzer.generate_signal(pair, candles_data)
                
                if signal:
                    self.display_signal(signal)
                    self.log_message(f"Signal generated: {signal.direction} on {signal.pair} with confidence {signal.confidence:.2f}%", "success")
                else:
                    self.log_message(f"No clear signal for {pair}.", "info")
                
            except Exception as e:
                self.log_message(f"Error during analysis for {pair}: {e}", "error")
                # If connection error during analysis, disconnect and re-prompt SSID
                if "connection" in str(e).lower() or "disconnected" in str(e).lower():
                    self.disconnect_and_reprompt_ssid()
                    break # Exit analysis loop
            
            time.sleep(self.analysis_interval.get() * 60) # Wait for specified interval
        
        self.log_message("Auto analysis stopped.", "info")
        self.start_analysis_button.config(state=tk.NORMAL)
        self.stop_analysis_button.config(state=tk.DISABLED)
        
    def stop_analysis(self):
        self.analysis_running = False
        self.log_message("Stopping analysis...", "info")
        
    def display_signal(self, signal: TradingSignal):
        """Display trading signal in GUI"""
        signal_text = f"\nNew Signal:\n"
        signal_text += f"  Pair: {signal.pair}\n"
        signal_text += f"  Direction: {signal.direction}\n"
        signal_text += f"  Confidence: {signal.confidence:.2f}% ({signal.quality})\n"
        signal_text += f"  Price: {signal.price}\n"
        signal_text += f"  Timeframe: {signal.timeframe} minutes\n"
        signal_text += f"  Indicators Agreement: {signal.indicators_agreement}\n"
        signal_text += f"  Note: {signal.note}\n"
        signal_text += f"  Time: {signal.timestamp.strftime("%Y-%m-%d %H:%M:%S")}\n"
        
        self.signal_display.insert(tk.END, signal_text)
        self.signal_display.see(tk.END)
        
        # Here you can add logic to automatically place the trade if confidence is high
        # self.connector.place_trade(signal.pair, signal.direction, 10.0, signal.timeframe)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TradingAssistantGUI()
    app.run()
