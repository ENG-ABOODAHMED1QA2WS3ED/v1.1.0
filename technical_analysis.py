# technical_analysis.py

import pandas as pd
import ta
import datetime
from random import randint

def fetch_candles(session, pair, timeframe='1m', limit=100):
    """
    جلب بيانات الشموع من المنصة عبر جلسة الاتصال
    """
    try:
        raw_data = session.get_chart_data(pair=pair, interval=timeframe, count=limit)
        candles = pd.DataFrame(raw_data)
        candles['timestamp'] = pd.to_datetime(candles['timestamp'], unit='s')
        candles.set_index('timestamp', inplace=True)
        candles.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        return candles
    except Exception as e:
        print("Error fetching candles:", str(e))
        return None

def analyze_candles(candles):
    """
    تحليل المؤشرات الفنية باستخدام مكتبة ta
    """
    # تطبيق المؤشرات
    candles['rsi'] = ta.momentum.RSIIndicator(candles['Close']).rsi()
    candles['macd'] = ta.trend.MACD(candles['Close']).macd()
    candles['ema20'] = ta.trend.EMAIndicator(candles['Close'], window=20).ema_indicator()
    candles['adx'] = ta.trend.ADXIndicator(candles['High'], candles['Low'], candles['Close']).adx()
    candles['bb_width'] = ta.volatility.BollingerBands(candles['Close']).bollinger_wband()

    # أخذ القيم الأخيرة فقط للتحليل
    last = candles.iloc[-1]

    score = 0
    total = 5

    if last['rsi'] < 30:
        direction = "🟢 UP"
        score += 1
    elif last['rsi'] > 70:
        direction = "🔴 DOWN"
        score += 1

    if last['macd'] > 0:
        direction = "🟢 UP"
        score += 1
    else:
        direction = "🔴 DOWN"

    if last['Close'] > last['ema20']:
        direction = "🟢 UP"
        score += 1
    else:
        direction = "🔴 DOWN"

    if last['adx'] > 20:
        score += 1

    if last['bb_width'] > 0.05:
        score += 1

    # الثقة والجودة
    confidence = round((score / total) * 100, 2)
    if confidence >= 90:
        quality = "Premium"
    elif confidence >= 75:
        quality = "Standerd"
    else:
        quality = "Basic"

    return {
        "direction": direction,
        "confidence": confidence,
        "quality": quality,
        "price": round(last['Close'], 5)
    }
