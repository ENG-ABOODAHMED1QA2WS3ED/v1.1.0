"""
Advanced Technical Analysis Module
وحدة التحليل الفني المتقدم

يتضمن جميع المؤشرات الفنية المطلوبة وتحليل أنماط الشموع
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings

# محاولة استيراد talib (اختياري)
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib غير متوفرة، سيتم استخدام الحسابات اليدوية")

# تجاهل التحذيرات غير المهمة
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class IndicatorResult:
    """نتيجة مؤشر فني"""
    name: str
    value: Union[float, Dict, List]
    signal: str  # BUY, SELL, NEUTRAL
    strength: float  # 0-100
    timeframe: str

@dataclass
class MarketCondition:
    """حالة السوق"""
    trend: str  # UPTREND, DOWNTREND, SIDEWAYS
    volatility: str  # HIGH, MEDIUM, LOW
    momentum: str  # STRONG, WEAK, NEUTRAL
    support_level: float
    resistance_level: float

class AdvancedTechnicalAnalyzer:
    """محلل فني متقدم مع جميع المؤشرات المطلوبة"""
    
    def __init__(self):
        self.timeframes = ['15s', '1m', '3m', '5m', '15m', '1h', '4h']
        self.min_periods = {
            'short': 14,
            'medium': 26,
            'long': 50
        }
        
    def prepare_data(self, candles: List[Dict]) -> pd.DataFrame:
        """تحضير البيانات للتحليل"""
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # التأكد من وجود جميع الأعمدة المطلوبة
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"العمود المطلوب {col} غير موجود في البيانات")
                return pd.DataFrame()
        
        # تحويل إلى أرقام
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna()
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, IndicatorResult]:
        """حساب المتوسطات المتحركة"""
        results = {}
        
        if len(df) < 50:
            return results
        
        try:
            # Simple Moving Averages
            sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
            sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
            
            # Exponential Moving Averages
            ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
            
            current_price = df['close'].iloc[-1]
            
            # تحديد الإشارة
            if current_price > sma_20 > sma_50:
                signal = "BUY"
                strength = 75
            elif current_price < sma_20 < sma_50:
                signal = "SELL"
                strength = 75
            else:
                signal = "NEUTRAL"
                strength = 50
            
            results['moving_average'] = IndicatorResult(
                name="Moving Average",
                value={'sma_20': sma_20, 'sma_50': sma_50, 'ema_12': ema_12, 'ema_26': ema_26},
                signal=signal,
                strength=strength,
                timeframe="current"
            )
            
        except Exception as e:
            logger.error(f"خطأ في حساب المتوسطات المتحركة: {e}")
        
        return results
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Optional[IndicatorResult]:
        """حساب مؤشر القوة النسبية RSI"""
        if len(df) < period + 1:
            return None
        
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # تحديد الإشارة
            if current_rsi < 30:
                signal = "BUY"
                strength = min(100, (30 - current_rsi) * 3)
            elif current_rsi > 70:
                signal = "SELL"
                strength = min(100, (current_rsi - 70) * 3)
            else:
                signal = "NEUTRAL"
                strength = 50
            
            return IndicatorResult(
                name="RSI",
                value=current_rsi,
                signal=signal,
                strength=strength,
                timeframe="current"
            )
            
        except Exception as e:
            logger.error(f"خطأ في حساب RSI: {e}")
            return None
    
    def calculate_macd(self, df: pd.DataFrame) -> Optional[IndicatorResult]:
        """حساب مؤشر MACD"""
        if len(df) < 26:
            return None
        
        try:
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]
            
            # تحديد الإشارة
            if current_macd > current_signal and current_histogram > 0:
                signal = "BUY"
                strength = min(100, abs(current_histogram) * 1000)
            elif current_macd < current_signal and current_histogram < 0:
                signal = "SELL"
                strength = min(100, abs(current_histogram) * 1000)
            else:
                signal = "NEUTRAL"
                strength = 50
            
            return IndicatorResult(
                name="MACD",
                value={
                    'macd': current_macd,
                    'signal': current_signal,
                    'histogram': current_histogram
                },
                signal=signal,
                strength=strength,
                timeframe="current"
            )
            
        except Exception as e:
            logger.error(f"خطأ في حساب MACD: {e}")
            return None
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Optional[IndicatorResult]:
        """حساب نطاقات بولينجر"""
        if len(df) < period:
            return None
        
        try:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            
            current_price = df['close'].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_middle = sma.iloc[-1]
            
            # تحديد الإشارة
            if current_price < current_lower:
                signal = "BUY"
                strength = min(100, (current_lower - current_price) / current_lower * 100 * 10)
            elif current_price > current_upper:
                signal = "SELL"
                strength = min(100, (current_price - current_upper) / current_upper * 100 * 10)
            else:
                signal = "NEUTRAL"
                strength = 50
            
            return IndicatorResult(
                name="Bollinger Bands",
                value={
                    'upper': current_upper,
                    'middle': current_middle,
                    'lower': current_lower,
                    'current_price': current_price
                },
                signal=signal,
                strength=strength,
                timeframe="current"
            )
            
        except Exception as e:
            logger.error(f"خطأ في حساب Bollinger Bands: {e}")
            return None
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Optional[IndicatorResult]:
        """حساب مؤشر الستوكاستيك"""
        if len(df) < k_period:
            return None
        
        try:
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            
            k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            current_k = k_percent.iloc[-1]
            current_d = d_percent.iloc[-1]
            
            # تحديد الإشارة
            if current_k < 20 and current_d < 20:
                signal = "BUY"
                strength = min(100, (20 - min(current_k, current_d)) * 4)
            elif current_k > 80 and current_d > 80:
                signal = "SELL"
                strength = min(100, (min(current_k, current_d) - 80) * 4)
            else:
                signal = "NEUTRAL"
                strength = 50
            
            return IndicatorResult(
                name="Stochastic",
                value={'k': current_k, 'd': current_d},
                signal=signal,
                strength=strength,
                timeframe="current"
            )
            
        except Exception as e:
            logger.error(f"خطأ في حساب Stochastic: {e}")
            return None
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Optional[IndicatorResult]:
        """حساب مؤشر ADX"""
        if len(df) < period + 1:
            return None
        
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # حساب True Range
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # حساب Directional Movement
            dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                              np.maximum(high - np.roll(high, 1), 0), 0)
            dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                               np.maximum(np.roll(low, 1) - low, 0), 0)
            
            # تطبيق المتوسط المتحرك
            tr_smooth = pd.Series(tr).rolling(window=period).mean()
            dm_plus_smooth = pd.Series(dm_plus).rolling(window=period).mean()
            dm_minus_smooth = pd.Series(dm_minus).rolling(window=period).mean()
            
            # حساب DI
            di_plus = 100 * dm_plus_smooth / tr_smooth
            di_minus = 100 * dm_minus_smooth / tr_smooth
            
            # حساب ADX
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window=period).mean()
            
            current_adx = adx.iloc[-1]
            current_di_plus = di_plus.iloc[-1]
            current_di_minus = di_minus.iloc[-1]
            
            # تحديد الإشارة
            if current_adx > 25:
                if current_di_plus > current_di_minus:
                    signal = "BUY"
                    strength = min(100, current_adx)
                else:
                    signal = "SELL"
                    strength = min(100, current_adx)
            else:
                signal = "NEUTRAL"
                strength = current_adx
            
            return IndicatorResult(
                name="ADX",
                value={
                    'adx': current_adx,
                    'di_plus': current_di_plus,
                    'di_minus': current_di_minus
                },
                signal=signal,
                strength=strength,
                timeframe="current"
            )
            
        except Exception as e:
            logger.error(f"خطأ في حساب ADX: {e}")
            return None
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> Optional[IndicatorResult]:
        """حساب مؤشر CCI"""
        if len(df) < period:
            return None
        
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (typical_price - sma_tp) / (0.015 * mad)
            current_cci = cci.iloc[-1]
            
            # تحديد الإشارة
            if current_cci < -100:
                signal = "BUY"
                strength = min(100, abs(current_cci + 100) / 2)
            elif current_cci > 100:
                signal = "SELL"
                strength = min(100, (current_cci - 100) / 2)
            else:
                signal = "NEUTRAL"
                strength = 50
            
            return IndicatorResult(
                name="CCI",
                value=current_cci,
                signal=signal,
                strength=strength,
                timeframe="current"
            )
            
        except Exception as e:
            logger.error(f"خطأ في حساب CCI: {e}")
            return None
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> Optional[IndicatorResult]:
        """حساب مؤشر Williams %R"""
        if len(df) < period:
            return None
        
        try:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            current_wr = williams_r.iloc[-1]
            
            # تحديد الإشارة
            if current_wr < -80:
                signal = "BUY"
                strength = min(100, abs(current_wr + 80) * 5)
            elif current_wr > -20:
                signal = "SELL"
                strength = min(100, (20 + current_wr) * 5)
            else:
                signal = "NEUTRAL"
                strength = 50
            
            return IndicatorResult(
                name="Williams %R",
                value=current_wr,
                signal=signal,
                strength=strength,
                timeframe="current"
            )
            
        except Exception as e:
            logger.error(f"خطأ في حساب Williams %R: {e}")
            return None
    
    def analyze_candlestick_patterns(self, df: pd.DataFrame) -> List[IndicatorResult]:
        """تحليل أنماط الشموع"""
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        try:
            # الحصول على آخر 3 شموع
            recent_candles = df.tail(3)
            current = recent_candles.iloc[-1]
            previous = recent_candles.iloc[-2]
            
            # حساب أحجام الجسم والظلال
            current_body = abs(current['close'] - current['open'])
            current_range = current['high'] - current['low']
            current_upper_shadow = current['high'] - max(current['open'], current['close'])
            current_lower_shadow = min(current['open'], current['close']) - current['low']
            
            # نمط Doji
            if current_range > 0 and current_body / current_range < 0.1:
                patterns.append(IndicatorResult(
                    name="Doji",
                    value="Doji Pattern Detected",
                    signal="NEUTRAL",
                    strength=60,
                    timeframe="current"
                ))
            
            # نمط Hammer
            if (current_lower_shadow > 2 * current_body and 
                current_upper_shadow < current_body and
                current_body > 0):
                
                signal = "BUY" if current['close'] > current['open'] else "NEUTRAL"
                patterns.append(IndicatorResult(
                    name="Hammer",
                    value="Hammer Pattern Detected",
                    signal=signal,
                    strength=70,
                    timeframe="current"
                ))
            
            # نمط Engulfing
            if len(recent_candles) >= 2:
                prev_body = abs(previous['close'] - previous['open'])
                
                # Bullish Engulfing
                if (previous['close'] < previous['open'] and  # شمعة هابطة سابقة
                    current['close'] > current['open'] and   # شمعة صاعدة حالية
                    current['open'] < previous['close'] and  # فتح أقل من إغلاق السابقة
                    current['close'] > previous['open'] and  # إغلاق أعلى من فتح السابقة
                    current_body > prev_body):               # جسم أكبر
                    
                    patterns.append(IndicatorResult(
                        name="Bullish Engulfing",
                        value="Bullish Engulfing Pattern Detected",
                        signal="BUY",
                        strength=80,
                        timeframe="current"
                    ))
                
                # Bearish Engulfing
                elif (previous['close'] > previous['open'] and  # شمعة صاعدة سابقة
                      current['close'] < current['open'] and   # شمعة هابطة حالية
                      current['open'] > previous['close'] and  # فتح أعلى من إغلاق السابقة
                      current['close'] < previous['open'] and  # إغلاق أقل من فتح السابقة
                      current_body > prev_body):               # جسم أكبر
                    
                    patterns.append(IndicatorResult(
                        name="Bearish Engulfing",
                        value="Bearish Engulfing Pattern Detected",
                        signal="SELL",
                        strength=80,
                        timeframe="current"
                    ))
            
        except Exception as e:
            logger.error(f"خطأ في تحليل أنماط الشموع: {e}")
        
        return patterns
    
    def determine_market_condition(self, df: pd.DataFrame) -> MarketCondition:
        """تحديد حالة السوق"""
        if len(df) < 50:
            return MarketCondition("UNKNOWN", "UNKNOWN", "UNKNOWN", 0.0, 0.0)
        
        try:
            # تحديد الاتجاه
            sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
            sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                trend = "UPTREND"
            elif current_price < sma_20 < sma_50:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"
            
            # تحديد التقلبات
            atr = df['high'].subtract(df['low']).rolling(window=14).mean().iloc[-1]
            price_range = df['close'].iloc[-1]
            volatility_ratio = atr / price_range * 100
            
            if volatility_ratio > 2:
                volatility = "HIGH"
            elif volatility_ratio > 1:
                volatility = "MEDIUM"
            else:
                volatility = "LOW"
            
            # تحديد الزخم
            rsi_result = self.calculate_rsi(df)
            if rsi_result and rsi_result.strength > 70:
                momentum = "STRONG"
            elif rsi_result and rsi_result.strength > 40:
                momentum = "NEUTRAL"
            else:
                momentum = "WEAK"
            
            # تحديد مستويات الدعم والمقاومة
            recent_highs = df['high'].tail(20)
            recent_lows = df['low'].tail(20)
            
            resistance_level = recent_highs.max()
            support_level = recent_lows.min()
            
            return MarketCondition(
                trend=trend,
                volatility=volatility,
                momentum=momentum,
                support_level=support_level,
                resistance_level=resistance_level
            )
            
        except Exception as e:
            logger.error(f"خطأ في تحديد حالة السوق: {e}")
            return MarketCondition("UNKNOWN", "UNKNOWN", "UNKNOWN", 0.0, 0.0)
    
    def comprehensive_analysis(self, candles: List[Dict]) -> Dict:
        """تحليل شامل للبيانات"""
        df = self.prepare_data(candles)
        
        if df.empty:
            return {'error': 'بيانات غير صالحة'}
        
        results = {
            'indicators': {},
            'patterns': [],
            'market_condition': None,
            'overall_signal': 'NEUTRAL',
            'confidence': 0,
            'timestamp': datetime.now()
        }
        
        try:
            # حساب جميع المؤشرات
            indicators = {}
            
            # المتوسطات المتحركة
            ma_results = self.calculate_moving_averages(df)
            indicators.update(ma_results)
            
            # RSI
            rsi_result = self.calculate_rsi(df)
            if rsi_result:
                indicators['rsi'] = rsi_result
            
            # MACD
            macd_result = self.calculate_macd(df)
            if macd_result:
                indicators['macd'] = macd_result
            
            # Bollinger Bands
            bb_result = self.calculate_bollinger_bands(df)
            if bb_result:
                indicators['bollinger_bands'] = bb_result
            
            # Stochastic
            stoch_result = self.calculate_stochastic(df)
            if stoch_result:
                indicators['stochastic'] = stoch_result
            
            # ADX
            adx_result = self.calculate_adx(df)
            if adx_result:
                indicators['adx'] = adx_result
            
            # CCI
            cci_result = self.calculate_cci(df)
            if cci_result:
                indicators['cci'] = cci_result
            
            # Williams %R
            wr_result = self.calculate_williams_r(df)
            if wr_result:
                indicators['williams_r'] = wr_result
            
            results['indicators'] = indicators
            
            # تحليل أنماط الشموع
            patterns = self.analyze_candlestick_patterns(df)
            results['patterns'] = patterns
            
            # تحديد حالة السوق
            market_condition = self.determine_market_condition(df)
            results['market_condition'] = market_condition
            
            # تحديد الإشارة الإجمالية
            buy_signals = 0
            sell_signals = 0
            total_strength = 0
            total_indicators = 0
            
            # جمع إشارات المؤشرات
            for indicator in indicators.values():
                total_indicators += 1
                total_strength += indicator.strength
                
                if indicator.signal == "BUY":
                    buy_signals += 1
                elif indicator.signal == "SELL":
                    sell_signals += 1
            
            # جمع إشارات الأنماط
            for pattern in patterns:
                total_indicators += 1
                total_strength += pattern.strength
                
                if pattern.signal == "BUY":
                    buy_signals += 1
                elif pattern.signal == "SELL":
                    sell_signals += 1
            
            # تحديد الإشارة النهائية
            if total_indicators > 0:
                confidence = total_strength / total_indicators
                
                if buy_signals > sell_signals and buy_signals >= 3:
                    overall_signal = "BUY"
                elif sell_signals > buy_signals and sell_signals >= 3:
                    overall_signal = "SELL"
                else:
                    overall_signal = "NEUTRAL"
                    confidence = min(confidence, 60)  # تقليل الثقة للإشارات المحايدة
                
                results['overall_signal'] = overall_signal
                results['confidence'] = min(100, confidence)
                results['buy_signals'] = buy_signals
                results['sell_signals'] = sell_signals
                results['total_indicators'] = total_indicators
            
        except Exception as e:
            logger.error(f"خطأ في التحليل الشامل: {e}")
            results['error'] = str(e)
        
        return results

# مثال على الاستخدام
if __name__ == "__main__":
    # إنشاء بيانات تجريبية
    import random
    from datetime import datetime, timedelta
    
    candles = []
    base_price = 1.1000
    
    for i in range(100):
        open_price = base_price + random.uniform(-0.01, 0.01)
        close_price = open_price + random.uniform(-0.005, 0.005)
        high_price = max(open_price, close_price) + random.uniform(0, 0.003)
        low_price = min(open_price, close_price) - random.uniform(0, 0.003)
        
        candles.append({
            'open': round(open_price, 5),
            'high': round(high_price, 5),
            'low': round(low_price, 5),
            'close': round(close_price, 5),
            'timestamp': datetime.now() - timedelta(minutes=100-i)
        })
        
        base_price = close_price
    
    # تشغيل التحليل
    analyzer = AdvancedTechnicalAnalyzer()
    results = analyzer.comprehensive_analysis(candles)
    
    print("نتائج التحليل الفني:")
    print(f"الإشارة الإجمالية: {results['overall_signal']}")
    print(f"مستوى الثقة: {results['confidence']:.1f}%")
    print(f"عدد المؤشرات: {results.get('total_indicators', 0)}")
    print(f"إشارات الشراء: {results.get('buy_signals', 0)}")
    print(f"إشارات البيع: {results.get('sell_signals', 0)}")

