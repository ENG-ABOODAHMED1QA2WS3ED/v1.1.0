# ai_predictor.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ta

class AIPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False

    def extract_features(self, df):
        """
        استخراج مؤشرات فنية للذكاء الصناعي
        """
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df['ema20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        df['bb_width'] = ta.volatility.BollingerBands(df['Close']).bollinger_wband()

        df = df.dropna()

        X = df[['rsi', 'macd', 'ema20', 'adx', 'bb_width']].copy()
        # التصنيف بناءً على حركة السعر التالية
        df['future'] = df['Close'].shift(-1)
        df['target'] = (df['future'] > df['Close']).astype(int)

        y = df['target']
        return X, y

    def train(self, candles_df):
        """
        تدريب النموذج على بيانات الشموع المستخرجة من المنصة
        """
        X, y = self.extract_features(candles_df)

        if len(X) < 30:
            print("❌ بيانات غير كافية لتدريب الذكاء الصناعي.")
            return False

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model.fit(X_train, y_train)
        self.trained = True
        print("✅ تم تدريب الذكاء الصناعي.")
        return True

    def predict_direction(self, last_candle):
        """
        توقع اتجاه السوق باستخدام آخر شمعة ومؤشراتها
        """
        if not self.trained:
            return "Unknown", 0

        input_data = last_candle[['rsi', 'macd', 'ema20', 'adx', 'bb_width']].values.reshape(1, -1)
        input_scaled = self.scaler.transform(input_data)

        prediction = self.model.predict(input_scaled)[0]
        prob = self.model.predict_proba(input_scaled)[0][prediction]

        direction = "🟢 UP" if prediction == 1 else "🔴 DOWN"
        confidence = round(prob * 100, 2)

        return direction, confidence