import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier

class AIAnalysis:
    """
    تستخدم نموذج RandomForestClassifier للتنبؤ باتجاهات السوق بناءً على البيانات التاريخية.
    تم تصميم هذه الوحدة لتكون متكاملة وقوية.
    """
    def __init__(self, n_estimators: int = 100, future_periods: int = 5):
        """
        يقوم بتهيئة محلل الذكاء الاصطناعي.

        Args:
            n_estimators (int): عدد الأشجار في نموذج الغابة العشوائية.
            future_periods (int): عدد الفترات الزمنية المستقبلية للتنبؤ بحركة السعر.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        self.future_periods = future_periods
        self.trained_features: Optional[List[str]] = None
        self.is_trained = False

    def _prepare_features(self, df: pd.DataFrame):
        """
        طريقة خاصة لإعداد الميزات والمتغير المستهدف (target).
        المتغير المستهدف 'target' هو متغير ثنائي:
        - 1: إذا كان سعر الإغلاق بعد `future_periods` أعلى.
        - 0: إذا كان سعر الإغلاق بعد `future_periods` أقل أو مساوٍ.
        """
        data = df.copy()
        data['future_close'] = data['close'].shift(-self.future_periods)
        data['target'] = (data['future_close'] > data['close']).astype(int)
        
        potential_features = [
            'RSI', 'MACD_diff', 'EMA_50', 'EMA_200', 'ADX',
            'BOLL_upper', 'BOLL_lower', 'volume'
        ]
        features = [col for col in potential_features if col in data.columns]
        
        if self.trained_features is None:
            self.trained_features = features
            
        data.dropna(subset=features + ['target'], inplace=True)
        return data[self.trained_features], data['target']

    def train(self, df: pd.DataFrame):
        """
        تدريب النموذج على البيانات التاريخية. يتم تشغيله مرة واحدة فقط.
        """
        if self.is_trained:
            return
        
        print("AIAnalysis: Preparing features and training model...")
        X, y = self._prepare_features(df)
        
        if X.empty:
            print("AIAnalysis Warning: No training data available after preparation.")
            return
        
        # يتم تدريب النموذج على كامل البيانات التاريخية المتاحة للحصول على أفضل أداء
        self.model.fit(X, y)
        self.is_trained = True
        print("AIAnalysis: Model training complete.")

    def predict(self, df_slice: pd.DataFrame) -> Dict[str, Any]:
        """
        يقوم بعمل تنبؤ على أحدث نقطة بيانات.

        Returns:
            Dict[str, Any]: قاموس يحتوي على 'signal' و 'confidence'.
        """
        if not self.is_trained or self.trained_features is None:
            raise RuntimeError("AI model has not been trained yet. Call train() first.")

        X_pred = df_slice[self.trained_features].tail(1)
        if X_pred.isnull().values.any():
            print("AIAnalysis Warning: NaN values in prediction input. Returning neutral signal.")
            return {'signal': 'UP', 'confidence': 0.5}

        prediction = self.model.predict(X_pred)[0]
        probabilities = self.model.predict_proba(X_pred)[0]
        
        signal = 'UP' if prediction == 1 else 'DOWN'
        confidence = probabilities[prediction]
        
        return {'signal': signal, 'confidence': confidence}