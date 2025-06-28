# signal_generator.py

from typing import Dict, TYPE_CHECKING
from fundamental_analysis import FundamentalAnalysis
from ai_analysis import AIAnalysis

# To prevent circular imports while still allowing type hints for IDEs
if TYPE_CHECKING:
    from main_application import PocketOptionAPI, TechnicalAnalysis

class SignalGenerator:
    """
    Consolidates signals from technical, fundamental, and AI analyses
    to produce a single, actionable trading signal with a quality score.
    """
    def __init__(self, api: 'PocketOptionAPI', technical_analyzer: 'TechnicalAnalysis', 
                 fundamental_analyzer: FundamentalAnalysis, ai_analyzer: AIAnalysis):
        self.api = api
        self.technical_analyzer = technical_analyzer
        self.fundamental_analyzer = fundamental_analyzer
        self.ai_analyzer = ai_analyzer
        
        # Weights define the influence of each analysis type on the final score
        self.weights = {'technical': 0.30, 'ai': 0.45, 'fundamental': 0.25}
        self.fundamental_impact_map = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
        self.high_impact_override_confidence = 0.95

    def _get_signal_quality(self, confidence: float) -> str:
        """Determines signal quality based on the final confidence score."""
        if confidence >= 0.85: return 'Premium'
        if 0.70 <= confidence < 0.85: return 'Standard'
        return 'Basic'

    def generate_signal(self, pair: str, duration_minutes: int = 5) -> Dict[str, str]:
        """
        Orchestrates the entire analysis pipeline and generates a consolidated signal.
        """
        print(f"\nSignalGenerator: Starting full analysis for {pair}...")
        
        # --- 1. Fetch Live and Historical Data ---
        price = self.api.get_live_price(pair)
        candles = self.api.get_candles(pair)
        if price is None or candles is None:
            raise ConnectionError("Failed to fetch market data from the API.")

        # --- 2. Run All Analyses ---
        # Train AI model on first run
        if not self.ai_analyzer.is_trained:
            self.ai_analyzer.train(candles)
        
        technical_input = self.technical_analyzer.analyze(candles)
        fundamental_input = self.fundamental_analyzer.analyze_pair(pair)
        ai_input = self.ai_analyzer.predict(candles)

        # --- 3. Consolidate Signals ---
        fundamental_impact = fundamental_input.get('impact', 'low').lower()
        if fundamental_impact == 'high':
            # High-impact events override other signals
            final_direction = fundamental_input['direction']
            final_confidence = self.high_impact_override_confidence
        else:
            # Calculate weighted score for non-high-impact scenarios
            tech_direction = 1 if technical_input['signal'] == 'UP' else -1
            ai_direction = 1 if ai_input['signal'] == 'UP' else -1
            fund_direction = 1 if fundamental_input['direction'] == 'UP' else -1

            tech_score = tech_direction * technical_input['confidence'] * self.weights['technical']
            ai_score = ai_direction * ai_input['confidence'] * self.weights['ai']
            fund_score = fund_direction * self.fundamental_impact_map.get(fundamental_impact, 0.4) * self.weights['fundamental']

            total_score = tech_score + ai_score + fund_score
            final_direction = 'UP' if total_score >= 0 else 'DOWN'
            # Confidence is the magnitude of the score, normalized by the sum of weights
            final_confidence = abs(total_score) / sum(self.weights.values())
        
        # --- 4. Format Final Output ---
        quality = self._get_signal_quality(final_confidence)
        direction_icon = '🟢' if final_direction == 'UP' else '🔴'

        return {
            'Price': f"{price:.5f}",
            'Pair': pair,
            'Direction': f"{direction_icon} {final_direction}",
            'Duration': f"{duration_minutes} min",
            'Confidence': f"{final_confidence:.1%}",
            'Quality': quality,
            'Note': 'Execute this trade on pocketoption.'
        }