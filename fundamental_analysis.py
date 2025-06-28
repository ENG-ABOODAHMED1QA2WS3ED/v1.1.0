# fundamental_analysis.py

import random
from datetime import date, timedelta
from typing import Dict, Any, Optional

import finnhub

class FundamentalAnalysis:
    """
    Performs fundamental analysis using the Finnhub API to find high-impact
    economic events relevant to a given currency pair.
    """
    def __init__(self, api_key: Optional[str]):
        self.client = None
        if not api_key:
            print("Warning: Finnhub API key not found. Fundamental analysis will be mocked.")
        else:
            try:
                self.client = finnhub.Client(api_key=api_key)
                self.client.company_profile2(symbol='AAPL') # Test connection
                print("FundamentalAnalysis: Finnhub client initialized successfully.")
            except Exception as e:
                print(f"Warning: Finnhub client failed to initialize: {e}. Fundamental analysis will be mocked.")
                self.client = None

    def _get_mock_analysis(self) -> Dict[str, Any]:
        """Provides a fallback mock result if the API is unavailable."""
        return {'impact': random.choice(['low', 'medium']), 'direction': random.choice(['UP', 'DOWN'])}

    def analyze_pair(self, pair: str) -> Dict[str, Any]:
        """
        Analyzes economic events for a specific currency pair for the current day.
        Returns a dictionary with 'impact' and a suggested 'direction'.
        """
        if not self.client:
            return self._get_mock_analysis()
            
        try:
            currencies = pair.split('/')
            if len(currencies) != 2:
                print(f"Warning: Invalid pair format for fundamental analysis: {pair}")
                return self._get_mock_analysis()

            base_currency, quote_currency = currencies[0], currencies[1]
            from_date = date.today().strftime("%Y-%m-%d")
            to_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

            calendar = self.client.economic_calendar(_from=from_date, to=to_date)
            events = calendar.get('economicCalendar', [])
            
            if not events:
                return {'impact': 'low', 'direction': 'UP'}

            high_impact_event = None
            # Find the first high-impact event relevant to the pair's currencies
            for event in events:
                event_country = event.get('country')
                if event_country in [base_currency, quote_currency] and event.get('impact') == 'high':
                    high_impact_event = event
                    break
            
            if high_impact_event:
                # Simplified logic: assume if actual > forecast, it's good for that currency's value
                actual = high_impact_event.get('actual', 0) or 0
                forecast = high_impact_event.get('forecast', 0) or 0
                
                direction = 'UP' if actual > forecast else 'DOWN'
                
                # If the event is for the quote currency, the effect on the pair is inverted
                # e.g., if USD strengthens in EUR/USD, the pair goes DOWN.
                if high_impact_event.get('country') == quote_currency:
                    direction = 'DOWN' if direction == 'UP' else 'UP'

                return {'impact': 'high', 'direction': direction}
            
            # If no high-impact events, return a medium impact with random direction
            return {'impact': 'medium', 'direction': random.choice(['UP', 'DOWN'])}

        except Exception as e:
            print(f"Error during fundamental analysis for {pair}: {e}")
            return self._get_mock_analysis()
