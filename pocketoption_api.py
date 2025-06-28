# pocketoption_api.py

import logging
from BinaryOptionsToolsV2.pocketoption import PocketOption



# Configure basic logging to monitor the API connection status
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PocketOptionAPI:
    """
    A wrapper class for the PocketOption API using the binaryoptiontools library.
    This class handles connection, authentication, and data retrieval.
    """
    def __init__(self, ssid: str):
        """
        Initializes the API connection using a session ID (SSID).

        Args:
            ssid (str): The session ID for authenticating with PocketOption.
        
        Raises:
            ConnectionError: If the connection to the API fails.
        """
        self.ssid = ssid
        self.api = None
        self._connect()

    def _connect(self):
        """
        Establishes and verifies the connection to the PocketOption servers.
        """
        logging.info("Attempting to establish connection with PocketOption...")
        self.api = PocketOption(ssid=self.ssid)
        
        # Check if the connection was successfully established
        if not self.api.check_connect():
            logging.error("Connection failed. The SSID may be invalid or expired, or there might be a network issue.")
            self.api = None
            raise ConnectionError("Failed to connect to PocketOption. Please check your SSID and network.")
        
        logging.info(f"Successfully connected. Account ID: {self.get_account_id()}")

    def get_balance(self) -> float:
        """
        Retrieves the current account balance.

        Returns:
            float: The real account balance, or None if not connected.
        """
        if not self.api:
            logging.warning("Cannot get balance, API not connected.")
            return None
        return self.api.get_balance()

    def get_account_id(self) -> int:
        """
        Retrieves the user's account ID.

        Returns:
            int: The user's account ID, or None if not connected.
        """
        if not self.api:
            logging.warning("Cannot get account ID, API not connected.")
            return None
        # The library often stores profile info after connection
        return self.api.profile.id if self.api.profile else None

    def get_assets(self) -> dict:
        """
        Fetches all available assets and categorizes them by type.

        Returns:
            dict: A dictionary where keys are asset types (crypto, commodity, otc, etc.)
                  and values are lists of asset names.
        """
        if not self.api:
            logging.warning("Cannot get assets, API not connected.")
            return {}

        logging.info("Fetching all available trading assets...")
        all_assets = self.api.get_all_asset()
        
        categorized_assets = {
            "crypto": [],
            "commodity": [],
            "otc": [],
            "stock": [],
            "currency": [],
            "other": []
        }

        for asset_name, asset_details in all_assets.items():
            # Check for OTC in the name first as it's a primary distinction
            if "OTC" in asset_name:
                categorized_assets["otc"].append(asset_name)
                continue
            
            # Use the 'type' field provided by the library for categorization
            asset_type = asset_details.get('type', 'other').lower()
            if asset_type in categorized_assets:
                categorized_assets[asset_type].append(asset_name)
            else:
                categorized_assets["other"].append(asset_name)
                
        logging.info(f"Found {len(all_assets)} assets, categorized successfully.")
        return categorized_assets
        
    def get_categorized_assets(self) -> dict:
        """
        Alias for get_assets() to maintain compatibility with main_app.py.
        
        Returns:
            dict: A dictionary of categorized assets.
        """
        return self.get_assets()
        
    def get_candles(self, pair: str, timeframe: int = 60, count: int = 100) -> list:
        """
        Fetches candlestick data for a specific trading pair.
        
        Args:
            pair (str): The trading pair to fetch candles for (e.g., "EURUSD").
            timeframe (int): The timeframe in seconds (default: 60 for 1 minute).
            count (int): The number of candles to retrieve (default: 100).
            
        Returns:
            list: A list of dictionaries containing candlestick data, or None if failed.
        """
        if not self.api:
            logging.warning("Cannot get candles, API not connected.")
            return None
            
        try:
            logging.info(f"Fetching {count} candles for {pair} at {timeframe}s timeframe...")
            # Convert timeframe from seconds to the format expected by the API
            interval_map = {
                15: "15s",  # 15 seconds
                60: "1m",   # 1 minute
                180: "3m",  # 3 minutes
                300: "5m",  # 5 minutes
                900: "15m", # 15 minutes
                3600: "1h", # 1 hour
                14400: "4h" # 4 hours
            }
            
            interval = interval_map.get(timeframe, "1m")  # Default to 1m if timeframe not in map
            candles = self.api.get_chart_data(pair=pair, interval=interval, count=count)
            
            if not candles:
                logging.warning(f"No candle data returned for {pair}")
                return None
                
            logging.info(f"Successfully fetched {len(candles)} candles for {pair}")
            return candles
            
        except Exception as e:
            logging.error(f"Error fetching candles for {pair}: {str(e)}")
            return None