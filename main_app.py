# main_app.py

import sys
import random
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QTextEdit, QMessageBox, QGroupBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

# Backend and API imports
from pocketoption_api import PocketOptionAPI

# Import TechnicalAnalyzer from trading_assistant.py
from trading_assistant import TechnicalAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime

def perform_full_analysis(pair: str, market_type: str, api_instance) -> dict:
    """
    Performs a comprehensive technical analysis on the selected pair.
    Uses the TechnicalAnalyzer class to analyze multiple technical indicators.
    
    Args:
        pair (str): The trading pair to analyze
        market_type (str): The market type (General or OTC)
        api_instance: The PocketOptionAPI instance to fetch data
        
    Returns:
        dict: Analysis results including direction, confidence, and quality
    """
    print(f"Performing analysis for {pair} on {market_type} market...")
    
    try:
        # Get candle data from the API
        candles = api_instance.get_candles(pair, timeframe=60, count=100)
        
        if not candles or len(candles) < 50:
            return {
                'Price': "N/A",
                'Pair': pair,
                'Direction': "N/A",
                'Duration': "N/A",
                'Confidence': "0%",
                'Quality': "N/A",
                'Error': "Insufficient data"
            }
        
        # Initialize the technical analyzer
        analyzer = TechnicalAnalyzer()
        
        # Generate trading signal
        signal = analyzer.generate_signal(pair, candles)
        
        if not signal:
            return {
                'Price': f"{candles[-1]['close']:.5f}",
                'Pair': pair,
                'Direction': "NEUTRAL",
                'Duration': "N/A",
                'Confidence': "Below 70%",
                'Quality': "Low",
                'Error': "No clear signal"
            }
        
        # Format the result
        result = {
            'Price': f"{signal.price:.5f}",
            'Pair': signal.pair,
            'Direction': f"{'🟢 UP' if signal.direction == 'UP' else '🔴 DOWN'}",
            'Duration': f"{signal.timeframe} MIN",
            'Confidence': f"{signal.confidence:.1f}%",
            'Quality': signal.quality
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return {
            'Price': "N/A",
            'Pair': pair,
            'Direction': "N/A",
            'Duration': "N/A",
            'Confidence': "0%",
            'Quality': "N/A",
            'Error': f"Analysis failed: {str(e)}"
        }

# --- GUI Classes with Integrated Logic ---

class LoginWindow(QWidget):
    """
    Login window to accept user SSID for platform connection.
    Emits a signal with the API instance upon successful login.
    """
    # Signal now passes the created API object
    login_successful = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trading Assistant - Login")
        self.setFixedSize(350, 150)
        self.init_ui()

    def init_ui(self):
        """Initializes the user interface of the login window."""
        layout = QVBoxLayout(self)

        title_label = QLabel("Enter Your Session ID (SSID)")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ssid_input = QLineEdit()
        self.ssid_input.setPlaceholderText("Enter your PocketOption SSID here")
        self.ssid_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.ssid_input.returnPressed.connect(self.attempt_login)

        self.login_button = QPushButton("Connect to Platform")
        self.login_button.clicked.connect(self.attempt_login)
        self.login_button.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.login_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; padding: 6px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        
        layout.addWidget(title_label)
        layout.addWidget(self.ssid_input)
        layout.addWidget(self.login_button)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def attempt_login(self):
        """Handles the login logic by connecting to the actual API."""
        ssid = self.ssid_input.text().strip()
        if not ssid:
            QMessageBox.warning(self, "Login Failed", "SSID cannot be empty.")
            return

        self.login_button.setText("Connecting...")
        self.login_button.setEnabled(False)
        QApplication.processEvents()

        try:
            # --- REAL API CONNECTION ---
            api_instance = PocketOptionAPI(ssid=ssid)
            logging.info("Login successful, API instance created.")
            self.login_successful.emit(api_instance) # Emit signal with the instance
            self.close()
        except ConnectionError as e:
            # --- HANDLE CONNECTION FAILURE ---
            logging.error(f"Login failed: {e}")
            QMessageBox.critical(
                self,
                "Login Failed",
                f"Connection could not be established.\n\nError: {e}\n\nPlease check your SSID and network connection."
            )
        finally:
            self.login_button.setText("Connect to Platform")
            self.login_button.setEnabled(True)

class MainWindow(QMainWindow):
    """
    The main application window. Requires a connected PocketOptionAPI instance.
    """
    def __init__(self, api_instance: PocketOptionAPI):
        super().__init__()
        if not api_instance:
            raise ValueError("MainWindow requires a valid PocketOptionAPI instance.")
        self.api = api_instance
        
        self.setWindowTitle("AI Trading Assistant Dashboard")
        self.setGeometry(100, 100, 500, 480)
        
        self.init_ui()
        self.connect_signals()
        
        # --- Initial Data Load ---
        self.update_account_info()
        self.update_asset_list() # Initial population of assets

    def init_ui(self):
        """Initializes the main user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)

        # Account Info Header
        header_layout = QHBoxLayout()
        header_font = QFont("Arial", 10)
        self.account_id_label = QLabel("Account ID: Loading...")
        self.balance_label = QLabel("Balance: Loading...")
        self.account_id_label.setFont(header_font)
        self.balance_label.setFont(header_font)
        spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        header_layout.addSpacerItem(spacer)
        header_layout.addWidget(self.account_id_label)
        header_layout.addWidget(self.balance_label)

        # Controls Group
        controls_group = QGroupBox("Analysis Configuration")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(10)

        # Market Type Selection
        market_type_layout = QHBoxLayout()
        market_type_label = QLabel("Market Type:")
        self.market_type_combo = QComboBox()
        self.market_type_combo.addItems(["General", "OTC"])
        market_type_layout.addWidget(market_type_label)
        market_type_layout.addWidget(self.market_type_combo)
        
        # Pair Selection
        pair_layout = QHBoxLayout()
        pair_label = QLabel("Trading Pair:")
        self.pair_combo = QComboBox()
        pair_layout.addWidget(pair_label)
        pair_layout.addWidget(self.pair_combo)

        controls_layout.addLayout(market_type_layout)
        controls_layout.addLayout(pair_layout)

        # Analysis Button
        self.analyze_button = QPushButton("Start Analysis")
        self.analyze_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.analyze_button.setStyleSheet(
            "QPushButton { background-color: #008CBA; color: white; padding: 10px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #007B9A; }"
        )
        
        # Result Display
        result_group = QGroupBox("Trade Recommendation")
        result_layout = QVBoxLayout(result_group)
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setFont(QFont("Courier New", 11))
        self.result_display.setPlaceholderText("Configure analysis and click 'Start Analysis'.")
        result_layout.addWidget(self.result_display)
        
        # Add widgets to main layout
        main_layout.addLayout(header_layout)
        main_layout.addWidget(controls_group)
        main_layout.addWidget(self.analyze_button)
        main_layout.addWidget(result_group)

    def connect_signals(self):
        """Connects widget signals to corresponding slots."""
        self.analyze_button.clicked.connect(self.run_analysis)
        self.market_type_combo.currentTextChanged.connect(self.update_asset_list)

    def update_account_info(self):
        """Fetches and displays the account ID and balance from the API."""
        account_id = self.api.get_account_id()
        balance = self.api.get_balance()
        
        self.account_id_label.setText(f"Account ID: {account_id if account_id else 'N/A'}")
        self.balance_label.setText(f"Balance: ${balance:.2f}" if balance is not None else "Balance: N/A")

    def update_asset_list(self):
        """Fetches and populates the trading pair dropdown based on market type."""
        selected_market = self.market_type_combo.currentText().lower()
        self.pair_combo.clear()
        self.pair_combo.setEnabled(False)
        self.pair_combo.addItem("Loading assets...")
        QApplication.processEvents()

        try:
            categorized_assets = self.api.get_categorized_assets()
            assets_to_display = []
            
            if selected_market == "otc":
                assets_to_display = categorized_assets.get('otc', [])
            elif selected_market == "general":
                # Combine all non-OTC assets
                for category, asset_list in categorized_assets.items():
                    if category != 'otc':
                        assets_to_display.extend(asset_list)
            
            self.pair_combo.clear()
            if assets_to_display:
                assets_to_display.sort() # Sort for better usability
                self.pair_combo.addItems(assets_to_display)
                self.pair_combo.setEnabled(True)
            else:
                self.pair_combo.addItem(f"No assets found for {selected_market.upper()}")
                
        except Exception as e:
            logging.error(f"Failed to update asset list: {e}")
            self.pair_combo.clear()
            self.pair_combo.addItem("Error loading assets")
            QMessageBox.warning(self, "API Error", f"Could not fetch asset list: {e}")

    def run_analysis(self):
        """Triggers the analysis for the selected pair and displays the result."""
        market_type = self.market_type_combo.currentText()
        selected_pair = self.pair_combo.currentText()
        
        if not selected_pair or "Loading" in selected_pair or "Error" in selected_pair or "No assets" in selected_pair:
            QMessageBox.warning(self, "Warning", "Please select a valid trading pair.")
            return

        self.result_display.setText(f"Analyzing {selected_pair}...")
        self.analyze_button.setEnabled(False)
        QApplication.processEvents()

        try:
            # Use the updated perform_full_analysis function with the API instance
            result = perform_full_analysis(selected_pair, market_type, self.api)
            self.display_recommendation(result)
        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            self.result_display.setText(f"An error occurred during analysis:\n{e}")
        finally:
            self.analyze_button.setEnabled(True)

    def display_recommendation(self, result: dict):
        """Formats and displays the final recommendation."""
        if 'Error' in result:
            recommendation_text = (
                f"Price:      {result.get('Price', 'N/A')}\n"
                f"Pair:       {result.get('Pair', 'N/A')}\n"
                f"Status:     Analysis Incomplete\n"
                "----------------------------------------\n"
                f"Error:      {result.get('Error', 'Unknown error')}\n"
                "Please try again or select a different pair."
            )
        else:
            recommendation_text = (
                f"Price:      {result.get('Price', 'N/A')}\n"
                f"Pair:       {result.get('Pair', 'N/A')}\n"
                f"Direction:  {result.get('Direction', 'N/A')} ({result.get('Duration', 'N/A')})\n"
                f"Confidence: {result.get('Confidence', 'N/A')}\n"
                f"Quality:    {result.get('Quality', 'N/A')}\n"
                "----------------------------------------\n"
                "Note: Execute this trade on pocketoption."
            )
        self.result_display.setText(recommendation_text)

# --- Application Controller ---

class ApplicationController:
    """Manages the application flow and window transitions."""
    def __init__(self, app: QApplication):
        self.app = app
        self.login_window = None
        self.main_window = None

    def run(self):
        """Starts the application by showing the login window."""
        self.login_window = LoginWindow()
        self.login_window.login_successful.connect(self.on_login_success)
        self.login_window.show()

    def on_login_success(self, api_instance: PocketOptionAPI):
        """Handles successful login. Creates and shows the main window."""
        if self.login_window:
            self.login_window.close()
            
        self.main_window = MainWindow(api_instance=api_instance)
        self.main_window.show()

def main():
    """Main function to configure logging and run the application."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    app = QApplication(sys.argv)
    controller = ApplicationController(app)
    controller.run()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()