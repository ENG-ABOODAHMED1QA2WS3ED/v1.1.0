# main_gui.py

import sys
import random
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QTextEdit, QMessageBox, QGroupBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QFont


# --- Mock Functions (for simulation purposes) ---
# These will be replaced by actual logic integrating with other modules.

def connect_to_platform(ssid: str) -> bool:
    """
    Simulates connecting to the trading platform with an SSID.
    In a real application, this would involve API calls.
    Returns True on success, False on failure.
    """
    print(f"Attempting to connect with SSID: {ssid}...")
    # Simulate a successful connection if the SSID is not empty.
    if ssid:
        print("Connection successful.")
        return True
    else:
        print("Connection failed: SSID is empty.")
        return False


def get_filtered_pairs() -> list[str]:
    """
    Simulates fetching trading pairs with a payout of 80% or higher.
    """
    print("Fetching high-payout trading pairs...")
    # In a real scenario, this would filter pairs based on live data.
    return [
        'EUR/USD', 'GBP/JPY', 'AUD/CAD', 'USD/CHF', 'NZD/USD', 'EUR/GBP'
    ]


def perform_full_analysis(pair: str, market_type: str) -> dict:
    """
    Simulates running the full analysis pipeline.
    This function will eventually call the signal_generator, ai_analysis,
    and fundamental_analysis modules.
    """
    print(f"Performing analysis for {pair} on {market_type} market...")
    directions = [('🟢 UP', '🔴 DOWN'), ('🔴 DOWN', '🟢 UP')]
    qualities = ['Premium', 'Standard', 'Basic']
    direction_choice = random.choice(directions)
    
    # This structure mirrors the output of the SignalGenerator class
    mock_result = {
        'Price': f"{random.uniform(1.05, 1.25):.5f}",
        'Pair': pair,
        'Direction': random.choice(direction_choice),
        'Duration': f"{random.choice([1, 2, 5])} min",
        'Confidence': f"{random.randint(65, 98)}%",
        'Quality': random.choice(qualities)
    }
    return mock_result


# --- GUI Classes ---

class LoginWindow(QWidget):
    """
    Login window to accept user SSID for platform connection.
    Emits a signal upon successful login.
    """
    login_successful = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trading Assistant - Login")
        self.setFixedSize(350, 150)
        self.init_ui()

    def init_ui(self):
        """Initializes the user interface of the login window."""
        layout = QVBoxLayout(self)

        # Title Label
        title_label = QLabel("Enter Your Session ID (SSID)")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # SSID Input
        self.ssid_input = QLineEdit()
        self.ssid_input.setPlaceholderText("Enter your PocketOption SSID here")
        self.ssid_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.ssid_input.returnPressed.connect(self.attempt_login)

        # Login Button
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
        """Handles the login logic."""
        ssid = self.ssid_input.text().strip()
        if connect_to_platform(ssid):
            self.login_successful.emit()
            self.close()
        else:
            QMessageBox.warning(
                self,
                "Login Failed",
                "Connection could not be established. Please check your SSID and try again."
            )


class MainWindow(QMainWindow):
    """
    The main application window that appears after a successful login.
    Contains controls for market and pair selection, and displays analysis results.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trading Assistant Dashboard")
        self.setGeometry(100, 100, 500, 480) # Adjusted height for new elements
        self.init_ui()

    def init_ui(self):
        """Initializes the main user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15) # Add space between elements

        # --- NEW: Account Info Header ---
        header_layout = QHBoxLayout()
        header_font = QFont("Arial", 10)
        
        self.account_id_label = QLabel("Account ID: N/A")
        self.account_id_label.setObjectName("account_id_label")
        self.account_id_label.setFont(header_font)
        
        self.balance_label = QLabel("Balance: N/A")
        self.balance_label.setObjectName("balance_label")
        self.balance_label.setFont(header_font)
        
        # Add a spacer to push labels to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        header_layout.addSpacerItem(spacer)
        header_layout.addWidget(self.account_id_label)
        header_layout.addWidget(self.balance_label)

        # --- Controls Group ---
        controls_group = QGroupBox("Analysis Configuration")
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(10)

        # Pair Selection
        pair_layout = QHBoxLayout()
        pair_label = QLabel("Trading Pair:")
        self.pair_combo = QComboBox()
        self.populate_pairs()
        pair_layout.addWidget(pair_label)
        pair_layout.addWidget(self.pair_combo)
        
        # --- NEW: Market Type Selection ---
        market_type_layout = QHBoxLayout()
        market_type_label = QLabel("Market Type:")
        self.market_type_combo = QComboBox()
        self.market_type_combo.setObjectName("market_type_combo")
        self.market_type_combo.addItems(["General", "OTC"])
        market_type_layout.addWidget(market_type_label)
        market_type_layout.addWidget(self.market_type_combo)
        
        # Add controls to the group layout in the specified order
        controls_layout.addLayout(pair_layout)
        controls_layout.addLayout(market_type_layout) # Added below the pair selection
        controls_group.setLayout(controls_layout)

        # --- Analysis Button ---
        self.analyze_button = QPushButton("Start Analysis")
        self.analyze_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.analyze_button.setStyleSheet(
            "QPushButton { background-color: #008CBA; color: white; padding: 10px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #007B9A; }"
        )
        self.analyze_button.clicked.connect(self.run_analysis)

        # --- Result Display ---
        result_group = QGroupBox("Trade Recommendation")
        result_layout = QVBoxLayout()
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setFont(QFont("Courier New", 11))
        self.result_display.setPlaceholderText("Analysis results will be displayed here...")
        result_layout.addWidget(self.result_display)
        result_group.setLayout(result_layout)
        
        # Add all widgets to the main layout
        main_layout.addLayout(header_layout) # Add the new header
        main_layout.addWidget(controls_group)
        main_layout.addWidget(self.analyze_button)
        main_layout.addWidget(result_group)
        
    def populate_pairs(self):
        """Fetches and populates the trading pair dropdown."""
        pairs = get_filtered_pairs()
        self.pair_combo.addItems(pairs)

    def run_analysis(self):
        """Triggers the analysis and displays the result."""
        # Updated to use the new QComboBox for market type
        market_type = self.market_type_combo.currentText()
        selected_pair = self.pair_combo.currentText()
        
        self.result_display.setText(f"Analyzing {selected_pair}...")
        QApplication.processEvents()  # Update the UI before the analysis

        # Simulate the analysis process
        result = perform_full_analysis(selected_pair, market_type)
        
        self.display_recommendation(result)

    def display_recommendation(self, result: dict):
        """Formats and displays the final recommendation."""
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


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    
    # The application flow:
    # 1. Create both windows.
    # 2. Connect the login window's success signal to show the main window.
    # 3. Show the login window first.
    login_window = LoginWindow()
    main_window = MainWindow()

    login_window.login_successful.connect(main_window.show)
    
    login_window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()