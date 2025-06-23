import tkinter as tk
from tkinter import ttk, messagebox

class TradingAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Trading Assistant - PocketOption")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")

        self.market_type = tk.StringVar()
        self.selected_pair = tk.StringVar()
        self.analysis_result = tk.StringVar()

        self.build_gui()

    def build_gui(self):
        title = tk.Label(self.root, text="AI Trading Assistant", font=("Arial", 20, "bold"), bg="#f0f0f0")
        title.pack(pady=10)

        # Market Type Selection
        market_frame = tk.Frame(self.root, bg="#f0f0f0")
        market_frame.pack(pady=10)

        tk.Label(market_frame, text="Select Market Type:", bg="#f0f0f0", font=("Arial", 12)).pack(side="left", padx=5)
        market_dropdown = ttk.Combobox(market_frame, textvariable=self.market_type, state="readonly", width=15)
        market_dropdown['values'] = ["General Market", "OTC Market"]
        market_dropdown.pack(side="left")
        market_dropdown.current(0)

        # Pair Selection
        pair_frame = tk.Frame(self.root, bg="#f0f0f0")
        pair_frame.pack(pady=10)

        tk.Label(pair_frame, text="Select Pair:", bg="#f0f0f0", font=("Arial", 12)).pack(side="left", padx=5)
        self.pair_dropdown = ttk.Combobox(pair_frame, textvariable=self.selected_pair, state="readonly", width=20)
        self.pair_dropdown.pack(side="left")
        self.pair_dropdown['values'] = ["Loading..."]  # سيتم تعبئتها لاحقًا ديناميكيًا

        # Analyze Button
        analyze_btn = tk.Button(self.root, text="Start Analysis", command=self.perform_analysis, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", padx=10, pady=5)
        analyze_btn.pack(pady=20)

        # Result Display Box
        result_frame = tk.LabelFrame(self.root, text="Analysis Result", font=("Arial", 12, "bold"), bg="white", width=500, height=200)
        result_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.result_label = tk.Label(result_frame, text="No analysis yet...", justify="left", bg="white", anchor="nw", font=("Consolas", 11))
        self.result_label.pack(fill="both", expand=True, padx=10, pady=10)

    def perform_analysis(self):
        # هذه فقط محاكاة مؤقتة، سيتم استبدالها لاحقًا بتحليل حقيقي
        example_result = """Price: 0.71283
Pair: AUDCAD_otc
Direction: 🔴 DOWN (5 MIN)
Confidence: 92.4%
Quality: Premium
Note: Execute this trade on PocketOption."""
        self.result_label.config(text=example_result)

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingAssistantApp(root)
    root.mainloop()
from login_manager import PocketOptionSession

# تسجيل الدخول
session_manager = PocketOptionSession()
if session_manager.request_ssid_and_connect():
    session = session_manager.get_session()
    user_data = session_manager.get_account_info()
    print("متصل ✅", user_data)
else:
    exit()
