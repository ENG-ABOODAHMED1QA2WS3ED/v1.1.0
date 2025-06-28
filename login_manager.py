# login_manager.py

import tkinter as tk
from tkinter import simpledialog, messagebox
from BinaryOptionsToolsV2.pocketoption import PocketOption

class PocketOptionSession:
    def __init__(self):
        self.session = None
        self.account_data = {}

    def request_ssid_and_connect(self):
        # واجهة إدخال SSID
        root = tk.Tk()
        root.withdraw()
        ssid = simpledialog.askstring("Login", "Enter your PocketOption SSID:")

        if not ssid:
            messagebox.showerror("Login Failed", "No SSID provided.")
            return False

        try:
            session = PocketOption(ssid=ssid)

            # التحقق من صحة الاتصال
            balance = session.get_balance()
            uid = session.profile.id if session.profile else None
            acc_type = "DEMO" if session.is_demo else "REAL"

            self.session = session
            self.account_data = {
                "uid": uid,
                "balance": balance,
                "account_type": acc_type
            }

            messagebox.showinfo("Login Success",
                f"UID: {uid}\nBalance: {balance}\nAccount Type: {acc_type}")
            return True

        except Exception as e:
            messagebox.showerror("Connection Error", f"Error: {str(e)}")
            return False

    def get_session(self):
        return self.session

    def get_account_info(self):
        return self.account_data
