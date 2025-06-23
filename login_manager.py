# login_manager.py

import tkinter as tk
from tkinter import simpledialog, messagebox
from BinaryOptionsToolsV2.pocketoption import syncronous

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
            session = syncronous.PocketOption()
            session.connect(ssid=ssid)

            # التحقق من صحة الاتصال
            balance = session.get_balance()
            uid = session.get_user_id()
            acc_type = session.get_account_type()

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
