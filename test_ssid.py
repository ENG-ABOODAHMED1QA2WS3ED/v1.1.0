# test_ssid.py

import sys
from BinaryOptionsToolsV2.pocketoption import PocketOption

def test_ssid(ssid_str):
    try:
        print(f"محاولة الاتصال باستخدام SSID: {ssid_str}")
        api = PocketOption(ssid=ssid_str)
        print("تم إنشاء كائن PocketOption بنجاح")
        
        # اختبار الاتصال
        if api.check_connect():
            print("تم الاتصال بنجاح!")
            print(f"معرف الحساب: {api.profile.id if api.profile else 'غير متوفر'}")
            print(f"الرصيد: {api.get_balance()}")
            return True
        else:
            print("فشل الاتصال رغم إنشاء الكائن بنجاح")
            return False
    except Exception as e:
        print(f"خطأ: {type(e).__name__}, {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("الاستخدام: python test_ssid.py <ssid>")
        sys.exit(1)
        
    ssid = sys.argv[1]
    success = test_ssid(ssid)
    
    if success:
        print("تم الاختبار بنجاح!")
    else:
        print("فشل الاختبار.")