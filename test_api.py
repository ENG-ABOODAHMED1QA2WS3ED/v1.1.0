# test_api.py

import logging
import sys
from pocketoption_api import PocketOptionAPI

# تكوين التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_connection(ssid):
    """اختبار الاتصال بواجهة API"""
    try:
        # إنشاء اتصال
        api = PocketOptionAPI(ssid=ssid)
        
        # اختبار الوظائف الأساسية
        print(f"معرف الحساب: {api.get_account_id()}")
        print(f"الرصيد: {api.get_balance()}")
        
        # الحصول على الأصول المتاحة
        assets = api.get_assets()
        print(f"عدد الأصول المتاحة: {sum(len(assets[category]) for category in assets)}")
        
        # اختبار الحصول على بيانات الشموع
        pair = next(iter(assets['currency']), 'EURUSD')
        candles = api.get_candles(pair, timeframe=60, count=10)
        if candles:
            print(f"تم الحصول على {len(candles)} شمعة لـ {pair}")
            print(f"آخر شمعة: {candles[-1]}")
        else:
            print(f"فشل في الحصول على بيانات الشموع لـ {pair}")
            
        return True
    except Exception as e:
        print(f"خطأ: {str(e)}")
        return False

def extract_ssid_from_json(json_str):
    """استخراج SSID من سلسلة JSON"""
    import json
    import re
    
    try:
        # محاولة استخراج JSON من النص
        match = re.search(r'\["auth",\s*({.*})\]', json_str)
        if match:
            json_data = json.loads(match.group(1))
            if 'session' in json_data:
                return json_data['session']
    except Exception as e:
        print(f"خطأ في استخراج SSID: {str(e)}")
    
    # إذا لم يتم العثور على JSON، استخدم النص كما هو
    return json_str

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("الاستخدام: python test_api.py <ssid>")
        sys.exit(1)
        
    input_text = sys.argv[1]
    ssid = extract_ssid_from_json(input_text)
    print(f"استخدام SSID: {ssid}")
    success = test_connection(ssid)
    
    if success:
        print("تم الاتصال بنجاح واختبار جميع الوظائف الأساسية!")
    else:
        print("فشل الاختبار. يرجى التحقق من SSID والاتصال بالإنترنت.")