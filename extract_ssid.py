# extract_ssid.py

import json
import re
import sys

def extract_ssid_from_json_string(json_str):
    try:
        # طباعة النص المدخل بالضبط كما هو
        print(f"النص المدخل الأصلي: {repr(json_str)}")
        
        # البحث عن كلمة session والقيمة التي تليها
        session_match = re.search(r'session[":]\s*["]*([^"\s,}]+)', json_str)
        if session_match:
            ssid = session_match.group(1)
            print(f"تم العثور على قيمة session: {ssid}")
            return ssid
        else:
            print("لم يتم العثور على قيمة session")
    except Exception as e:
        print(f"خطأ في استخراج SSID: {e}")
    
    return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("الاستخدام: python extract_ssid.py <json_string>")
        sys.exit(1)
    
    json_str = sys.argv[1]
    ssid = extract_ssid_from_json_string(json_str)
    
    if ssid:
        print(f"تم استخراج SSID: {ssid}")
        # اختبار SSID المستخرج
        print("\nاختبار SSID المستخرج:")
        from subprocess import run
        run([sys.executable, "test_ssid.py", ssid])
    else:
        print("فشل في استخراج SSID من النص المقدم.")