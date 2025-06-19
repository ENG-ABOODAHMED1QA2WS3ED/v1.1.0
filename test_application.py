"""
تطبيق اختبار مبسط لمساعد تداول الخيارات الثنائية
يعمل بدون tkinter للاختبار في بيئة Linux
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pocketoption_api import PocketOptionAPI
from technical_analysis import AdvancedTechnicalAnalyzer
import time
from datetime import datetime

def test_api_connection():
    """اختبار الاتصال بواجهة برمجة التطبيقات"""
    print("🔄 اختبار الاتصال بواجهة برمجة التطبيقات...")
    
    api = PocketOptionAPI()
    
    # اختبار الاتصال بـ SSID وهمي
    test_ssid = "test_ssid_123456789_for_testing"
    
    if api.connect(test_ssid):
        print("✅ تم الاتصال بنجاح!")
        
        # اختبار الحصول على معلومات الحساب
        account_info = api.get_account_info()
        print(f"📊 معلومات الحساب: {account_info}")
        
        # اختبار الحصول على الأصول المتاحة
        assets_otc = api.get_available_assets("OTC")
        assets_regular = api.get_available_assets("Regular")
        print(f"💱 أصول OTC: {len(assets_otc)} أصل")
        print(f"💱 أصول عادية: {len(assets_regular)} أصل")
        
        # اختبار الحصول على بيانات الشموع
        if assets_otc:
            test_asset = assets_otc[0]
            candles = api.get_candles(test_asset, count=50)
            print(f"🕯️ بيانات الشموع لـ {test_asset}: {len(candles)} شمعة")
            
            if candles:
                latest_candle = candles[-1]
                print(f"📈 آخر شمعة: O:{latest_candle['open']:.5f} H:{latest_candle['high']:.5f} L:{latest_candle['low']:.5f} C:{latest_candle['close']:.5f}")
        
        api.disconnect()
        return True
    else:
        print("❌ فشل في الاتصال!")
        return False

def test_technical_analysis():
    """اختبار التحليل الفني"""
    print("\n🔄 اختبار التحليل الفني...")
    
    analyzer = AdvancedTechnicalAnalyzer()
    
    # إنشاء بيانات تجريبية
    import random
    from datetime import datetime, timedelta
    
    candles = []
    base_price = 1.1000
    
    print("📊 إنشاء بيانات تجريبية...")
    for i in range(100):
        open_price = base_price + random.uniform(-0.01, 0.01)
        close_price = open_price + random.uniform(-0.005, 0.005)
        high_price = max(open_price, close_price) + random.uniform(0, 0.003)
        low_price = min(open_price, close_price) - random.uniform(0, 0.003)
        
        candles.append({
            'open': round(open_price, 5),
            'high': round(high_price, 5),
            'low': round(low_price, 5),
            'close': round(close_price, 5),
            'timestamp': datetime.now() - timedelta(minutes=100-i)
        })
        
        base_price = close_price
    
    print(f"✅ تم إنشاء {len(candles)} شمعة")
    
    # تشغيل التحليل الشامل
    print("🔍 تشغيل التحليل الشامل...")
    results = analyzer.comprehensive_analysis(candles)
    
    if 'error' in results:
        print(f"❌ خطأ في التحليل: {results['error']}")
        return False
    
    print("✅ تم التحليل بنجاح!")
    print(f"📊 الإشارة الإجمالية: {results['overall_signal']}")
    print(f"🎯 مستوى الثقة: {results['confidence']:.1f}%")
    print(f"📈 إشارات الشراء: {results.get('buy_signals', 0)}")
    print(f"📉 إشارات البيع: {results.get('sell_signals', 0)}")
    print(f"🔢 إجمالي المؤشرات: {results.get('total_indicators', 0)}")
    
    # عرض تفاصيل المؤشرات
    print("\n📋 تفاصيل المؤشرات:")
    for name, indicator in results['indicators'].items():
        print(f"  • {indicator.name}: {indicator.signal} (قوة: {indicator.strength:.1f})")
    
    # عرض الأنماط المكتشفة
    if results['patterns']:
        print("\n🕯️ أنماط الشموع المكتشفة:")
        for pattern in results['patterns']:
            print(f"  • {pattern.name}: {pattern.signal} (قوة: {pattern.strength:.1f})")
    
    # عرض حالة السوق
    if results['market_condition']:
        mc = results['market_condition']
        print(f"\n🌊 حالة السوق:")
        print(f"  • الاتجاه: {mc.trend}")
        print(f"  • التقلبات: {mc.volatility}")
        print(f"  • الزخم: {mc.momentum}")
        print(f"  • الدعم: {mc.support_level:.5f}")
        print(f"  • المقاومة: {mc.resistance_level:.5f}")
    
    return True

def test_signal_generation():
    """اختبار توليد الإشارات"""
    print("\n🔄 اختبار توليد الإشارات...")
    
    api = PocketOptionAPI()
    analyzer = AdvancedTechnicalAnalyzer()
    
    # محاكاة الاتصال
    if api.connect("test_ssid_for_signal_generation"):
        print("✅ تم الاتصال للاختبار")
        
        # الحصول على أصل للاختبار
        assets = api.get_available_assets("OTC")
        if assets:
            test_asset = assets[0]
            print(f"🎯 اختبار الأصل: {test_asset}")
            
            # الحصول على بيانات الشموع
            candles = api.get_candles(test_asset, count=100)
            
            if candles:
                print(f"📊 تم الحصول على {len(candles)} شمعة")
                
                # تشغيل التحليل
                results = analyzer.comprehensive_analysis(candles)
                
                if results['overall_signal'] != 'NEUTRAL' and results['confidence'] >= 70:
                    print("🎯 إشارة تداول مولدة:")
                    print(f"  • الزوج: {test_asset}")
                    print(f"  • الاتجاه: {results['overall_signal']}")
                    print(f"  • الثقة: {results['confidence']:.1f}%")
                    print(f"  • السعر: {candles[-1]['close']:.5f}")
                    
                    # تحديد جودة الإشارة
                    if results['confidence'] >= 90:
                        quality = "Premium"
                    elif results['confidence'] >= 80:
                        quality = "Standard"
                    else:
                        quality = "Basic"
                    
                    print(f"  • الجودة: {quality}")
                    print(f"  • الوقت: {datetime.now().strftime('%H:%M:%S')}")
                    print("  • ملاحظة: Execute this trade on PocketOption")
                    
                    return True
                else:
                    print("⚠️ لا توجد إشارة واضحة في البيانات التجريبية")
                    return True
            else:
                print("❌ لم يتم الحصول على بيانات الشموع")
                return False
        else:
            print("❌ لم يتم الحصول على الأصول")
            return False
    else:
        print("❌ فشل في الاتصال للاختبار")
        return False

def main():
    """الدالة الرئيسية للاختبار"""
    print("🚀 بدء اختبار مساعد تداول الخيارات الثنائية")
    print("=" * 60)
    
    # اختبار واجهة برمجة التطبيقات
    api_test = test_api_connection()
    
    # اختبار التحليل الفني
    analysis_test = test_technical_analysis()
    
    # اختبار توليد الإشارات
    signal_test = test_signal_generation()
    
    print("\n" + "=" * 60)
    print("📋 ملخص نتائج الاختبار:")
    print(f"  • اختبار واجهة برمجة التطبيقات: {'✅ نجح' if api_test else '❌ فشل'}")
    print(f"  • اختبار التحليل الفني: {'✅ نجح' if analysis_test else '❌ فشل'}")
    print(f"  • اختبار توليد الإشارات: {'✅ نجح' if signal_test else '❌ فشل'}")
    
    if api_test and analysis_test and signal_test:
        print("\n🎉 جميع الاختبارات نجحت! التطبيق جاهز للاستخدام.")
        print("\n📝 ملاحظات مهمة:")
        print("  • هذا الاختبار يستخدم بيانات وهمية")
        print("  • للاستخدام الحقيقي، تحتاج إلى SSID صحيح من PocketOption")
        print("  • التطبيق الرئيسي يتطلب tkinter (متوفر في Windows)")
        print("  • تأكد من تثبيت جميع المتطلبات قبل التشغيل")
        return True
    else:
        print("\n❌ بعض الاختبارات فشلت. يرجى مراجعة الأخطاء أعلاه.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 خطأ غير متوقع: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

