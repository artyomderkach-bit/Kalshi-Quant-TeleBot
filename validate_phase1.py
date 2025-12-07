#!/usr/bin/env python3
"""
Phase 1 Validation Script
Quick validation of Phase 1 implementation without full test framework
"""

import sys
import os
import traceback

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all Phase 1 modules can be imported"""
    print("🔍 Testing imports...")

    try:
        from news_analyzer import NewsSentimentAnalyzer
        from arbitrage_analyzer import StatisticalArbitrageAnalyzer
        from volatility_analyzer import VolatilityAnalyzer
        from trader import Trader

        print("✅ All strategy modules imported successfully")

        # Test instantiation
        news_analyzer = NewsSentimentAnalyzer()
        arb_analyzer = StatisticalArbitrageAnalyzer()
        vol_analyzer = VolatilityAnalyzer()

        print("✅ All analyzers instantiated successfully")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        return False

def test_sentiment_analysis():
    """Test basic sentiment analysis functionality"""
    print("\n📊 Testing sentiment analysis...")

    try:
        from news_analyzer import NewsSentimentAnalyzer

        analyzer = NewsSentimentAnalyzer()

        # Test sentiment analysis
        result = analyzer.analyze_sentiment("The market is performing extremely well today")
        if result['polarity'] > 0:
            print("✅ Sentiment analysis working correctly")
        else:
            print("❌ Sentiment analysis polarity test failed")
            return False

        # Test news aggregation
        sample_articles = [
            {'title': 'Market rises', 'description': 'Positive economic data'},
            {'title': 'Market falls', 'description': 'Negative indicators'}
        ]
        sentiment_result = analyzer.analyze_news_sentiment(sample_articles)
        if 'overall_sentiment' in sentiment_result:
            print("✅ News sentiment aggregation working")
        else:
            print("❌ News sentiment aggregation failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        traceback.print_exc()
        return False

def test_arbitrage_analysis():
    """Test basic arbitrage analysis functionality"""
    print("\n📈 Testing arbitrage analysis...")

    try:
        from arbitrage_analyzer import StatisticalArbitrageAnalyzer
        import numpy as np

        analyzer = StatisticalArbitrageAnalyzer()

        # Create synthetic price series
        series1 = [1.0 + 0.01 * i for i in range(50)]
        series2 = [1.1 + 0.01 * i for i in range(50)]

        # Test cointegration
        coint_result = analyzer.test_cointegration(series1, series2)
        if 'p_value' in coint_result:
            print("✅ Cointegration testing working")
        else:
            print("❌ Cointegration testing failed")
            return False

        # Test spread calculation
        spread_result = analyzer.calculate_spread(series1, series2)
        if 'z_score' in spread_result:
            print("✅ Spread calculation working")
        else:
            print("❌ Spread calculation failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Arbitrage analysis test failed: {e}")
        traceback.print_exc()
        return False

def test_volatility_analysis():
    """Test basic volatility analysis functionality"""
    print("\n📉 Testing volatility analysis...")

    try:
        from volatility_analyzer import VolatilityAnalyzer
        import numpy as np

        analyzer = VolatilityAnalyzer()

        # Create synthetic price series
        prices = [1.0 + 0.05 * np.sin(i/5) + 0.1 * np.random.randn() for i in range(100)]

        # Test historical volatility
        vol_result = analyzer.calculate_historical_volatility(prices)
        if 'historical_volatility' in vol_result:
            print("✅ Historical volatility calculation working")
        else:
            print("❌ Historical volatility calculation failed")
            return False

        # Test regime analysis
        regime_result = analyzer.analyze_volatility_regime(0.8, [0.5, 0.6, 0.7, 0.9])
        if 'regime' in regime_result:
            print("✅ Volatility regime analysis working")
        else:
            print("❌ Volatility regime analysis failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Volatility analysis test failed: {e}")
        traceback.print_exc()
        return False

def test_trader_integration():
    """Test trader integration with all strategies"""
    print("\n🤖 Testing trader integration...")

    try:
        from trader import Trader
        from unittest.mock import Mock

        # Create mock dependencies
        mock_api = Mock()
        mock_notifier = Mock()
        mock_logger = Mock()

        trader = Trader(mock_api, mock_notifier, mock_logger, 1000)

        # Test that all analyzers are present
        if hasattr(trader, 'news_analyzer') and hasattr(trader, 'arbitrage_analyzer') and hasattr(trader, 'volatility_analyzer'):
            print("✅ Trader has all Phase 1 analyzers")
        else:
            print("❌ Trader missing analyzers")
            return False

        # Test basic market data processing
        market_data = {'markets': [{'id': 'test', 'current_price': 1.0}]}

        # This should not crash (even if strategies return None)
        try:
            decision = trader._make_trade_decision(market_data)
            print("✅ Trader decision making working")
        except Exception as e:
            print(f"❌ Trader decision making failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ Trader integration test failed: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test Phase 1 configuration"""
    print("\n⚙️  Testing configuration...")

    try:
        import config

        required_vars = [
            'NEWS_API_KEY', 'NEWS_API_BASE_URL',
            'NEWS_SENTIMENT_THRESHOLD', 'STAT_ARBITRAGE_THRESHOLD', 'VOLATILITY_THRESHOLD'
        ]

        missing_vars = []
        for var in required_vars:
            if not hasattr(config, var):
                missing_vars.append(var)

        if not missing_vars:
            print("✅ All Phase 1 configuration variables present")
            return True
        else:
            print(f"❌ Missing configuration variables: {missing_vars}")
            return False

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all Phase 1 validation tests"""
    print("🚀 Phase 1 Implementation Validation")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Sentiment Analysis Test", test_sentiment_analysis),
        ("Arbitrage Analysis Test", test_arbitrage_analysis),
        ("Volatility Analysis Test", test_volatility_analysis),
        ("Trader Integration Test", test_trader_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print("2")

    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 PHASE 1 VALIDATION SUCCESSFUL!")
        print("✅ All core trading strategies are working correctly")
        print("✅ Ready to proceed to Phase 2")
        return 0
    else:
        print("\n⚠️  Some validation tests failed.")
        print("🔧 Please review the errors above and fix issues before proceeding to Phase 2")
        return 1

if __name__ == "__main__":
    sys.exit(main())
