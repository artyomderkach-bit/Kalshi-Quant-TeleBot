#!/usr/bin/env python3
"""
Phase 1 Trading Strategies Test Suite
Tests all three core quantitative trading strategies implemented in Phase 1
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from news_analyzer import NewsSentimentAnalyzer
from arbitrage_analyzer import StatisticalArbitrageAnalyzer
from volatility_analyzer import VolatilityAnalyzer
from trader import Trader
import config

class TestNewsSentimentAnalyzer(unittest.TestCase):
    """Test News Sentiment Analysis Strategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = NewsSentimentAnalyzer()
        self.sample_articles = [
            {
                'title': 'Stock Market Rises on Positive Economic Data',
                'description': 'Markets surged today following strong economic indicators.',
                'content': 'The stock market experienced significant gains today as investors reacted positively to the latest economic data.'
            },
            {
                'title': 'Economic Downturn Causes Market Panic',
                'description': 'Investors are worried about the deteriorating economic conditions.',
                'content': 'Market sentiment turned negative as economic indicators showed concerning trends.'
            },
            {
                'title': 'Weather Update: Normal Conditions Expected',
                'description': 'Meteorological forecast indicates standard weather patterns.',
                'content': 'Weather conditions are expected to remain within normal parameters for the coming days.'
            }
        ]

    def test_sentiment_analysis(self):
        """Test basic sentiment analysis functionality"""
        text = "The market is performing extremely well today"
        sentiment = self.analyzer.analyze_sentiment(text)

        self.assertIsInstance(sentiment, dict)
        self.assertIn('polarity', sentiment)
        self.assertIn('subjectivity', sentiment)
        self.assertGreater(sentiment['polarity'], 0)  # Should be positive

    def test_text_preprocessing(self):
        """Test text preprocessing functionality"""
        raw_text = "Check this link: https://example.com/article @user #hashtag!"
        processed = self.analyzer.preprocess_text(raw_text)

        self.assertNotIn('https://', processed)
        self.assertNotIn('@', processed)
        self.assertNotIn('#', processed)
        self.assertEqual(processed, "Check this link article user hashtag")

    def test_news_sentiment_aggregation(self):
        """Test aggregation of sentiment across multiple articles"""
        result = self.analyzer.analyze_news_sentiment(self.sample_articles)

        self.assertIsInstance(result, dict)
        self.assertIn('overall_sentiment', result)
        self.assertIn('confidence', result)
        self.assertIn('article_count', result)
        self.assertEqual(result['article_count'], 3)

    def test_sentiment_trading_decision(self):
        """Test trading decision based on sentiment analysis"""
        # Test positive sentiment
        positive_sentiment = {
            'overall_sentiment': 0.8,
            'confidence': 0.9
        }
        decision = self.analyzer.should_trade_based_on_sentiment(positive_sentiment)
        self.assertTrue(decision['should_trade'])
        self.assertEqual(decision['direction'], 'long')

        # Test negative sentiment
        negative_sentiment = {
            'overall_sentiment': -0.8,
            'confidence': 0.9
        }
        decision = self.analyzer.should_trade_based_on_sentiment(negative_sentiment)
        self.assertTrue(decision['should_trade'])
        self.assertEqual(decision['direction'], 'short')

        # Test low confidence
        low_confidence = {
            'overall_sentiment': 0.9,
            'confidence': 0.2
        }
        decision = self.analyzer.should_trade_based_on_sentiment(low_confidence)
        self.assertFalse(decision['should_trade'])

    @patch('news_analyzer.requests.Session.get')
    def test_news_fetching(self, mock_get):
        """Test news fetching with mocked API response"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'articles': self.sample_articles[:1]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        articles = self.analyzer.fetch_news('test query')
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], 'Stock Market Rises on Positive Economic Data')


class TestStatisticalArbitrageAnalyzer(unittest.TestCase):
    """Test Statistical Arbitrage Strategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = StatisticalArbitrageAnalyzer()

        # Create synthetic price series with known cointegration
        np.random.seed(42)
        self.price_series_1 = [1.0 + 0.01 * i + 0.1 * np.sin(i/10) for i in range(100)]
        self.price_series_2 = [1.2 + 0.01 * i + 0.1 * np.sin(i/10) + 0.05 * np.random.randn() for i in range(100)]

    def test_cointegration_test(self):
        """Test cointegration testing functionality"""
        result = self.analyzer.test_cointegration(self.price_series_1, self.price_series_2)

        self.assertIsInstance(result, dict)
        self.assertIn('cointegrated', result)
        self.assertIn('p_value', result)
        self.assertIn('confidence', result)

    def test_spread_calculation(self):
        """Test spread calculation and z-score analysis"""
        result = self.analyzer.calculate_spread(self.price_series_1, self.price_series_2)

        self.assertIsInstance(result, dict)
        self.assertIn('z_score', result)
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertIn('spread', result)

    def test_market_pair_analysis(self):
        """Test complete market pair analysis"""
        market1 = {
            'id': 'market1',
            'title': 'Test Market 1',
            'current_price': 1.0,
            'price_history': self.price_series_1
        }
        market2 = {
            'id': 'market2',
            'title': 'Test Market 2',
            'current_price': 1.2,
            'price_history': self.price_series_2
        }

        result = self.analyzer.analyze_market_pair(market1, market2)

        self.assertIsInstance(result, dict)
        self.assertIn('arbitrage_opportunity', result)
        self.assertIn('signal', result)
        self.assertIn('confidence', result)
        self.assertIn('z_score', result)

    def test_arbitrage_opportunities_scanning(self):
        """Test scanning multiple markets for arbitrage opportunities"""
        markets = [
            {
                'id': f'market_{i}',
                'title': f'Test Market {i}',
                'current_price': 1.0 + i * 0.1,
                'price_history': [1.0 + i * 0.1 + 0.01 * j for j in range(100)]
            }
            for i in range(3)
        ]

        opportunities = self.analyzer.find_arbitrage_opportunities(markets)
        self.assertIsInstance(opportunities, list)

    def test_arbitrage_execution_decision(self):
        """Test arbitrage execution decision making"""
        arbitrage_analysis = {
            'signal': 'LONG_SPREAD',
            'confidence': 0.8,
            'z_score': 2.5,
            'market1': {'id': 'm1', 'current_price': 1.0},
            'market2': {'id': 'm2', 'current_price': 1.2}
        }

        decision = self.analyzer.should_execute_arbitrage(arbitrage_analysis)

        self.assertIsInstance(decision, dict)
        self.assertIn('should_execute', decision)
        self.assertIn('confidence', decision)


class TestVolatilityAnalyzer(unittest.TestCase):
    """Test Volatility-Based Trading Strategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = VolatilityAnalyzer()

        # Create synthetic price series with known volatility patterns
        np.random.seed(42)
        self.high_vol_prices = [1.0 + 0.05 * np.sin(i/5) + 0.1 * np.random.randn() for i in range(100)]
        self.low_vol_prices = [1.0 + 0.01 * i + 0.02 * np.random.randn() for i in range(100)]

    def test_historical_volatility_calculation(self):
        """Test historical volatility calculation"""
        result = self.analyzer.calculate_historical_volatility(self.high_vol_prices)

        self.assertIsInstance(result, dict)
        self.assertIn('historical_volatility', result)
        self.assertIn('realized_volatility', result)
        self.assertGreater(result['historical_volatility'], 0)

    def test_volatility_regime_analysis(self):
        """Test volatility regime classification"""
        # Test with high volatility history
        regime = self.analyzer.analyze_volatility_regime(0.8, [0.5, 0.6, 0.7, 0.9, 0.8])

        self.assertIsInstance(regime, dict)
        self.assertIn('regime', regime)
        self.assertIn('confidence', regime)
        self.assertIn('percentile', regime)

    def test_garch_model_fitting(self):
        """Test GARCH model fitting"""
        returns = np.diff(np.log(np.array(self.high_vol_prices, dtype=float)))
        result = self.analyzer.fit_garch_model(returns.tolist())

        self.assertIsInstance(result, dict)
        # Note: GARCH fitting might fail with synthetic data, but structure should be correct
        self.assertIn('conditional_volatility', result)

    def test_volatility_signals(self):
        """Test volatility-based trading signal generation"""
        signals = self.analyzer.detect_volatility_signals(0.8, [0.5, 0.6, 0.7, 0.9])

        self.assertIsInstance(signals, dict)
        self.assertIn('volatility_signal', signals)
        self.assertIn('direction', signals)
        self.assertIn('confidence', signals)

    def test_market_volatility_analysis(self):
        """Test complete market volatility analysis"""
        market_data = {
            'id': 'test_market',
            'title': 'Test Market',
            'current_price': 1.0,
            'price_history': self.high_vol_prices
        }

        result = self.analyzer.analyze_market_volatility(market_data)

        self.assertIsInstance(result, dict)
        self.assertIn('market_id', result)
        self.assertIn('volatility_analysis', result)
        self.assertIn('signal_analysis', result)

    def test_volatility_trading_decision(self):
        """Test volatility-based trading decision"""
        volatility_analysis = {
            'signal_analysis': {
                'volatility_signal': 'MEAN_REVERSION_SHORT',
                'direction': 'short',
                'confidence': 0.8
            }
        }

        decision = self.analyzer.should_trade_based_on_volatility(volatility_analysis)

        self.assertIsInstance(decision, dict)
        self.assertIn('should_trade', decision)
        self.assertIn('direction', decision)


class TestMultiStrategyIntegration(unittest.TestCase):
    """Test integration of all Phase 1 strategies"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock API, notifier, logger for trader testing
        self.mock_api = Mock()
        self.mock_notifier = Mock()
        self.mock_logger = Mock()

        # Create trader instance
        self.trader = Trader(self.mock_api, self.mock_notifier, self.mock_logger, 1000)

    def test_trader_initialization(self):
        """Test that trader initializes with all analyzers"""
        self.assertIsNotNone(self.trader.news_analyzer)
        self.assertIsNotNone(self.trader.arbitrage_analyzer)
        self.assertIsNotNone(self.trader.volatility_analyzer)

    def test_news_sentiment_strategy(self):
        """Test news sentiment strategy in trader"""
        # Mock market data
        market_data = {
            'markets': [{'id': 'test_market', 'current_price': 1.0}]
        }

        # Mock the news analyzer to return a trading signal
        with patch.object(self.trader.news_analyzer, 'get_market_relevant_news') as mock_news:
            with patch.object(self.trader.news_analyzer, 'should_trade_based_on_sentiment') as mock_decision:
                mock_news.return_value = {'overall_sentiment': 0.8, 'confidence': 0.9}
                mock_decision.return_value = {
                    'should_trade': True,
                    'direction': 'long',
                    'sentiment_score': 0.8,
                    'confidence': 0.9
                }

                decision = self.trader._make_trade_decision(market_data)

                self.assertIsNotNone(decision)
                self.assertEqual(decision['strategy'], 'news_sentiment')

    def test_arbitrage_strategy(self):
        """Test statistical arbitrage strategy in trader"""
        # Mock market data with multiple markets
        market_data = {
            'markets': [
                {'id': 'market1', 'current_price': 1.0},
                {'id': 'market2', 'current_price': 1.1}
            ]
        }

        # Mock arbitrage analyzer
        with patch.object(self.trader.arbitrage_analyzer, 'find_arbitrage_opportunities') as mock_find:
            with patch.object(self.trader.arbitrage_analyzer, 'should_execute_arbitrage') as mock_execute:
                mock_find.return_value = [{
                    'signal': 'LONG_SPREAD',
                    'confidence': 0.8,
                    'z_score': 2.0,
                    'market1': {'id': 'market1', 'current_price': 1.0},
                    'market2': {'id': 'market2', 'current_price': 1.1}
                }]
                mock_execute.return_value = {
                    'should_execute': True,
                    'confidence': 0.8,
                    'position_size': 0.5,
                    'market1': {'id': 'market1', 'current_price': 1.0},
                    'market2': {'id': 'market2', 'current_price': 1.1}
                }

                decision = self.trader._make_trade_decision(market_data)

                self.assertIsNotNone(decision)
                self.assertEqual(decision['strategy'], 'statistical_arbitrage')

    def test_volatility_strategy(self):
        """Test volatility strategy in trader"""
        market_data = {
            'markets': [{'id': 'test_market', 'current_price': 1.0}]
        }

        with patch.object(self.trader, '_volatility_analysis') as mock_vol:
            mock_vol.return_value = {
                'should_trade': True,
                'direction': 'short',
                'confidence': 0.8,
                'market_data': {'id': 'test_market', 'current_price': 1.0}
            }

            decision = self.trader._make_trade_decision(market_data)

            self.assertIsNotNone(decision)
            self.assertEqual(decision['strategy'], 'volatility_based')

    def test_strategy_priority(self):
        """Test that strategies execute in correct priority order"""
        market_data = {
            'markets': [{'id': 'test_market', 'current_price': 1.0}]
        }

        # Mock news strategy to return a signal (should be chosen first)
        with patch.object(self.trader.news_analyzer, 'get_market_relevant_news') as mock_news:
            with patch.object(self.trader.news_analyzer, 'should_trade_based_on_sentiment') as mock_decision:
                mock_news.return_value = {'overall_sentiment': 0.8, 'confidence': 0.9}
                mock_decision.return_value = {
                    'should_trade': True,
                    'direction': 'long',
                    'sentiment_score': 0.8,
                    'confidence': 0.9
                }

                decision = self.trader._make_trade_decision(market_data)

                self.assertEqual(decision['strategy'], 'news_sentiment')

    def test_no_opportunities_fallback(self):
        """Test behavior when no trading opportunities are found"""
        market_data = {
            'markets': [{'id': 'test_market', 'current_price': 1.0}]
        }

        # Mock all strategies to return no signals
        with patch.object(self.trader.news_analyzer, 'get_market_relevant_news') as mock_news:
            with patch.object(self.trader.news_analyzer, 'should_trade_based_on_sentiment') as mock_decision:
                with patch.object(self.trader, '_statistical_arbitrage') as mock_arb:
                    with patch.object(self.trader, '_volatility_analysis') as mock_vol:
                        mock_news.return_value = {'overall_sentiment': 0.1, 'confidence': 0.3}
                        mock_decision.return_value = {'should_trade': False}
                        mock_arb.return_value = []
                        mock_vol.return_value = None

                        decision = self.trader._make_trade_decision(market_data)

                        self.assertIsNone(decision)


if __name__ == '__main__':
    # Add verbose output
    unittest.main(verbosity=2)
