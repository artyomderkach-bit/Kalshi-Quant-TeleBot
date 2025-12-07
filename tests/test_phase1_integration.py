#!/usr/bin/env python3
"""
Phase 1 Integration Tests
Tests the complete Phase 1 system integration
"""

import unittest
import sys
import os
import subprocess
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bot_state import fetch_balance, fetch_positions, fetch_status, fetch_performance
from kalshi_api import KalshiAPI
import config

class TestPhase1Integration(unittest.TestCase):
    """Test Phase 1 system integration"""

    def setUp(self):
        """Set up test environment"""
        self.mock_api = Mock(spec=KalshiAPI)

    @patch('kalshi_api.KalshiAPI')
    def test_bot_state_balance_fetching(self, mock_api_class):
        """Test balance fetching integration"""
        # Mock API responses
        mock_api_instance = Mock()
        mock_api_class.return_value = mock_api_instance
        mock_api_instance.get_account_balance.return_value = {
            'available_balance': 50000,  # cents
            'portfolio_value': 100000,  # cents
            'unrealized_pnl': 15000,    # cents
            'realized_pnl': 5000,       # cents
        }

        result = fetch_balance(mock_api_instance)

        self.assertIsInstance(result, dict)
        self.assertIn('summary', result)
        self.assertIn('raw', result)
        self.assertEqual(result['summary']['available'], 500.0)  # $500.00
        self.assertEqual(result['summary']['total_equity'], 1000.0)  # $1000.00

    @patch('kalshi_api.KalshiAPI')
    def test_bot_state_positions_fetching(self, mock_api_class):
        """Test positions fetching integration"""
        mock_api_instance = Mock()
        mock_api_class.return_value = mock_api_instance
        mock_api_instance.get_positions.return_value = {
            'positions': [
                {
                    'market_id': 'test_market_1',
                    'quantity': 10,
                    'entry_price': 5000,  # cents
                    'current_price': 5500,  # cents
                },
                {
                    'market_id': 'test_market_2',
                    'quantity': -5,
                    'entry_price': 2000,  # cents
                    'current_price': 1800,  # cents
                }
            ]
        }

        result = fetch_positions(mock_api_instance)

        self.assertIsInstance(result, dict)
        self.assertIn('positions', result)
        self.assertIn('count', result)
        self.assertEqual(result['count'], 2)
        self.assertEqual(len(result['positions']), 2)

    @patch('kalshi_api.KalshiAPI')
    def test_bot_state_performance_fetching(self, mock_api_class):
        """Test performance metrics integration"""
        mock_api_instance = Mock()
        mock_api_class.return_value = mock_api_instance
        mock_api_instance.get_orders.return_value = {
            'orders': [
                {'count': 10, 'avg_price': 5000, 'yes_price': None},
                {'count': 5, 'avg_price': None, 'yes_price': 5500},
                {'count': 0, 'avg_price': None, 'yes_price': None},  # No count
            ]
        }

        result = fetch_performance(mock_api_instance)

        self.assertIsInstance(result, dict)
        self.assertIn('totalTrades', result)
        self.assertIn('totalContracts', result)
        self.assertEqual(result['totalTrades'], 3)
        self.assertEqual(result['totalContracts'], 15)  # 10 + 5

    @patch('kalshi_api.KalshiAPI')
    def test_bot_state_status_integration(self, mock_api_class):
        """Test complete status integration"""
        mock_api_instance = Mock()
        mock_api_class.return_value = mock_api_instance

        # Mock all API calls
        mock_api_instance.get_exchange_status.return_value = {'status': 'operational'}
        mock_api_instance.get_account_balance.return_value = {'available_balance': 100000}
        mock_api_instance.get_positions.return_value = {'positions': []}

        result = fetch_status(mock_api_instance)

        self.assertIsInstance(result, dict)
        self.assertIn('exchange_status', result)
        self.assertIn('balance_summary', result)
        self.assertIn('positions_count', result)
        self.assertIn('active_strategies', result)
        self.assertIn('timestamp', result)

        # Check that Phase 1 strategies are listed
        self.assertIn('news_sentiment', result['active_strategies'])
        self.assertIn('statistical_arbitrage', result['active_strategies'])
        self.assertIn('volatility_based', result['active_strategies'])

    def test_config_phase1_variables(self):
        """Test that all Phase 1 configuration variables are available"""
        phase1_vars = [
            'NEWS_API_KEY',
            'NEWS_API_BASE_URL',
            'NEWS_SENTIMENT_THRESHOLD',
            'STAT_ARBITRAGE_THRESHOLD',
            'VOLATILITY_THRESHOLD'
        ]

        for var in phase1_vars:
            self.assertTrue(hasattr(config, var), f"Missing Phase 1 config variable: {var}")

    def test_dependencies_availability(self):
        """Test that Phase 1 dependencies are importable"""
        try:
            import newsapi
            import transformers
            import torch
            import nltk
            import textblob
            from statsmodels.tsa.stattools import coint
            from arch import arch_model
            import sklearn
        except ImportError as e:
            self.fail(f"Phase 1 dependency not available: {e}")

    def test_strategy_module_imports(self):
        """Test that all Phase 1 strategy modules can be imported"""
        try:
            from news_analyzer import NewsSentimentAnalyzer
            from arbitrage_analyzer import StatisticalArbitrageAnalyzer
            from volatility_analyzer import VolatilityAnalyzer
        except ImportError as e:
            self.fail(f"Phase 1 strategy module import failed: {e}")

        # Test instantiation
        news_analyzer = NewsSentimentAnalyzer()
        arb_analyzer = StatisticalArbitrageAnalyzer()
        vol_analyzer = VolatilityAnalyzer()

        self.assertIsInstance(news_analyzer, NewsSentimentAnalyzer)
        self.assertIsInstance(arb_analyzer, StatisticalArbitrageAnalyzer)
        self.assertIsInstance(vol_analyzer, VolatilityAnalyzer)


class TestPhase1CLI(unittest.TestCase):
    """Test Phase 1 CLI functionality"""

    def test_bot_state_cli_balance(self):
        """Test bot_state.py CLI for balance command"""
        # This would require running the actual CLI, but we'll mock it
        from bot_state import run

        mock_api = Mock()
        mock_api.get_account_balance.return_value = {'available_balance': 100000}

        with patch('kalshi_api.KalshiAPI', return_value=mock_api):
            result = run('balance')

            self.assertIsInstance(result, dict)
            self.assertIn('summary', result)
            self.assertEqual(result['summary']['available'], 1000.0)

    def test_bot_state_cli_positions(self):
        """Test bot_state.py CLI for positions command"""
        from bot_state import run

        mock_api = Mock()
        mock_api.get_positions.return_value = {'positions': []}

        with patch('kalshi_api.KalshiAPI', return_value=mock_api):
            result = run('positions')

            self.assertIsInstance(result, dict)
            self.assertIn('positions', result)
            self.assertIn('count', result)
            self.assertEqual(result['count'], 0)

    def test_bot_state_cli_performance(self):
        """Test bot_state.py CLI for performance command"""
        from bot_state import run

        mock_api = Mock()
        mock_api.get_orders.return_value = {'orders': []}

        with patch('kalshi_api.KalshiAPI', return_value=mock_api):
            result = run('performance')

            self.assertIsInstance(result, dict)
            self.assertIn('totalTrades', result)
            self.assertEqual(result['totalTrades'], 0)


class TestPhase1EndToEnd(unittest.TestCase):
    """End-to-end tests for Phase 1 functionality"""

    def test_complete_strategy_orchestration(self):
        """Test the complete strategy orchestration flow"""
        from trader import Trader

        # Create mocks
        mock_api = Mock()
        mock_notifier = Mock()
        mock_logger = Mock()

        # Mock market data
        market_data = {
            'markets': [
                {'id': 'market1', 'current_price': 1.0},
                {'id': 'market2', 'current_price': 1.1}
            ]
        }

        trader = Trader(mock_api, mock_notifier, mock_logger, 1000)

        # Mock all strategies to return no signals (normal case)
        with patch.object(trader.news_analyzer, 'get_market_relevant_news') as mock_news:
            with patch.object(trader.news_analyzer, 'should_trade_based_on_sentiment') as mock_sentiment:
                with patch.object(trader, '_statistical_arbitrage') as mock_arb:
                    with patch.object(trader, '_volatility_analysis') as mock_vol:

                        mock_news.return_value = {'overall_sentiment': 0.1, 'confidence': 0.3}
                        mock_sentiment.return_value = {'should_trade': False}
                        mock_arb.return_value = []
                        mock_vol.return_value = None

                        decision = trader._make_trade_decision(market_data)

                        # Should return None when no opportunities found
                        self.assertIsNone(decision)

    def test_strategy_error_handling(self):
        """Test error handling in strategy orchestration"""
        from trader import Trader

        mock_api = Mock()
        mock_notifier = Mock()
        mock_logger = Mock()

        trader = Trader(mock_api, mock_notifier, mock_logger, 1000)

        market_data = {'markets': [{'id': 'test', 'current_price': 1.0}]}

        # Mock news analyzer to raise an exception
        with patch.object(trader.news_analyzer, 'get_market_relevant_news') as mock_news:
            with patch.object(trader.news_analyzer, 'should_trade_based_on_sentiment') as mock_sentiment:
                mock_news.side_effect = Exception("API Error")
                mock_sentiment.return_value = {'should_trade': False}

                # Should not crash, should continue to other strategies
                decision = trader._make_trade_decision(market_data)

                # Should log error but continue
                mock_logger.error.assert_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
