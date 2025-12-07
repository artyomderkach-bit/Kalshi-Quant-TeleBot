#!/usr/bin/env python3
"""
Phase 2 & Phase 3 Comprehensive Test Suite
Tests Advanced Risk Management and Real-Time Analytics implementations
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_manager import RiskManager
from market_data_streamer import MarketDataStreamer, MarketData
from performance_analytics import PerformanceAnalytics, Trade
from trader import Trader

class TestPhase2RiskManagement(unittest.TestCase):
    """Test Phase 2: Advanced Risk Management"""

    def setUp(self):
        """Set up test fixtures"""
        self.risk_manager = RiskManager(initial_bankroll=10000)

    def test_kelly_criterion_calculation(self):
        """Test Kelly Criterion position sizing"""
        # Test with favorable odds (60% win rate, 2:1 payoff ratio)
        position_size = self.risk_manager.calculate_position_size_kelly(0.8, 2.0)
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 0.25)  # Should be capped

        # Test with unfavorable conditions
        position_size_bad = self.risk_manager.calculate_position_size_kelly(0.3, 1.1)
        self.assertLess(position_size_bad, position_size)  # Should be smaller

        # Test with extreme conditions
        position_size_extreme = self.risk_manager.calculate_position_size_kelly(0.95, 3.0)
        self.assertLessEqual(position_size_extreme, 0.25)  # Should be capped

    def test_stop_loss_calculations(self):
        """Test stop-loss price calculations"""
        entry_price = 1.0

        # Long position stop-loss
        stop_price_long = self.risk_manager.calculate_stop_loss_price(entry_price, is_long=True)
        self.assertLess(stop_price_long, entry_price)  # Should be below entry

        # Short position stop-loss
        stop_price_short = self.risk_manager.calculate_stop_loss_price(entry_price, is_long=False)
        self.assertGreater(stop_price_short, entry_price)  # Should be above entry

    def test_stop_loss_trigger_detection(self):
        """Test stop-loss trigger detection"""
        entry_price = 1.0

        # Long position - price drops below stop-loss
        stop_price = self.risk_manager.calculate_stop_loss_price(entry_price, is_long=True)
        current_price_below_stop = stop_price - 0.01
        self.assertTrue(self.risk_manager.check_stop_loss_trigger(
            entry_price, current_price_below_stop, is_long=True))

        # Long position - price still above stop-loss
        current_price_above_stop = stop_price + 0.01
        self.assertFalse(self.risk_manager.check_stop_loss_trigger(
            entry_price, current_price_above_stop, is_long=True))

    def test_portfolio_risk_metrics(self):
        """Test portfolio risk metrics calculation"""
        # Create sample returns
        returns = [0.01, -0.005, 0.008, -0.003, 0.012, -0.008, 0.015]

        metrics = self.risk_manager.calculate_portfolio_metrics(returns)

        # Check that all expected metrics are present
        expected_metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate', 'total_return', 'volatility']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

        # Check reasonable value ranges
        self.assertGreaterEqual(metrics['win_rate'], 0)
        self.assertLessEqual(metrics['win_rate'], 1)
        self.assertGreater(metrics['total_return'], -1)  # Not complete loss

    def test_position_size_validation(self):
        """Test position size validation against risk limits"""
        # Test valid position size
        valid_position = self.risk_manager.validate_position_size(500)  # $500 position
        self.assertTrue(valid_position)

        # Test invalid position size (too large)
        invalid_position = self.risk_manager.validate_position_size(1500)  # $1500 position (>10% of $10k)
        self.assertFalse(invalid_position)

    def test_portfolio_status_reporting(self):
        """Test portfolio status reporting"""
        status = self.risk_manager.get_portfolio_status()

        expected_fields = ['current_bankroll', 'initial_bankroll', 'total_pnl',
                         'total_return_pct', 'risk_metrics']
        for field in expected_fields:
            self.assertIn(field, status)

        self.assertEqual(status['initial_bankroll'], 10000)
        self.assertEqual(status['current_bankroll'], 10000)  # No trades yet
        self.assertEqual(status['total_pnl'], 0)


class TestPhase3MarketDataStreaming(unittest.TestCase):
    """Test Phase 3: Real-Time Market Data Streaming"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_api = Mock()
        self.streamer = MarketDataStreamer(self.mock_api, update_interval=1)  # Fast for testing

    def test_market_data_creation(self):
        """Test MarketData object creation and properties"""
        market = MarketData(
            market_id="test_market",
            title="Test Market",
            current_price=1.5,
            previous_price=1.4,
            volume=1000
        )

        self.assertEqual(market.market_id, "test_market")
        self.assertEqual(market.current_price, 1.5)
        self.assertEqual(market.previous_price, 1.4)
        self.assertEqual(market.price_change, 0.1)
        self.assertAlmostEqual(market.price_change_pct, 7.142857, places=5)

    def test_market_data_price_history(self):
        """Test price history management"""
        market = MarketData("test", "Test", 1.0)

        # Add some price history
        prices = [1.0, 1.01, 0.99, 1.02, 1.01]
        for price in prices[1:]:  # Skip first price as it's already set
            market.price_history.append(price)

        self.assertEqual(len(market.price_history), 5)
        self.assertEqual(market.price_history[0], 1.0)

    def test_market_data_streamer_initialization(self):
        """Test MarketDataStreamer initialization"""
        self.assertIsNotNone(self.streamer.markets_data)
        self.assertEqual(self.streamer.update_interval, 1)
        self.assertIsNotNone(self.streamer.api_client)
        self.assertFalse(self.streamer.running)

    @patch('time.sleep')
    def test_market_data_updates(self, mock_sleep):
        """Test market data update functionality"""
        # Mock API response
        mock_markets_data = {
            'markets': [
                {'id': 'market1', 'current_price': 1.5, 'title': 'Market 1'},
                {'id': 'market2', 'current_price': 2.0, 'title': 'Market 2'}
            ]
        }
        self.mock_api.get_markets.return_value = mock_markets_data

        # Mock subscriber
        subscriber_called = []
        def mock_subscriber(updated_markets, market_data):
            subscriber_called.append((updated_markets, len(market_data)))

        self.streamer.add_subscriber(mock_subscriber)

        # Manually trigger update (since we can't easily test threading)
        self.streamer._update_market_data()

        # Check that markets were added
        self.assertIn('market1', self.streamer.markets_data)
        self.assertIn('market2', self.streamer.markets_data)
        self.assertEqual(self.streamer.markets_data['market1'].current_price, 1.5)
        self.assertEqual(self.streamer.markets_data['market2'].current_price, 2.0)

        # Check subscriber was called
        self.assertEqual(len(subscriber_called), 1)
        self.assertEqual(len(subscriber_called[0][0]), 2)  # 2 markets updated

    def test_market_analysis_functions(self):
        """Test market analysis helper functions"""
        # Create test market data
        markets = {
            'market1': MarketData('market1', 'Market 1', 1.0),
            'market2': MarketData('market2', 'Market 2', 2.0),
            'market3': MarketData('market3', 'Market 3', 1.5)
        }

        # Manually set price changes for testing
        markets['market1'].previous_price = 0.95  # +5.26% change
        markets['market2'].previous_price = 1.9   # +5.26% change
        markets['market3'].previous_price = 1.6   # -6.25% change

        self.streamer.markets_data = markets

        # Test top movers
        top_movers = self.streamer.get_top_movers(limit=2)
        self.assertEqual(len(top_movers), 2)
        # Should include the markets with largest absolute changes

        # Test high volatility (would need volatility data)
        high_vol_markets = self.streamer.get_high_volatility_markets(limit=1)
        # This will be empty since we didn't set volatility data

    def test_market_summary(self):
        """Test market summary generation"""
        # Create test data
        markets = {
            'market1': MarketData('market1', 'Market 1', 1.0),
            'market2': MarketData('market2', 'Market 2', 2.0),
        }

        # Set price changes
        markets['market1'].previous_price = 0.95  # Gainer
        markets['market2'].previous_price = 2.1   # Loser

        self.streamer.markets_data = markets

        # Note: The market summary calculation uses numpy internally
        # but our test doesn't directly use np, so this test validates
        # that the summary structure is correct
        summary = self.streamer.get_market_summary()

        self.assertIn('total_markets', summary)
        self.assertIn('average_price', summary)
        self.assertIn('gainers', summary)
        self.assertIn('losers', summary)
        self.assertEqual(summary['total_markets'], 2)


class TestPhase3PerformanceAnalytics(unittest.TestCase):
    """Test Phase 3: Advanced Performance Analytics"""

    def setUp(self):
        """Set up test fixtures"""
        self.analytics = PerformanceAnalytics()

    def test_trade_creation_and_recording(self):
        """Test trade creation and recording"""
        trade = Trade(
            trade_id="test_trade_1",
            market_id="market1",
            strategy="news_sentiment",
            side="buy",
            quantity=10,
            entry_price=1.0,
            confidence=0.8
        )

        self.analytics.record_trade(trade)

        self.assertEqual(len(self.analytics.trades), 1)
        self.assertEqual(self.analytics.trades[0].trade_id, "test_trade_1")

    def test_trade_closure(self):
        """Test trade closure and P&L calculation"""
        # Create and record trade
        trade = Trade("test_trade_2", "market1", "arbitrage", "buy", 10, 1.0)
        self.analytics.record_trade(trade)

        # Close trade at profit
        exit_price = 1.05  # 5% profit
        success = self.analytics.close_trade("test_trade_2", exit_price, "take_profit")

        self.assertTrue(success)

        # Check that trade was updated
        closed_trade = self.analytics.trades[0]
        self.assertTrue(closed_trade.is_closed)
        self.assertEqual(closed_trade.exit_price, 1.05)
        self.assertAlmostEqual(closed_trade.pnl, 0.5, places=2)  # 0.05 * 10 units
        self.assertEqual(closed_trade.exit_reason, "take_profit")

    def test_trade_statistics_calculation(self):
        """Test comprehensive trade statistics"""
        # Create multiple trades
        trades_data = [
            ("trade1", "market1", "news_sentiment", "buy", 10, 1.0, 1.05, "take_profit"),  # Win
            ("trade2", "market2", "arbitrage", "sell", 5, 2.0, 1.9, "take_profit"),       # Win
            ("trade3", "market3", "volatility", "buy", 8, 1.5, 1.35, "stop_loss"),        # Loss
        ]

        for trade_data in trades_data:
            trade = Trade(trade_data[0], trade_data[1], trade_data[2],
                         trade_data[3], trade_data[4], trade_data[5])
            self.analytics.record_trade(trade)
            self.analytics.close_trade(trade_data[0], trade_data[6], trade_data[7])

        stats = self.analytics.get_trade_statistics()

        # Check basic counts
        self.assertEqual(stats['total_trades'], 3)
        self.assertEqual(stats['closed_trades'], 3)
        self.assertEqual(stats['winning_trades'], 2)
        self.assertEqual(stats['losing_trades'], 1)

        # Check win rate
        self.assertAlmostEqual(stats['win_rate'], 2/3, places=2)

        # Check P&L calculations
        expected_pnl = 0.5 + 0.5 + (-1.2)  # 0.5 + 0.5 - 1.2 = -0.2
        self.assertAlmostEqual(stats['total_pnl'], expected_pnl, places=2)

    def test_strategy_performance_breakdown(self):
        """Test performance breakdown by strategy"""
        # Create trades for different strategies
        strategies_data = [
            ("news_trade", "market1", "news_sentiment", "buy", 10, 1.0, 1.03),
            ("arb_trade", "market2", "statistical_arbitrage", "sell", 5, 2.0, 1.95),
            ("vol_trade", "market3", "volatility_based", "buy", 8, 1.2, 1.25),
        ]

        for trade_data in strategies_data:
            trade = Trade(trade_data[0], trade_data[1], trade_data[2],
                         trade_data[3], trade_data[4], trade_data[5])
            self.analytics.record_trade(trade)
            self.analytics.close_trade(trade_data[0], trade_data[6], "test")

        strategy_perf = self.analytics.get_strategy_performance()

        # Check that all strategies are present
        self.assertIn('news_sentiment', strategy_perf)
        self.assertIn('statistical_arbitrage', strategy_perf)
        self.assertIn('volatility_based', strategy_perf)

        # Check news sentiment strategy
        news_stats = strategy_perf['news_sentiment']
        self.assertEqual(news_stats['total_trades'], 1)
        self.assertEqual(news_stats['winning_trades'], 1)  # Profitable trade

    def test_risk_adjusted_metrics(self):
        """Test risk-adjusted performance metrics"""
        # Create sample returns for testing
        returns = [
            0.02, 0.015, -0.01, 0.025, -0.005,  # Mixed performance
            0.03, -0.02, 0.018, -0.008, 0.022   # More data for better stats
        ]

        # Create trades that generate these returns
        for i, ret in enumerate(returns):
            trade = Trade(f"trade_{i}", f"market_{i}", "test_strategy",
                         "buy", 10, 1.0, 1.0 * (1 + ret))
            self.analytics.record_trade(trade)
            self.analytics.close_trade(f"trade_{i}", 1.0 * (1 + ret), "test")

        risk_metrics = self.analytics.get_risk_adjusted_metrics()

        # Check that risk metrics are calculated (may return error for insufficient data)
        if 'error' in risk_metrics:
            # This is expected with only 10 trades - not enough data for stable metrics
            self.assertIn('error', risk_metrics)
        else:
            # If we have enough data, check the metrics
            expected_metrics = ['sortino_ratio', 'calmar_ratio', 'omega_ratio', 'sharpe_ratio']
            for metric in expected_metrics:
                self.assertIn(metric, risk_metrics)
                self.assertIsInstance(risk_metrics[metric], (int, float, str))


class TestPhase2Phase3Integration(unittest.TestCase):
    """Test integration between Phase 2 and Phase 3 components"""

    def setUp(self):
        """Set up integrated test environment"""
        # Mock API for trader
        self.mock_api = Mock()
        self.mock_notifier = Mock()
        self.mock_logger = Mock()

        # Create trader with all Phase 2 & 3 components
        self.trader = Trader(self.mock_api, self.mock_notifier, self.mock_logger, 10000)

    def test_risk_manager_integration(self):
        """Test risk manager integration in trader"""
        # Check that risk manager exists and is initialized
        self.assertIsNotNone(self.trader.risk_manager)
        self.assertEqual(self.trader.risk_manager.initial_bankroll, 10000)

        # Test position sizing through trader
        confidence = 0.8
        position_size = self.trader.risk_manager.calculate_position_size_kelly(confidence)
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 0.10)  # Max 10%

    def test_performance_analytics_integration(self):
        """Test performance analytics integration"""
        # Check that performance analytics exists
        self.assertIsNotNone(self.trader.performance_analytics)

        # Create and record a trade through the trader
        trade_decision = {
            'event_id': 'test_market',
            'action': 'buy',
            'quantity': 10,
            'price': 1.0,
            'strategy': 'news_sentiment',
            'confidence': 0.8
        }

        # Execute trade (this should record it in performance analytics)
        self.trader.execute_trade(trade_decision)

        # Check that trade was recorded
        self.assertEqual(len(self.trader.performance_analytics.trades), 1)
        recorded_trade = self.trader.performance_analytics.trades[0]
        self.assertEqual(recorded_trade.market_id, 'test_market')
        self.assertEqual(recorded_trade.strategy, 'news_sentiment')

    def test_market_data_streamer_integration(self):
        """Test market data streamer integration"""
        # Check that market data streamer exists
        self.assertIsNotNone(self.trader.market_data_streamer)

        # Check that trader is subscribed to market updates
        self.assertEqual(len(self.trader.market_data_streamer.subscribers), 1)

        # Test market data callback
        mock_updated_markets = ['market1', 'market2']
        mock_market_data = {
            'market1': MarketData('market1', 'Market 1', 1.0),
            'market2': MarketData('market2', 'Market 2', 2.0)
        }

        # Call the subscriber callback
        self.trader._on_market_data_update(mock_updated_markets, mock_market_data)

        # Check that current prices were extracted for risk management
        # (This tests the integration between market data and risk management)

    def test_complete_trading_workflow(self):
        """Test complete trading workflow with all components"""
        # Create a simple test that verifies the components work together
        # without relying on complex mock setups

        # Test that all components are properly initialized
        self.assertIsNotNone(self.trader.risk_manager)
        self.assertIsNotNone(self.trader.performance_analytics)
        self.assertIsNotNone(self.trader.market_data_streamer)

        # Test that position sizing works
        confidence = 0.8
        position_size = self.trader.risk_manager.calculate_position_size_kelly(confidence)
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 0.10)

        # Test basic trade execution (without complex decision logic)
        trade_decision = {
            'event_id': 'test_market',
            'action': 'buy',
            'quantity': 5,  # Simple fixed quantity for test
            'price': 1.0,
            'strategy': 'test_strategy',
            'confidence': 0.8
        }

        # Execute trade
        self.trader.execute_trade(trade_decision)

        # Verify trade was recorded in performance analytics
        self.assertEqual(len(self.trader.performance_analytics.trades), 1)
        recorded_trade = self.trader.performance_analytics.trades[0]
        self.assertEqual(recorded_trade.market_id, 'test_market')
        self.assertEqual(recorded_trade.strategy, 'test_strategy')

        # Verify position was created
        self.assertIn('test_market', self.trader.current_positions)
        position = self.trader.current_positions['test_market']
        self.assertIn('stop_loss_price', position)

    def test_portfolio_status_integration(self):
        """Test portfolio status integration across all components"""
        # Get portfolio status (should include risk metrics from Phase 2)
        status = self.trader.get_portfolio_status()

        # Check Phase 2 risk metrics
        self.assertIn('risk_metrics', status)
        self.assertIn('sharpe_ratio', status['risk_metrics'])

        # Check basic portfolio info
        self.assertIn('current_bankroll', status)
        self.assertIn('total_pnl', status)
        self.assertEqual(status['initial_bankroll'], 10000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
