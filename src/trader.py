import pandas as pd
import numpy as np
import logging
from config import BANKROLL, NEWS_SENTIMENT_THRESHOLD, STAT_ARBITRAGE_THRESHOLD, VOLATILITY_THRESHOLD, MAX_POSITION_SIZE_PERCENTAGE, STOP_LOSS_PERCENTAGE
from news_analyzer import NewsSentimentAnalyzer
from arbitrage_analyzer import StatisticalArbitrageAnalyzer
from volatility_analyzer import VolatilityAnalyzer

class Trader:
    def __init__(self, api, notifier, logger, bankroll):
        self.api = api
        self.notifier = notifier
        self.logger = logger
        self.bankroll = bankroll
        self.current_positions = {}
        self.news_analyzer = NewsSentimentAnalyzer()
        self.arbitrage_analyzer = StatisticalArbitrageAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()

    def analyze_market(self, market_data):
        # Enhanced analysis with news sentiment
        return self._make_trade_decision(market_data)

    def _make_trade_decision(self, market_data):
        """
        Enhanced trade decision making with multiple strategies
        Priority: News Sentiment → Statistical Arbitrage → Volatility Analysis
        """
        trade_decision = None

        # Strategy 1: News Sentiment Analysis
        try:
            sentiment_analysis = self.news_analyzer.get_market_relevant_news()
            sentiment_decision = self.news_analyzer.should_trade_based_on_sentiment(
                sentiment_analysis, NEWS_SENTIMENT_THRESHOLD
            )

            if sentiment_decision['should_trade']:
                self.logger.info(f"News sentiment signal: {sentiment_decision['reason']}")

                # Find suitable market to trade based on sentiment
                if market_data and 'markets' in market_data and market_data['markets']:
                    market = market_data['markets'][0]  # Simple selection - could be enhanced
                    event_id = market.get('id')
                    current_price = market.get('current_price')

                    if event_id and current_price:
                        action = 'buy' if sentiment_decision['direction'] == 'long' else 'sell'
                        quantity = 1  # Base quantity - will be adjusted by risk management

                        trade_decision = {
                            'event_id': event_id,
                            'action': action,
                            'quantity': quantity,
                            'price': current_price,
                            'strategy': 'news_sentiment',
                            'sentiment_score': sentiment_decision['sentiment_score'],
                            'confidence': sentiment_decision['confidence']
                        }

                        self.logger.info(f"News sentiment trade decision: {action} {event_id} "
                                       f"at {current_price} (sentiment: {sentiment_decision['sentiment_score']:.3f})")

        except Exception as e:
            self.logger.error(f"Error in news sentiment analysis: {e}")

        # Strategy 2: Statistical Arbitrage (if no sentiment signal)
        if not trade_decision:
            try:
                arbitrage_opportunities = self._statistical_arbitrage(market_data)
                if arbitrage_opportunities:
                    # Take the highest confidence opportunity
                    best_opportunity = arbitrage_opportunities[0]
                    execution_decision = self.arbitrage_analyzer.should_execute_arbitrage(
                        best_opportunity, risk_tolerance=0.7
                    )

                    if execution_decision['should_execute']:
                        self.logger.info(f"Arbitrage signal: {execution_decision['reason']}")

                        # For simplicity, focus on one side of the arbitrage pair
                        # In a real implementation, you'd trade both sides
                        market1 = execution_decision['market1']
                        market2 = execution_decision['market2']

                        if best_opportunity['signal'] == 'LONG_SPREAD':
                            event_id = market1['id']
                            action = 'buy'
                        else:  # SHORT_SPREAD
                            event_id = market1['id']
                            action = 'sell'

                        quantity = int(execution_decision['position_size'] * 10)  # Scale up for meaningful position
                        quantity = max(1, quantity)  # Minimum 1 unit

                        trade_decision = {
                            'event_id': event_id,
                            'action': action,
                            'quantity': quantity,
                            'price': market1['current_price'],
                            'strategy': 'statistical_arbitrage',
                            'z_score': best_opportunity['z_score'],
                            'confidence': execution_decision['confidence'],
                            'arbitrage_pair': [market1['id'], market2['id']]
                        }

                        self.logger.info(f"Arbitrage trade decision: {action} {event_id} "
                                       f"(z-score: {best_opportunity['z_score']:.3f})")

            except Exception as e:
                self.logger.error(f"Error in statistical arbitrage: {e}")

        # Strategy 3: Volatility Analysis (if no other signals)
        if not trade_decision:
            try:
                volatility_decision = self._volatility_analysis(market_data)
                if volatility_decision and volatility_decision.get('should_trade'):
                    self.logger.info(f"Volatility signal: {volatility_decision['reason']}")

                    # Find market for volatility-based trade
                    if market_data and 'markets' in market_data and market_data['markets']:
                        market = market_data['markets'][0]  # Could be enhanced to select based on volatility
                        event_id = market.get('id')
                        current_price = market.get('current_price')

                        if event_id and current_price and volatility_decision.get('direction'):
                            action = 'buy' if volatility_decision['direction'] == 'long' else 'sell'
                            quantity = 1  # Base quantity

                            trade_decision = {
                                'event_id': event_id,
                                'action': action,
                                'quantity': quantity,
                                'price': current_price,
                                'strategy': 'volatility_based',
                                'volatility_regime': volatility_decision.get('volatility_regime'),
                                'confidence': volatility_decision['confidence'],
                                'signal_type': volatility_decision.get('signal_type')
                            }

                            self.logger.info(f"Volatility trade decision: {action} {event_id} "
                                           f"(regime: {volatility_decision.get('volatility_regime')})")

            except Exception as e:
                self.logger.error(f"Error in volatility analysis: {e}")

        return trade_decision

    def execute_trade(self, trade_decision):
        if not trade_decision:
            self.logger.info("No trade decision to execute.")
            return

        event_id = trade_decision['event_id']
        action = trade_decision['action']
        quantity = trade_decision['quantity']
        price = trade_decision['price']

        # Risk Management: Position Sizing
        max_trade_value = self.bankroll * MAX_POSITION_SIZE_PERCENTAGE
        if (quantity * price) > max_trade_value:
            self.logger.warning(f"Trade value ({quantity * price}) exceeds max position size ({max_trade_value}). Adjusting quantity.")
            quantity = int(max_trade_value / price)
            if quantity == 0:
                self.logger.warning("Adjusted quantity is zero. Skipping trade.")
                return

        try:
            if action == 'buy':
                self.logger.info(f"Executing buy trade for event {event_id} at price {price} for {quantity} units.")
                # Simulate API call
                # self.api.buy_contract(event_id, quantity, price)
                self.current_positions[event_id] = self.current_positions.get(event_id, 0) + quantity
                self.notifier.send_trade_notification(f"Bought {quantity} units of {event_id} at {price}.")
            elif action == 'sell':
                self.logger.info(f"Executing sell trade for event {event_id} at price {price} for {quantity} units.")
                # Simulate API call
                # self.api.sell_contract(event_id, quantity, price)
                self.current_positions[event_id] = self.current_positions.get(event_id, 0) - quantity
                self.notifier.send_trade_notification(f"Sold {quantity} units of {event_id} at {price}.")

            # Risk Management: Stop-Loss (simplified, would need real-time price monitoring)
            if event_id in self.current_positions and self.current_positions[event_id] > 0:
                # This is a very simplified stop-loss. In a real bot, you'd monitor the price
                # and compare it to the entry price. For now, just a placeholder.
                pass

        except Exception as e:
            self.logger.error(f"Error executing trade for {event_id}: {e}")
            self.notifier.send_error_notification(f"Trade execution error for {event_id}: {e}")

    # Placeholder methods for future strategies - will be implemented in Phase 1
    def _news_sentiment_analysis(self, news_data):
        """
        Placeholder for news sentiment analysis - now handled by NewsSentimentAnalyzer
        This method is kept for backward compatibility but delegates to the new analyzer
        """
        self.logger.info("News sentiment analysis now handled by NewsSentimentAnalyzer")
        return 0.7  # Default positive sentiment

    def _statistical_arbitrage(self, market_data):
        """
        Find statistical arbitrage opportunities in market data
        """
        if not market_data or 'markets' not in market_data:
            return []

        markets = market_data['markets']
        if len(markets) < 2:
            return []  # Need at least 2 markets for arbitrage

        # Prepare market data with price history for arbitrage analysis
        # Note: In a real implementation, you'd need historical price data
        # For now, we'll simulate with current prices and some noise
        arbitrage_ready_markets = []

        for market in markets[:10]:  # Limit to first 10 markets for performance
            market_id = market.get('id')
            current_price = market.get('current_price', 0.5)

            if market_id and current_price:
                # Generate synthetic price history for demonstration
                # In production, this would come from historical data
                price_history = self._generate_price_history(current_price)

                arbitrage_ready_markets.append({
                    'id': market_id,
                    'title': market.get('title', ''),
                    'current_price': current_price,
                    'price_history': price_history
                })

        if len(arbitrage_ready_markets) < 2:
            return []

        # Find arbitrage opportunities
        opportunities = self.arbitrage_analyzer.find_arbitrage_opportunities(arbitrage_ready_markets)

        self.logger.info(f"Found {len(opportunities)} arbitrage opportunities")
        return opportunities

    def _generate_price_history(self, current_price: float, periods: int = 100) -> List[float]:
        """
        Generate synthetic price history for arbitrage analysis.
        In production, this would be real historical data.
        """
        # Start with a random walk around the current price
        prices = [current_price]

        # Generate random walk with mean reversion
        for i in range(periods - 1):
            # Add some noise with slight mean reversion
            change = np.random.normal(0, 0.02) - 0.001 * (prices[-1] - current_price)
            new_price = max(0.01, min(0.99, prices[-1] + change))  # Keep in [0.01, 0.99]
            prices.append(new_price)

        return prices

    def _volatility_analysis(self, market_data):
        """
        Analyze volatility patterns for trading opportunities
        """
        if not market_data or 'markets' not in market_data:
            return None

        markets = market_data['markets']
        if not markets:
            return None

        # Analyze volatility for the first few markets
        volatility_opportunities = []

        for market in markets[:5]:  # Limit analysis for performance
            market_id = market.get('id')
            current_price = market.get('current_price', 0.5)

            if market_id and current_price:
                # Prepare market data for volatility analysis
                price_history = self._generate_price_history(current_price, periods=150)  # More data for volatility

                market_data_for_analysis = {
                    'id': market_id,
                    'title': market.get('title', ''),
                    'current_price': current_price,
                    'price_history': price_history
                }

                # Analyze volatility
                volatility_result = self.volatility_analyzer.analyze_market_volatility(market_data_for_analysis)

                if 'error' not in volatility_result:
                    # Check if this presents a trading opportunity
                    trade_decision = self.volatility_analyzer.should_trade_based_on_volatility(
                        volatility_result, risk_tolerance=0.6
                    )

                    if trade_decision['should_trade']:
                        volatility_opportunities.append({
                            'market': market_data_for_analysis,
                            'volatility_analysis': volatility_result,
                            'trade_decision': trade_decision
                        })

        if not volatility_opportunities:
            return None

        # Return the highest confidence opportunity
        best_opportunity = max(volatility_opportunities,
                              key=lambda x: x['trade_decision']['confidence'])

        self.logger.info(f"Found {len(volatility_opportunities)} volatility-based opportunities")

        # Return the trade decision for the best opportunity
        decision = best_opportunity['trade_decision']
        decision.update({
            'market_data': best_opportunity['market'],
            'volatility_analysis': best_opportunity['volatility_analysis']
        })

        return decision

    def run_trading_strategy(self):
        """
        Main trading strategy orchestration with multiple quantitative strategies
        Priority: News Sentiment → Statistical Arbitrage → Volatility Analysis
        """
        self.logger.info("Running multi-strategy trading system: News Sentiment + Arbitrage + Volatility")

        # Get market data from API
        market_data = self.api.fetch_market_data()
        if not market_data:
            self.logger.info("No market data available")
            return

        # Run multi-strategy analysis
        trade_decision = self._make_trade_decision(market_data)

        if trade_decision:
            strategy_name = trade_decision.get('strategy', 'unknown')
            self.logger.info(f"Executing trade via {strategy_name} strategy")
            self.execute_trade(trade_decision)
        else:
            self.logger.info("No profitable opportunities found across all strategies")


