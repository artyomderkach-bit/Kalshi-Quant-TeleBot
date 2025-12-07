#!/usr/bin/env python3
"""Simplified risk management module for Phase 2 - Kalshi trading bot."""

import logging
import numpy as np
from typing import Dict, Any
from config import BANKROLL, MAX_POSITION_SIZE_PERCENTAGE, STOP_LOSS_PERCENTAGE

logger = logging.getLogger(__name__)

class RiskManager:
    """Simplified risk management for Phase 2 - essential features only."""

    def __init__(self, initial_bankroll: float = BANKROLL):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll

    def calculate_position_size_kelly(self, confidence: float, win_loss_ratio: float = 2.0) -> float:
        """
        Simplified Kelly Criterion position sizing.

        Args:
            confidence: Strategy confidence (0-1)
            win_loss_ratio: Expected win/loss ratio

        Returns:
            Position size as fraction of bankroll (0-0.25 max)
        """
        if confidence <= 0 or confidence >= 1:
            return 0.02  # Minimum position size

        # Simplified Kelly: f = confidence * win_loss_ratio / (win_loss_ratio + 1)
        kelly_fraction = confidence * win_loss_ratio / (win_loss_ratio + 1)

        # Conservative: use half-Kelly and cap at 10%
        position_size = min(kelly_fraction * 0.5, 0.10)

        return max(position_size, 0.01)  # Minimum 1%

    def calculate_stop_loss_price(self, entry_price: float, is_long: bool = True) -> float:
        """
        Calculate simple stop-loss price based on percentage.

        Args:
            entry_price: Entry price
            is_long: True for long positions, False for short

        Returns:
            Stop-loss price
        """
        if is_long:
            return entry_price * (1 - STOP_LOSS_PERCENTAGE)
        else:
            return entry_price * (1 + STOP_LOSS_PERCENTAGE)

    def check_stop_loss_trigger(self, entry_price: float, current_price: float, is_long: bool = True) -> bool:
        """
        Check if stop-loss should be triggered.

        Args:
            entry_price: Entry price
            current_price: Current price
            is_long: True for long positions

        Returns:
            True if stop-loss triggered
        """
        stop_price = self.calculate_stop_loss_price(entry_price, is_long)

        if is_long:
            return current_price <= stop_price
        else:
            return current_price >= stop_price

    def calculate_portfolio_metrics(self, returns: list = None) -> Dict[str, float]:
        """
        Calculate basic portfolio risk metrics.

        Args:
            returns: List of daily returns (optional)

        Returns:
            Basic risk metrics
        """
        if not returns:
            returns = [0.01, -0.005, 0.008, -0.003, 0.012]  # Sample returns

        returns_array = np.array(returns)

        # Sharpe Ratio (simplified - assuming 2% risk-free rate)
        excess_returns = returns_array - 0.02/252  # Daily risk-free rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Win Rate
        winning_trades = sum(1 for r in returns_array if r > 0)
        win_rate = winning_trades / len(returns_array) if returns_array.size > 0 else 0

        return {
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_return': float(np.prod(1 + returns_array) - 1),
            'volatility': float(np.std(returns_array) * np.sqrt(252))
        }

    def validate_position_size(self, position_value: float) -> bool:
        """
        Validate position size against risk limits.

        Args:
            position_value: Dollar value of position

        Returns:
            True if position size is acceptable
        """
        max_position_value = self.current_bankroll * MAX_POSITION_SIZE_PERCENTAGE
        return position_value <= max_position_value

    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Get basic portfolio status.

        Returns:
            Portfolio status summary
        """
        metrics = self.calculate_portfolio_metrics()

        return {
            'current_bankroll': self.current_bankroll,
            'initial_bankroll': self.initial_bankroll,
            'total_pnl': self.current_bankroll - self.initial_bankroll,
            'total_return_pct': ((self.current_bankroll / self.initial_bankroll) - 1) * 100,
            'risk_metrics': metrics
        }
