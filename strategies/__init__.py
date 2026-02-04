"""
Strategy Templates - Pre-configured trading strategies.

Available Strategies:
- conservative: Lower risk, tighter stops, smaller positions
- balanced: Default balanced approach
- aggressive: Higher risk tolerance, larger positions
- scalping: Quick trades, tight profits/stops
- swing: Longer holds, wider stops
"""
from .strategy_templates import (
    get_strategy,
    list_strategies,
    apply_strategy,
    STRATEGIES,
    StrategyConfig
)

__all__ = [
    'get_strategy',
    'list_strategies',
    'apply_strategy',
    'STRATEGIES',
    'StrategyConfig'
]
