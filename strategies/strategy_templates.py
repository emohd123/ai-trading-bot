"""
Strategy Templates - Pre-configured trading strategy profiles.

Each strategy defines:
- Risk parameters (stop loss, profit targets)
- Position sizing rules
- Entry/exit thresholds
- Regime-specific behavior

Usage:
    from strategies import get_strategy, apply_strategy
    
    # Get a strategy config
    config = get_strategy('conservative')
    
    # Apply to bot config
    apply_strategy('aggressive')
"""
import os
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    description: str
    
    # Risk parameters
    stop_loss_pct: float          # Base stop loss percentage
    profit_target_pct: float      # Base profit target percentage
    min_profit_pct: float         # Minimum profit to take
    trailing_stop_pct: float      # Trailing stop distance
    
    # Position sizing
    base_position_pct: float      # Base position as % of portfolio
    max_position_pct: float       # Maximum position size
    position_scale_factor: float  # Scale factor for sizing
    
    # Entry thresholds
    min_confidence: float         # Minimum AI confidence to enter
    min_confluence: int           # Minimum confluence count
    buy_threshold: float          # AI score threshold for buy
    
    # Exit behavior
    use_trailing_profit: bool     # Enable trailing profit targets
    quick_profit_mode: bool       # Take profits quickly
    
    # Regime behavior
    trade_downtrend: bool         # Allow trading in downtrends
    downtrend_size_mult: float    # Position size multiplier in downtrend
    
    # Risk limits
    max_daily_loss_pct: float     # Maximum daily loss allowed
    max_consecutive_losses: int   # Max losses before pause
    
    def to_dict(self) -> Dict:
        return asdict(self)


# Pre-defined strategy templates
STRATEGIES: Dict[str, StrategyConfig] = {
    
    "conservative": StrategyConfig(
        name="conservative",
        description="Lower risk strategy with tight stops and smaller positions. Best for capital preservation.",
        stop_loss_pct=0.5,
        profit_target_pct=1.0,
        min_profit_pct=0.2,
        trailing_stop_pct=0.2,
        base_position_pct=15.0,
        max_position_pct=25.0,
        position_scale_factor=0.7,
        min_confidence=0.65,
        min_confluence=6,
        buy_threshold=0.62,
        use_trailing_profit=True,
        quick_profit_mode=True,
        trade_downtrend=False,
        downtrend_size_mult=0.3,
        max_daily_loss_pct=3.0,
        max_consecutive_losses=3
    ),
    
    "balanced": StrategyConfig(
        name="balanced",
        description="Balanced approach with moderate risk. Good for steady growth.",
        stop_loss_pct=0.75,
        profit_target_pct=1.5,
        min_profit_pct=0.25,
        trailing_stop_pct=0.3,
        base_position_pct=20.0,
        max_position_pct=40.0,
        position_scale_factor=1.0,
        min_confidence=0.55,
        min_confluence=5,
        buy_threshold=0.58,
        use_trailing_profit=True,
        quick_profit_mode=False,
        trade_downtrend=True,
        downtrend_size_mult=0.5,
        max_daily_loss_pct=5.0,
        max_consecutive_losses=4
    ),
    
    "aggressive": StrategyConfig(
        name="aggressive",
        description="Higher risk strategy with larger positions. For experienced traders.",
        stop_loss_pct=1.0,
        profit_target_pct=2.5,
        min_profit_pct=0.4,
        trailing_stop_pct=0.5,
        base_position_pct=30.0,
        max_position_pct=60.0,
        position_scale_factor=1.5,
        min_confidence=0.50,
        min_confluence=4,
        buy_threshold=0.55,
        use_trailing_profit=True,
        quick_profit_mode=False,
        trade_downtrend=True,
        downtrend_size_mult=0.7,
        max_daily_loss_pct=8.0,
        max_consecutive_losses=5
    ),
    
    "scalping": StrategyConfig(
        name="scalping",
        description="Quick trades with tight stops and small profits. High frequency.",
        stop_loss_pct=0.3,
        profit_target_pct=0.5,
        min_profit_pct=0.15,
        trailing_stop_pct=0.1,
        base_position_pct=25.0,
        max_position_pct=35.0,
        position_scale_factor=1.0,
        min_confidence=0.55,
        min_confluence=4,
        buy_threshold=0.55,
        use_trailing_profit=False,
        quick_profit_mode=True,
        trade_downtrend=True,
        downtrend_size_mult=0.8,
        max_daily_loss_pct=4.0,
        max_consecutive_losses=5
    ),
    
    "swing": StrategyConfig(
        name="swing",
        description="Longer holds with wider stops. Captures bigger moves.",
        stop_loss_pct=2.0,
        profit_target_pct=5.0,
        min_profit_pct=1.0,
        trailing_stop_pct=1.0,
        base_position_pct=15.0,
        max_position_pct=30.0,
        position_scale_factor=0.8,
        min_confidence=0.65,
        min_confluence=6,
        buy_threshold=0.65,
        use_trailing_profit=True,
        quick_profit_mode=False,
        trade_downtrend=False,
        downtrend_size_mult=0.4,
        max_daily_loss_pct=6.0,
        max_consecutive_losses=3
    ),
    
    "trend_following": StrategyConfig(
        name="trend_following",
        description="Only trades with the trend. Higher win rate, fewer trades.",
        stop_loss_pct=1.0,
        profit_target_pct=3.0,
        min_profit_pct=0.5,
        trailing_stop_pct=0.5,
        base_position_pct=20.0,
        max_position_pct=40.0,
        position_scale_factor=1.2,
        min_confidence=0.60,
        min_confluence=5,
        buy_threshold=0.60,
        use_trailing_profit=True,
        quick_profit_mode=False,
        trade_downtrend=False,
        downtrend_size_mult=0.0,  # No trading in downtrend
        max_daily_loss_pct=5.0,
        max_consecutive_losses=4
    )
}


def get_strategy(name: str) -> Optional[StrategyConfig]:
    """
    Get a strategy configuration by name.
    
    Args:
        name: Strategy name (conservative, balanced, aggressive, scalping, swing)
    
    Returns:
        StrategyConfig or None if not found
    """
    return STRATEGIES.get(name.lower())


def list_strategies() -> List[Dict]:
    """
    List all available strategies with descriptions.
    
    Returns:
        List of strategy summaries
    """
    return [
        {
            "name": s.name,
            "description": s.description,
            "risk_level": "low" if s.stop_loss_pct < 0.6 else "medium" if s.stop_loss_pct < 1.0 else "high",
            "stop_loss": f"{s.stop_loss_pct}%",
            "profit_target": f"{s.profit_target_pct}%"
        }
        for s in STRATEGIES.values()
    ]


def apply_strategy(name: str, config_module=None) -> bool:
    """
    Apply a strategy to the bot configuration.
    
    Args:
        name: Strategy name
        config_module: Config module to update (defaults to config.py)
    
    Returns:
        True if applied successfully
    """
    strategy = get_strategy(name)
    if not strategy:
        return False
    
    if config_module is None:
        try:
            import config as config_module
        except ImportError:
            return False
    
    # Apply strategy settings to config
    try:
        # Risk parameters
        if hasattr(config_module, 'STOP_LOSS_PCT'):
            config_module.STOP_LOSS_PCT = strategy.stop_loss_pct / 100
        if hasattr(config_module, 'PROFIT_TARGET'):
            config_module.PROFIT_TARGET = strategy.profit_target_pct / 100
        if hasattr(config_module, 'MIN_PROFIT_PCT'):
            config_module.MIN_PROFIT_PCT = strategy.min_profit_pct / 100
        
        # Position sizing
        if hasattr(config_module, 'POSITION_SIZE_PCT'):
            config_module.POSITION_SIZE_PCT = strategy.base_position_pct / 100
        
        # Entry thresholds
        if hasattr(config_module, 'MIN_CONFIDENCE'):
            config_module.MIN_CONFIDENCE = strategy.min_confidence
        if hasattr(config_module, 'MIN_CONFLUENCE_BUY'):
            config_module.MIN_CONFLUENCE_BUY = strategy.min_confluence
        if hasattr(config_module, 'BUY_SCORE_THRESHOLD'):
            config_module.BUY_SCORE_THRESHOLD = strategy.buy_threshold
        
        # Risk limits
        if hasattr(config_module, 'DAILY_LOSS_LIMIT_PCT'):
            config_module.DAILY_LOSS_LIMIT_PCT = strategy.max_daily_loss_pct / 100
        
        return True
        
    except Exception:
        return False


def save_strategy(strategy: StrategyConfig, filepath: str):
    """Save a strategy configuration to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(strategy.to_dict(), f, indent=2)


def load_strategy(filepath: str) -> Optional[StrategyConfig]:
    """Load a strategy configuration from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return StrategyConfig(**data)
    except Exception:
        return None


# Display strategies when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Available Trading Strategies")
    print("=" * 60)
    
    for strategy in list_strategies():
        print(f"\n{strategy['name'].upper()}")
        print(f"  Description: {strategy['description']}")
        print(f"  Risk Level: {strategy['risk_level']}")
        print(f"  Stop Loss: {strategy['stop_loss']}")
        print(f"  Profit Target: {strategy['profit_target']}")
    
    print("\n" + "=" * 60)
    print("Usage: from strategies import get_strategy, apply_strategy")
    print("       apply_strategy('conservative')")
    print("=" * 60)
