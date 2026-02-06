"""
Paper Trading Mode - Simulated Trading
Simulates trades without real money for strategy testing.

Phase 6: Features & Functionality
"""
import os
import json
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime
from collections import deque

import config

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper trading system that simulates real trading.
    Tracks performance separately from live trading.
    """
    
    STATE_FILE = os.path.join(config.DATA_DIR, "paper_trading_state.json")
    
    def __init__(self, initial_balance: float = 1000.0):
        """
        Initialize paper trader.
        
        Args:
            initial_balance: Starting USDT balance
        """
        # Account state
        self.initial_balance = initial_balance
        self.usdt_balance = initial_balance
        self.btc_balance = 0.0
        
        # Positions
        self.positions = []  # List of open positions
        self.max_positions = getattr(config, 'MAX_POSITIONS', 2)
        
        # Trade history
        self.trade_history = []  # All trades
        self.equity_curve = []   # Balance over time
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
        # Session tracking
        self.session_start = datetime.now()
        self.last_price = 0.0
        
        self._load_state()
    
    def _load_state(self):
        """Load paper trading state from file"""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.usdt_balance = data.get('usdt_balance', self.initial_balance)
                self.btc_balance = data.get('btc_balance', 0)
                self.positions = data.get('positions', [])
                self.trade_history = data.get('trade_history', [])[-500:]
                self.total_trades = data.get('total_trades', 0)
                self.winning_trades = data.get('winning_trades', 0)
                self.total_pnl = data.get('total_pnl', 0)
                self.peak_balance = data.get('peak_balance', self.initial_balance)
                self.max_drawdown = data.get('max_drawdown', 0)
                self.initial_balance = data.get('initial_balance', self.initial_balance)
                
                start = data.get('session_start')
                if start:
                    self.session_start = datetime.fromisoformat(start)
                
                logger.info(f"Loaded paper trading: balance=${self.usdt_balance:.2f}, {len(self.positions)} positions")
        except Exception as e:
            logger.warning(f"Could not load paper trading state: {e}")
    
    def _save_state(self):
        """Save paper trading state to file"""
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            data = {
                'usdt_balance': self.usdt_balance,
                'btc_balance': self.btc_balance,
                'positions': self.positions,
                'trade_history': self.trade_history[-500:],
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'total_pnl': self.total_pnl,
                'peak_balance': self.peak_balance,
                'max_drawdown': self.max_drawdown,
                'initial_balance': self.initial_balance,
                'session_start': self.session_start.isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save paper trading state: {e}")
    
    def reset(self, initial_balance: float = None):
        """Reset paper trading account"""
        if initial_balance:
            self.initial_balance = initial_balance
        
        self.usdt_balance = self.initial_balance
        self.btc_balance = 0.0
        self.positions = []
        self.trade_history = []
        self.equity_curve = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.session_start = datetime.now()
        
        self._save_state()
        logger.info(f"Paper trading reset with ${self.initial_balance:.2f}")
    
    def get_balance(self, asset: str = "USDT") -> float:
        """Get balance for asset"""
        if asset.upper() == "USDT":
            return self.usdt_balance
        elif asset.upper() == "BTC":
            return self.btc_balance
        return 0.0
    
    def get_account_value(self, current_price: float) -> float:
        """Get total account value in USDT"""
        btc_value = self.btc_balance * current_price
        
        # Add value of open positions
        position_value = 0
        for pos in self.positions:
            position_value += pos['quantity'] * current_price
        
        return self.usdt_balance + btc_value + position_value
    
    def update_price(self, current_price: float):
        """Update with current price and track equity"""
        self.last_price = current_price
        
        # Calculate total value
        total_value = self.get_account_value(current_price)
        
        # Track peak and drawdown
        if total_value > self.peak_balance:
            self.peak_balance = total_value
        
        current_dd = (self.peak_balance - total_value) / self.peak_balance if self.peak_balance > 0 else 0
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        # Add to equity curve (every hour max)
        if not self.equity_curve or (datetime.now() - datetime.fromisoformat(self.equity_curve[-1]['timestamp'])).seconds >= 3600:
            self.equity_curve.append({
                'timestamp': datetime.now().isoformat(),
                'value': total_value,
                'price': current_price
            })
            self._save_state()
    
    def can_open_position(self) -> bool:
        """Check if new position can be opened"""
        return len(self.positions) < self.max_positions
    
    def place_market_buy(
        self,
        current_price: float,
        quote_amount: float = None
    ) -> Dict[str, Any]:
        """
        Simulate market buy order.
        
        Args:
            current_price: Current BTC price
            quote_amount: USDT amount to spend
            
        Returns:
            Order result dict
        """
        quote_amount = quote_amount or getattr(config, 'TRADE_AMOUNT_USDT', 40)
        
        # Check if enough balance
        if self.usdt_balance < quote_amount:
            return {"status": "error", "message": f"Insufficient balance: ${self.usdt_balance:.2f} < ${quote_amount:.2f}"}
        
        # Check position limit
        if not self.can_open_position():
            return {"status": "error", "message": f"Max positions ({self.max_positions}) reached"}
        
        # Simulate order with 0.1% slippage
        slippage = 0.001
        fill_price = current_price * (1 + slippage)
        quantity = quote_amount / fill_price
        
        # Apply trading fee (0.1%)
        fee = quote_amount * 0.001
        actual_cost = quote_amount + fee
        
        if self.usdt_balance < actual_cost:
            return {"status": "error", "message": f"Insufficient balance for fees"}
        
        # Execute
        self.usdt_balance -= actual_cost
        trade_id = len(self.trade_history) + 1
        
        position = {
            'trade_id': trade_id,
            'entry_price': fill_price,
            'quantity': quantity,
            'cost': quote_amount,
            'fee': fee,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'highest_price': fill_price,
            'lowest_price': fill_price
        }
        
        self.positions.append(position)
        
        # Record trade
        self.trade_history.append({
            'id': trade_id,
            'type': 'BUY',
            'time': position['time'],
            'price': fill_price,
            'quantity': quantity,
            'amount': quote_amount,
            'fee': fee
        })
        
        self._save_state()
        
        logger.info(f"[PAPER] BUY {quantity:.8f} BTC @ ${fill_price:,.2f} (${quote_amount:.2f})")
        
        return {
            "status": "filled",
            "side": "BUY",
            "price": fill_price,
            "quantity": quantity,
            "quote_amount": quote_amount,
            "fee": fee,
            "order_id": f"paper_{trade_id}",
            "paper_trade": True
        }
    
    def place_market_sell(
        self,
        current_price: float,
        position_index: int = 0
    ) -> Dict[str, Any]:
        """
        Simulate market sell order.
        
        Args:
            current_price: Current BTC price
            position_index: Index of position to close
            
        Returns:
            Order result dict
        """
        if not self.positions:
            return {"status": "error", "message": "No positions to sell"}
        
        if position_index >= len(self.positions):
            position_index = 0
        
        position = self.positions[position_index]
        quantity = position['quantity']
        entry_price = position['entry_price']
        
        # Simulate order with 0.1% slippage (negative for sell)
        slippage = 0.001
        fill_price = current_price * (1 - slippage)
        
        # Calculate value and fee
        gross_value = quantity * fill_price
        fee = gross_value * 0.001
        net_value = gross_value - fee
        
        # Calculate P/L
        cost = position['cost']
        pnl = net_value - cost
        pnl_percent = (pnl / cost) * 100
        
        # Execute
        self.usdt_balance += net_value
        self.positions.pop(position_index)
        
        # Update stats
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        
        trade_id = len(self.trade_history) + 1
        
        # Record trade
        self.trade_history.append({
            'id': trade_id,
            'type': 'SELL',
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'price': fill_price,
            'quantity': quantity,
            'amount': gross_value,
            'fee': fee,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'entry_price': entry_price,
            'buy_trade_id': position['trade_id']
        })
        
        self._save_state()
        
        logger.info(f"[PAPER] SELL {quantity:.8f} BTC @ ${fill_price:,.2f} | P/L: ${pnl:+.2f} ({pnl_percent:+.2f}%)")
        
        return {
            "status": "filled",
            "side": "SELL",
            "price": fill_price,
            "quantity": quantity,
            "quote_amount": net_value,
            "fee": fee,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "order_id": f"paper_{trade_id}",
            "paper_trade": True
        }
    
    def check_positions(self, current_price: float) -> List[Dict]:
        """
        Check all positions and update tracking.
        
        Returns:
            List of positions with current P/L
        """
        position_status = []
        
        for i, pos in enumerate(self.positions):
            # Update high/low tracking
            if current_price > pos['highest_price']:
                self.positions[i]['highest_price'] = current_price
            if current_price < pos['lowest_price']:
                self.positions[i]['lowest_price'] = current_price
            
            entry_price = pos['entry_price']
            if entry_price and entry_price > 0:
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                pnl_dollar = (current_price - entry_price) * pos['quantity']
            else:
                pnl_percent = 0
                pnl_dollar = 0
            
            position_status.append({
                'index': i,
                'entry_price': pos['entry_price'],
                'quantity': pos['quantity'],
                'current_price': current_price,
                'pnl_percent': pnl_percent,
                'pnl_dollar': pnl_dollar,
                'highest_price': pos['highest_price'],
                'lowest_price': pos['lowest_price'],
                'time': pos['time']
            })
        
        return position_status
    
    def get_performance(self) -> Dict:
        """Get paper trading performance metrics"""
        total_value = self.get_account_value(self.last_price) if self.last_price > 0 else self.usdt_balance
        total_return = (total_value - self.initial_balance) / self.initial_balance * 100 if self.initial_balance > 0 else 0
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Calculate average win/loss
        wins = [t['pnl'] for t in self.trade_history if t.get('type') == 'SELL' and t.get('pnl', 0) > 0]
        losses = [abs(t['pnl']) for t in self.trade_history if t.get('type') == 'SELL' and t.get('pnl', 0) < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else (float('inf') if wins else 0)
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.usdt_balance,
            'total_value': total_value,
            'total_return_pct': round(total_return, 2),
            'total_pnl': round(self.total_pnl, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': round(win_rate, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
            'peak_balance': round(self.peak_balance, 2),
            'max_drawdown_pct': round(self.max_drawdown * 100, 2),
            'open_positions': len(self.positions),
            'session_start': self.session_start.isoformat(),
            'equity_curve_points': len(self.equity_curve)
        }
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent trade history"""
        return self.trade_history[-limit:]
    
    def get_equity_curve(self, limit: int = 100) -> List[Dict]:
        """Get equity curve data"""
        return self.equity_curve[-limit:]


# Singleton instance
_paper_trader: Optional[PaperTrader] = None


def get_paper_trader(initial_balance: float = 1000.0) -> PaperTrader:
    """Get or create singleton paper trader"""
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTrader(initial_balance)
    return _paper_trader


def is_paper_mode() -> bool:
    """Check if paper trading mode is enabled"""
    return getattr(config, 'PAPER_TRADING_MODE', False)
