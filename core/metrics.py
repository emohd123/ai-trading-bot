"""
Metrics Collection Module
Collects and exports metrics in Prometheus format for monitoring.
"""
import time
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects metrics for Prometheus export"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self._lock = threading.Lock()
        
        # Counter metrics
        self.api_calls_total = defaultdict(int)  # {operation: count}
        self.api_errors_total = defaultdict(int)  # {operation: count}
        self.trades_total = 0
        self.trades_wins = 0
        self.trades_losses = 0
        
        # Histogram metrics (track recent values)
        self.api_response_times = defaultdict(lambda: deque(maxlen=100))  # {operation: [times]}
        self.trade_execution_times = deque(maxlen=100)
        self.loop_iteration_times = deque(maxlen=100)
        
        # Gauge metrics (current values)
        self.current_price = 0.0
        self.balance_usdt = 0.0
        self.balance_btc = 0.0
        self.total_value = 0.0
        self.active_positions = 0
        self.connection_status = "unknown"
        
        # Timestamps
        self.last_api_call = {}
        self.last_trade_time = None
        self.start_time = datetime.now()
    
    def record_api_call(self, operation: str, duration: float, success: bool = True):
        """Record an API call"""
        with self._lock:
            self.api_calls_total[operation] += 1
            if success:
                self.api_response_times[operation].append(duration)
            else:
                self.api_errors_total[operation] += 1
            self.last_api_call[operation] = datetime.now()
    
    def record_trade(self, pnl: float, execution_time: float):
        """Record a trade"""
        with self._lock:
            self.trades_total += 1
            if pnl > 0:
                self.trades_wins += 1
            else:
                self.trades_losses += 1
            self.trade_execution_times.append(execution_time)
            self.last_trade_time = datetime.now()
    
    def record_loop_iteration(self, duration: float):
        """Record bot loop iteration time"""
        with self._lock:
            self.loop_iteration_times.append(duration)
    
    def update_gauge(self, metric: str, value: float):
        """Update a gauge metric"""
        with self._lock:
            if metric == "current_price":
                self.current_price = value
            elif metric == "balance_usdt":
                self.balance_usdt = value
            elif metric == "balance_btc":
                self.balance_btc = value
            elif metric == "total_value":
                self.total_value = value
            elif metric == "active_positions":
                self.active_positions = int(value)
            elif metric == "connection_status":
                self.connection_status = str(value)
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        with self._lock:
            lines = []
            
            # Counter: API calls
            for operation, count in self.api_calls_total.items():
                safe_op = operation.replace("-", "_").replace(" ", "_").lower()
                lines.append(f'api_calls_total{{operation="{operation}"}} {count}')
            
            # Counter: API errors
            for operation, count in self.api_errors_total.items():
                safe_op = operation.replace("-", "_").replace(" ", "_").lower()
                lines.append(f'api_errors_total{{operation="{operation}"}} {count}')
            
            # Counter: Trades
            lines.append(f'trades_total {self.trades_total}')
            lines.append(f'trades_wins_total {self.trades_wins}')
            lines.append(f'trades_losses_total {self.trades_losses}')
            
            # Histogram: API response times (average)
            for operation, times in self.api_response_times.items():
                if times:
                    avg_time = sum(times) / len(times)
                    safe_op = operation.replace("-", "_").replace(" ", "_").lower()
                    lines.append(f'api_response_time_seconds{{operation="{operation}"}} {avg_time:.4f}')
            
            # Histogram: Trade execution times (average)
            if self.trade_execution_times:
                avg_time = sum(self.trade_execution_times) / len(self.trade_execution_times)
                lines.append(f'trade_execution_time_seconds {avg_time:.4f}')
            
            # Histogram: Loop iteration times (average)
            if self.loop_iteration_times:
                avg_time = sum(self.loop_iteration_times) / len(self.loop_iteration_times)
                lines.append(f'bot_loop_iteration_time_seconds {avg_time:.4f}')
            
            # Gauge: Current values
            lines.append(f'current_price_usdt {self.current_price}')
            lines.append(f'balance_usdt {self.balance_usdt}')
            lines.append(f'balance_btc {self.balance_btc}')
            lines.append(f'total_value_usdt {self.total_value}')
            lines.append(f'active_positions {self.active_positions}')
            lines.append(f'connection_status{{status="{self.connection_status}"}} 1')
            
            # Uptime
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            lines.append(f'bot_uptime_seconds {uptime_seconds:.0f}')
            
            # Win rate
            if self.trades_total > 0:
                win_rate = (self.trades_wins / self.trades_total) * 100
                lines.append(f'win_rate_percent {win_rate:.2f}')
            else:
                lines.append(f'win_rate_percent 0')
            
            return "\n".join(lines) + "\n"
    
    def get_metrics_summary(self) -> Dict:
        """Get metrics summary as dictionary"""
        with self._lock:
            return {
                "api_calls": dict(self.api_calls_total),
                "api_errors": dict(self.api_errors_total),
                "trades": {
                    "total": self.trades_total,
                    "wins": self.trades_wins,
                    "losses": self.trades_losses,
                    "win_rate": (self.trades_wins / self.trades_total * 100) if self.trades_total > 0 else 0
                },
                "avg_api_response_times": {
                    op: sum(times) / len(times) if times else 0
                    for op, times in self.api_response_times.items()
                },
                "avg_trade_execution_time": (
                    sum(self.trade_execution_times) / len(self.trade_execution_times)
                    if self.trade_execution_times else 0
                ),
                "avg_loop_iteration_time": (
                    sum(self.loop_iteration_times) / len(self.loop_iteration_times)
                    if self.loop_iteration_times else 0
                ),
                "current_state": {
                    "price": self.current_price,
                    "balance_usdt": self.balance_usdt,
                    "balance_btc": self.balance_btc,
                    "total_value": self.total_value,
                    "active_positions": self.active_positions,
                    "connection_status": self.connection_status
                },
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            }


# Singleton instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get singleton metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
