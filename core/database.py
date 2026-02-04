"""
Database Layer - SQLite for Persistent Storage
Stores trade history, ML predictions, and performance metrics.

Phase 7: Performance Optimization
"""
import os
import sqlite3
import json
import logging
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
import threading

import config

logger = logging.getLogger(__name__)


class Database:
    """
    SQLite database for persistent storage.
    Thread-safe with connection pooling.
    """
    
    DB_FILE = os.path.join(config.DATA_DIR, "trading_bot.db")
    
    def __init__(self):
        """Initialize database"""
        self._local = threading.local()
        self._lock = threading.Lock()
        
        os.makedirs(config.DATA_DIR, exist_ok=True)
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.DB_FILE,
                check_same_thread=False,
                timeout=30
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()
    
    def _init_schema(self):
        """Initialize database schema"""
        with self.get_cursor() as cursor:
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER UNIQUE,
                    type TEXT NOT NULL,
                    time TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    amount REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    pnl_percent REAL DEFAULT 0,
                    exit_type TEXT,
                    regime TEXT,
                    confidence REAL,
                    confluence INTEGER,
                    ai_score REAL,
                    entry_price REAL,
                    highest_price REAL,
                    lowest_price REAL,
                    duration_minutes REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # ML predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    price REAL NOT NULL,
                    prediction_up REAL,
                    prediction_down REAL,
                    confidence REAL,
                    model_votes TEXT,
                    regime TEXT,
                    actual_direction TEXT,
                    was_correct INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Performance snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_value REAL,
                    usdt_balance REAL,
                    btc_balance REAL,
                    open_positions INTEGER,
                    total_pnl REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indicator accuracy table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indicator_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator TEXT NOT NULL,
                    regime TEXT,
                    accuracy REAL,
                    samples INTEGER,
                    bullish_accuracy REAL,
                    bearish_accuracy REAL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Error logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    message TEXT,
                    category TEXT,
                    severity TEXT,
                    context TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_type ON trades(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_timestamp ON ml_predictions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_snapshots(timestamp)")
    
    # =========================================================================
    # TRADE OPERATIONS
    # =========================================================================
    
    def insert_trade(self, trade: Dict) -> int:
        """Insert a trade record"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO trades (
                    trade_id, type, time, price, quantity, amount,
                    pnl, pnl_percent, exit_type, regime, confidence,
                    confluence, ai_score, entry_price, highest_price,
                    lowest_price, duration_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get('id') or trade.get('trade_id'),
                trade.get('type'),
                trade.get('time'),
                trade.get('price'),
                trade.get('quantity'),
                trade.get('amount'),
                trade.get('pnl', 0),
                trade.get('pnl_percent', 0),
                trade.get('exit_type'),
                trade.get('regime'),
                trade.get('confidence'),
                trade.get('confluence'),
                trade.get('ai_score'),
                trade.get('entry_price'),
                trade.get('highest_price'),
                trade.get('lowest_price'),
                trade.get('duration_minutes')
            ))
            return cursor.lastrowid
    
    def get_trades(
        self,
        limit: int = 100,
        trade_type: str = None,
        start_time: str = None,
        end_time: str = None
    ) -> List[Dict]:
        """Get trades with optional filters"""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if trade_type:
            query += " AND type = ?"
            params.append(trade_type)
        
        if start_time:
            query += " AND time >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND time <= ?"
            params.append(end_time)
        
        query += " ORDER BY time DESC LIMIT ?"
        params.append(limit)
        
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_trade_stats(self, days: int = 30) -> Dict:
        """Get trade statistics for period"""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        
        with self.get_cursor() as cursor:
            # Total stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl,
                    AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
                    AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss,
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss
                FROM trades 
                WHERE type = 'SELL' AND time >= ?
            """, (cutoff,))
            
            row = cursor.fetchone()
            
            if row and row['total_trades'] > 0:
                win_rate = (row['winning_trades'] / row['total_trades']) * 100
                profit_factor = row['gross_profit'] / row['gross_loss'] if row['gross_loss'] > 0 else 0
            else:
                win_rate = 0
                profit_factor = 0
            
            return {
                'total_trades': row['total_trades'] if row else 0,
                'winning_trades': row['winning_trades'] if row else 0,
                'total_pnl': round(row['total_pnl'] or 0, 2),
                'avg_win': round(row['avg_win'] or 0, 2),
                'avg_loss': round(row['avg_loss'] or 0, 2),
                'win_rate': round(win_rate, 1),
                'profit_factor': round(profit_factor, 2)
            }
    
    # =========================================================================
    # ML PREDICTION OPERATIONS
    # =========================================================================
    
    def insert_prediction(self, prediction: Dict) -> int:
        """Insert ML prediction record"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO ml_predictions (
                    timestamp, price, prediction_up, prediction_down,
                    confidence, model_votes, regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.get('timestamp', datetime.now().isoformat()),
                prediction.get('price'),
                prediction.get('prediction_up'),
                prediction.get('prediction_down'),
                prediction.get('confidence'),
                json.dumps(prediction.get('model_votes', {})),
                prediction.get('regime')
            ))
            return cursor.lastrowid
    
    def update_prediction_outcome(
        self,
        prediction_id: int,
        actual_direction: str,
        was_correct: bool
    ):
        """Update prediction with actual outcome"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE ml_predictions 
                SET actual_direction = ?, was_correct = ?
                WHERE id = ?
            """, (actual_direction, 1 if was_correct else 0, prediction_id))
    
    def get_ml_accuracy(self, days: int = 7) -> Dict:
        """Get ML prediction accuracy"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(was_correct) as correct
                FROM ml_predictions 
                WHERE timestamp >= ? AND was_correct IS NOT NULL
            """, (cutoff,))
            
            row = cursor.fetchone()
            
            if row and row['total'] > 0:
                accuracy = (row['correct'] / row['total']) * 100
            else:
                accuracy = 0
            
            return {
                'total_predictions': row['total'] if row else 0,
                'correct_predictions': row['correct'] if row else 0,
                'accuracy': round(accuracy, 1)
            }
    
    # =========================================================================
    # PERFORMANCE SNAPSHOT OPERATIONS
    # =========================================================================
    
    def insert_performance_snapshot(self, snapshot: Dict) -> int:
        """Insert performance snapshot"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO performance_snapshots (
                    timestamp, total_value, usdt_balance, btc_balance,
                    open_positions, total_pnl, total_trades, win_rate,
                    sharpe_ratio, max_drawdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.get('timestamp', datetime.now().isoformat()),
                snapshot.get('total_value'),
                snapshot.get('usdt_balance'),
                snapshot.get('btc_balance'),
                snapshot.get('open_positions'),
                snapshot.get('total_pnl'),
                snapshot.get('total_trades'),
                snapshot.get('win_rate'),
                snapshot.get('sharpe_ratio'),
                snapshot.get('max_drawdown')
            ))
            return cursor.lastrowid
    
    def get_equity_curve(self, days: int = 30) -> List[Dict]:
        """Get equity curve data"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT timestamp, total_value, usdt_balance, btc_balance
                FROM performance_snapshots 
                WHERE timestamp >= ?
                ORDER BY timestamp
            """, (cutoff,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # ERROR LOG OPERATIONS
    # =========================================================================
    
    def log_error(
        self,
        error_type: str,
        message: str,
        category: str = None,
        severity: str = None,
        context: Dict = None
    ) -> int:
        """Log an error"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO error_logs (
                    timestamp, error_type, message, category, severity, context
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                error_type,
                message,
                category,
                severity,
                json.dumps(context) if context else None
            ))
            return cursor.lastrowid
    
    def get_recent_errors(self, limit: int = 50) -> List[Dict]:
        """Get recent error logs"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM error_logs 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # CLEANUP OPERATIONS
    # =========================================================================
    
    def cleanup_old_data(self, days: int = 90):
        """Remove data older than specified days"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self.get_cursor() as cursor:
            # Keep trade history longer
            cursor.execute("DELETE FROM ml_predictions WHERE timestamp < ?", (cutoff,))
            cursor.execute("DELETE FROM performance_snapshots WHERE timestamp < ?", (cutoff,))
            cursor.execute("DELETE FROM error_logs WHERE timestamp < ?", (cutoff,))
            
            # Vacuum to reclaim space
            cursor.execute("VACUUM")
        
        logger.info(f"Cleaned up data older than {days} days")
    
    def close(self):
        """Close database connection"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Singleton instance
_database: Optional[Database] = None


def get_database() -> Database:
    """Get or create singleton database"""
    global _database
    if _database is None:
        _database = Database()
    return _database
