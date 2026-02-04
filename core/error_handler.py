"""
Centralized Error Handler
Categorizes errors, implements retry logic, and provides recovery strategies.

Phase 2: Code quality improvements
"""
import logging
import time
import traceback
from typing import Callable, Any, Optional, Dict, Tuple
from functools import wraps
from enum import Enum
from datetime import datetime, timedelta
import json
import os

import config

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for handling"""
    API_ERROR = "api_error"           # Binance API errors
    NETWORK_ERROR = "network_error"   # Connection/timeout errors
    DATA_ERROR = "data_error"         # Invalid/missing data
    LOGIC_ERROR = "logic_error"       # Business logic errors
    RATE_LIMIT = "rate_limit"         # Rate limiting
    AUTH_ERROR = "auth_error"         # Authentication errors
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels"""
    LOW = "low"           # Log and continue
    MEDIUM = "medium"     # Retry with backoff
    HIGH = "high"         # Alert and pause
    CRITICAL = "critical" # Stop trading


class ErrorRecord:
    """Record of an error occurrence"""
    
    def __init__(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Dict = None
    ):
        self.error = error
        self.error_type = type(error).__name__
        self.message = str(error)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()
    
    def to_dict(self) -> Dict:
        return {
            'error_type': self.error_type,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback
        }


class ErrorHandler:
    """
    Centralized error handler with retry logic and recovery strategies.
    """
    
    ERROR_LOG_FILE = os.path.join(config.DATA_DIR, "error_log.json")
    MAX_ERROR_HISTORY = 100
    
    # Error patterns for categorization
    API_ERRORS = ['BinanceAPIException', 'APIError', 'InvalidOrder']
    NETWORK_ERRORS = ['ConnectionError', 'Timeout', 'TimeoutError', 'ReadTimeout']
    RATE_LIMIT_ERRORS = ['RateLimitError', '-1015', 'Too many requests']
    AUTH_ERRORS = ['AuthenticationError', '-2015', 'Invalid API-key']
    
    def __init__(self):
        self.error_history = []
        self.error_counts = {}  # {category: count}
        self.last_errors = {}   # {category: timestamp}
        self.circuit_breaker = {}  # {operation: {failures, last_failure, is_open}}
        
        self._load_error_history()
    
    def _load_error_history(self):
        """Load error history from file"""
        try:
            if os.path.exists(self.ERROR_LOG_FILE):
                with open(self.ERROR_LOG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.error_history = data.get('history', [])[-self.MAX_ERROR_HISTORY:]
                self.error_counts = data.get('counts', {})
        except Exception:
            pass
    
    def _save_error_history(self):
        """Save error history to file"""
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            data = {
                'history': self.error_history[-self.MAX_ERROR_HISTORY:],
                'counts': self.error_counts,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.ERROR_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def categorize_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Categorize an error and determine severity"""
        error_str = str(error)
        error_type = type(error).__name__
        
        # Check API errors
        if any(e in error_type or e in error_str for e in self.API_ERRORS):
            if 'insufficient' in error_str.lower() or 'balance' in error_str.lower():
                return ErrorCategory.API_ERROR, ErrorSeverity.MEDIUM
            return ErrorCategory.API_ERROR, ErrorSeverity.MEDIUM
        
        # Check network errors
        if any(e in error_type or e in error_str for e in self.NETWORK_ERRORS):
            return ErrorCategory.NETWORK_ERROR, ErrorSeverity.MEDIUM
        
        # Check rate limit
        if any(e in error_type or e in error_str for e in self.RATE_LIMIT_ERRORS):
            return ErrorCategory.RATE_LIMIT, ErrorSeverity.MEDIUM
        
        # Check auth errors
        if any(e in error_type or e in error_str for e in self.AUTH_ERRORS):
            return ErrorCategory.AUTH_ERROR, ErrorSeverity.CRITICAL
        
        # Check for data errors
        if any(e in error_type for e in ['KeyError', 'ValueError', 'IndexError', 'TypeError']):
            return ErrorCategory.DATA_ERROR, ErrorSeverity.LOW
        
        # Check for logic errors
        if any(e in error_type for e in ['AssertionError', 'RuntimeError']):
            return ErrorCategory.LOGIC_ERROR, ErrorSeverity.HIGH
        
        return ErrorCategory.UNKNOWN, ErrorSeverity.LOW
    
    def handle_error(
        self,
        error: Exception,
        context: Dict = None,
        operation: str = "unknown"
    ) -> ErrorRecord:
        """
        Handle an error: categorize, log, and determine recovery.
        
        Args:
            error: The exception
            context: Additional context information
            operation: Name of the operation that failed
            
        Returns:
            ErrorRecord with handling information
        """
        category, severity = self.categorize_error(error)
        
        record = ErrorRecord(
            error=error,
            category=category,
            severity=severity,
            context={**(context or {}), 'operation': operation}
        )
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"[{category.value}] {operation}: {record.message}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"[{category.value}] {operation}: {record.message}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"[{category.value}] {operation}: {record.message}")
        else:
            logger.debug(f"[{category.value}] {operation}: {record.message}")
        
        # Track error
        self.error_history.append(record.to_dict())
        self.error_counts[category.value] = self.error_counts.get(category.value, 0) + 1
        self.last_errors[category.value] = datetime.now()
        
        # Update circuit breaker
        self._update_circuit_breaker(operation, failed=True)
        
        # Save history
        self._save_error_history()
        
        return record
    
    def _update_circuit_breaker(self, operation: str, failed: bool = True):
        """Update circuit breaker state for an operation"""
        if operation not in self.circuit_breaker:
            self.circuit_breaker[operation] = {
                'failures': 0,
                'last_failure': None,
                'is_open': False,
                'open_until': None
            }
        
        cb = self.circuit_breaker[operation]
        
        if failed:
            cb['failures'] += 1
            cb['last_failure'] = datetime.now()
            
            # Open circuit after 5 consecutive failures
            if cb['failures'] >= 5:
                cb['is_open'] = True
                cb['open_until'] = datetime.now() + timedelta(minutes=5)
                logger.warning(f"Circuit breaker opened for {operation}")
        else:
            # Success - reset failures
            cb['failures'] = 0
            cb['is_open'] = False
            cb['open_until'] = None
    
    def is_circuit_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for an operation"""
        if operation not in self.circuit_breaker:
            return False
        
        cb = self.circuit_breaker[operation]
        
        if cb['is_open']:
            if cb['open_until'] and datetime.now() > cb['open_until']:
                # Circuit has timed out, allow retry
                cb['is_open'] = False
                cb['failures'] = 0
                return False
            return True
        
        return False
    
    def get_retry_delay(
        self,
        category: ErrorCategory,
        attempt: int = 1
    ) -> float:
        """Get retry delay based on error category and attempt number"""
        base_delays = {
            ErrorCategory.RATE_LIMIT: 60,     # 60s for rate limits
            ErrorCategory.NETWORK_ERROR: 5,   # 5s for network
            ErrorCategory.API_ERROR: 10,      # 10s for API
            ErrorCategory.DATA_ERROR: 1,      # 1s for data
            ErrorCategory.UNKNOWN: 5          # 5s default
        }
        
        base = base_delays.get(category, 5)
        # Exponential backoff with cap
        delay = min(base * (2 ** (attempt - 1)), 300)  # Max 5 minutes
        
        return delay
    
    def get_recovery_strategy(self, category: ErrorCategory) -> str:
        """Get recovery strategy for error category"""
        strategies = {
            ErrorCategory.API_ERROR: "retry_with_backoff",
            ErrorCategory.NETWORK_ERROR: "retry_with_backoff",
            ErrorCategory.RATE_LIMIT: "wait_and_retry",
            ErrorCategory.AUTH_ERROR: "stop_and_alert",
            ErrorCategory.DATA_ERROR: "skip_and_continue",
            ErrorCategory.LOGIC_ERROR: "log_and_continue",
            ErrorCategory.UNKNOWN: "log_and_continue"
        }
        return strategies.get(category, "log_and_continue")
    
    def get_error_stats(self) -> Dict:
        """Get error statistics"""
        recent_errors = [
            e for e in self.error_history
            if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            'total_errors': len(self.error_history),
            'last_24h': len(recent_errors),
            'by_category': self.error_counts,
            'circuit_breakers': {
                op: cb['is_open'] for op, cb in self.circuit_breaker.items()
            }
        }
    
    def record_failure(self, operation: str):
        """Record a failure for circuit breaker (convenience method)"""
        self._update_circuit_breaker(operation, failed=True)
    
    def record_success(self, operation: str):
        """Record a success for circuit breaker (convenience method)"""
        self._update_circuit_breaker(operation, failed=False)
    
    def log_error(self, error: Exception, context: Dict = None, operation: str = "unknown"):
        """Log an error with context (convenience method)"""
        return self.handle_error(error, context=context, operation=operation)
    
    def get_recent_errors(self, limit: int = 10) -> list:
        """Get recent errors from history"""
        return self.error_history[-limit:]
    
    def reset_circuit_breakers(self):
        """Reset all circuit breakers"""
        self.circuit_breaker = {}


# Singleton instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create singleton error handler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def with_retry(
    max_retries: int = 3,
    operation_name: str = "operation"
) -> Callable:
    """
    Decorator for automatic retry with error handling.
    
    Usage:
        @with_retry(max_retries=3, operation_name="fetch_price")
        def fetch_price():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            handler = get_error_handler()
            
            # Check circuit breaker
            if handler.is_circuit_open(operation_name):
                logger.warning(f"Circuit open for {operation_name}, skipping")
                return None
            
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    # Success - reset circuit breaker
                    handler._update_circuit_breaker(operation_name, failed=False)
                    return result
                    
                except Exception as e:
                    last_error = e
                    record = handler.handle_error(
                        error=e,
                        context={'attempt': attempt, 'max_retries': max_retries},
                        operation=operation_name
                    )
                    
                    # Don't retry critical errors
                    if record.severity == ErrorSeverity.CRITICAL:
                        raise
                    
                    # Get retry delay
                    if attempt < max_retries:
                        delay = handler.get_retry_delay(record.category, attempt)
                        logger.info(f"Retrying {operation_name} in {delay:.1f}s (attempt {attempt}/{max_retries})")
                        time.sleep(delay)
            
            # All retries exhausted - return None instead of raising
            logger.error(f"All {max_retries} retries exhausted for {operation_name}")
            return None
            
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    default: Any = None,
    operation_name: str = "operation",
    context: Dict = None
) -> Any:
    """
    Safely execute a function with error handling.
    Returns default value on error instead of raising.
    
    Usage:
        result = safe_execute(risky_function, default=[], operation_name="fetch_data")
    """
    handler = get_error_handler()
    
    try:
        return func()
    except Exception as e:
        handler.handle_error(e, context=context, operation=operation_name)
        return default
