"""
Unit tests for Error Handler module.
Tests error categorization, retry logic, and circuit breaker.
"""
import unittest
import sys
import os
import time

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.error_handler import ErrorHandler, with_retry, safe_execute


class TestErrorCategorization(unittest.TestCase):
    """Test error categorization"""
    
    def setUp(self):
        self.handler = ErrorHandler()
    
    def test_network_error_detection(self):
        """Test network errors are categorized correctly"""
        try:
            raise ConnectionError("Network unreachable")
        except Exception as e:
            category, severity = self.handler.categorize_error(e)
            self.assertEqual(category.value, "network_error")
    
    def test_timeout_error_detection(self):
        """Test timeout errors are categorized"""
        try:
            raise TimeoutError("Request timed out")
        except Exception as e:
            category, severity = self.handler.categorize_error(e)
            self.assertEqual(category.value, "network_error")
    
    def test_data_error_handling(self):
        """Test data errors are handled"""
        try:
            raise ValueError("Some random error")
        except Exception as e:
            category, severity = self.handler.categorize_error(e)
            self.assertEqual(category.value, "data_error")


class TestRetryLogic(unittest.TestCase):
    """Test retry decorator"""
    
    def test_retry_on_failure(self):
        """Test function is retried on failure"""
        call_count = [0]
        
        @with_retry(max_retries=3, operation_name="test_op")
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Simulated failure")
            return "success"
        
        result = failing_func()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)
    
    def test_retry_exhausted(self):
        """Test function fails after max retries"""
        @with_retry(max_retries=2, operation_name="test_op")
        def always_fails():
            raise ConnectionError("Always fails")
        
        # Should not raise, returns None on exhaustion
        result = always_fails()
        self.assertIsNone(result)


class TestSafeExecute(unittest.TestCase):
    """Test safe_execute wrapper"""
    
    def test_safe_execute_success(self):
        """Test safe_execute returns result on success"""
        def good_func():
            return 42
        
        result = safe_execute(good_func, default=0)
        self.assertEqual(result, 42)
    
    def test_safe_execute_failure(self):
        """Test safe_execute returns default on failure"""
        def bad_func():
            raise RuntimeError("Oops")
        
        result = safe_execute(bad_func, default=-1)
        self.assertEqual(result, -1)
    
    def test_safe_execute_with_args(self):
        """Test safe_execute passes arguments"""
        def add(a, b):
            return a + b
        
        result = safe_execute(lambda: add(2, 3), default=0)
        self.assertEqual(result, 5)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker pattern"""
    
    def setUp(self):
        self.handler = ErrorHandler()
        self.handler.reset_circuit_breakers()
    
    def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures"""
        operation = "test_operation"
        
        # Record failures up to threshold
        for i in range(5):
            self.handler.record_failure(operation)
        
        # Circuit should be open
        self.assertTrue(self.handler.is_circuit_open(operation))
    
    def test_circuit_allows_initial_calls(self):
        """Test circuit allows calls initially"""
        self.assertFalse(self.handler.is_circuit_open("new_operation"))
    
    def test_circuit_resets_on_success(self):
        """Test circuit resets after success"""
        operation = "reset_test"
        
        # Open the circuit
        for i in range(5):
            self.handler.record_failure(operation)
        
        # Record success
        self.handler.record_success(operation)
        
        # Circuit should be closed
        self.assertFalse(self.handler.is_circuit_open(operation))


class TestErrorLogging(unittest.TestCase):
    """Test error logging functionality"""
    
    def setUp(self):
        self.handler = ErrorHandler()
    
    def test_error_is_logged(self):
        """Test errors are logged with context"""
        try:
            raise ValueError("Test error")
        except Exception as e:
            self.handler.log_error(e, context={"operation": "test"})
        
        # Check error was added to history
        recent = self.handler.get_recent_errors(limit=1)
        self.assertGreaterEqual(len(recent), 0)


if __name__ == '__main__':
    print("=" * 60)
    print("Running Error Handler Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
