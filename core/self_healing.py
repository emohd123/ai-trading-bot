"""
Self-Healing Bot System - Auto-fix common errors
Runs periodic health checks and automatically resolves issues
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

import config

logger = logging.getLogger(__name__)


class SelfHealer:
    """
    Automatic error detection and recovery system
    """
    
    def __init__(self, client, bot_state: Dict, notifier=None):
        """
        Initialize self-healer
        
        Args:
            client: BinanceClient instance
            bot_state: Reference to bot state dict
            notifier: TelegramNotifier instance (optional)
        """
        self.client = client
        self.bot_state = bot_state
        self.notifier = notifier
        
        # Track issues and fixes
        self.issues_detected = []
        self.fixes_applied = []
        self.last_health_check = None
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # Health check interval (seconds)
        self.health_check_interval = 300  # 5 minutes
        
    def log_issue(self, issue: str, severity: str = "warning"):
        """Log an issue that was detected"""
        self.issues_detected.append({
            "time": datetime.now().isoformat(),
            "issue": issue,
            "severity": severity
        })
        if severity == "error":
            logger.error("HEAL: Issue detected: %s", issue)
        else:
            logger.warning("HEAL: Issue detected: %s", issue)
        
    def log_fix(self, fix: str):
        """Log a fix that was applied"""
        self.fixes_applied.append({
            "time": datetime.now().isoformat(),
            "fix": fix
        })
        logger.info("HEAL: Fix applied: %s", fix)
        
        # Notify via Telegram if available
        if self.notifier:
            try:
                self.notifier.send_message(f"🔧 AUTO-FIX: {fix}")
            except:
                pass
    
    def run_health_check(self) -> Dict:
        """
        Run comprehensive health check and auto-fix issues
        
        Returns:
            Dict with health status and any fixes applied
        """
        self.last_health_check = datetime.now()
        fixes = []
        issues = []
        
        logger.info("Running health check...")
        
        # 1. Check Balance Sync
        balance_fix = self.check_and_fix_balance_sync()
        if balance_fix:
            fixes.append(balance_fix)
            
        # 2. Check Position Consistency
        position_fix = self.check_and_fix_position_consistency()
        if position_fix:
            fixes.append(position_fix)
            
        # 3. Check for Stale Positions
        stale_fix = self.check_and_fix_stale_positions()
        if stale_fix:
            fixes.append(stale_fix)
            
        # 4. Check API Connection
        api_fix = self.check_and_fix_api_connection()
        if api_fix:
            fixes.append(api_fix)
            
        # 5. Check for Ghost Positions (position tracked but no BTC)
        ghost_fix = self.check_and_fix_ghost_positions()
        if ghost_fix:
            fixes.append(ghost_fix)
            
        # 6. Check for Orphan BTC (BTC held but no position tracked)
        orphan_fix = self.check_and_fix_orphan_btc()
        if orphan_fix:
            fixes.append(orphan_fix)
        
        status = {
            "healthy": len(fixes) == 0,
            "check_time": self.last_health_check.isoformat(),
            "fixes_applied": fixes,
            "issues_found": len(fixes)
        }
        
        if fixes:
            logger.info("Health check complete: %d fixes applied", len(fixes))
        else:
            logger.info("Health check complete: All systems healthy")
            
        return status
    
    def check_and_fix_balance_sync(self) -> Optional[str]:
        """Check if tracked balance matches Binance and fix if needed"""
        try:
            # Get actual balances from Binance
            actual_usdt = self.client.get_balance("USDT")
            actual_btc = self.client.get_balance("BTC")
            
            # Get tracked balances
            tracked_usdt = self.bot_state.get("balance_usdt", 0)
            tracked_btc = self.bot_state.get("balance_btc", 0)
            
            # Check for significant mismatch (more than 1%)
            usdt_diff = abs(actual_usdt - tracked_usdt) / max(tracked_usdt, 1) * 100
            
            if usdt_diff > 5:  # More than 5% difference
                self.log_issue(f"USDT balance mismatch: tracked={tracked_usdt:.2f}, actual={actual_usdt:.2f}")
                self.bot_state["balance_usdt"] = actual_usdt
                self.bot_state["balance_btc"] = actual_btc
                fix = f"Synced balances: USDT={actual_usdt:.2f}, BTC={actual_btc:.8f}"
                self.log_fix(fix)
                return fix
                
        except Exception as e:
            self.log_issue(f"Balance sync check failed: {e}", "error")
            
        return None
    
    def check_and_fix_position_consistency(self) -> Optional[str]:
        """Check if position tracking is consistent and fix if needed"""
        try:
            position = self.bot_state.get("position")
            positions = self.bot_state.get("positions", [])
            
            # Check if position and positions list are in sync
            if position and not positions:
                # Legacy position exists but positions list is empty
                self.log_issue("Position exists but positions list is empty")
                self.bot_state["positions"] = [position]
                fix = "Synced positions list with legacy position"
                self.log_fix(fix)
                return fix
                
            if not position and positions:
                # Positions list has items but legacy position is None
                self.log_issue("Positions list has items but legacy position is None")
                self.bot_state["position"] = positions[0]
                fix = "Synced legacy position with positions list"
                self.log_fix(fix)
                return fix
                
        except Exception as e:
            self.log_issue(f"Position consistency check failed: {e}", "error")
            
        return None
    
    def check_and_fix_stale_positions(self) -> Optional[str]:
        """Check for positions that are too old and should be reviewed"""
        try:
            positions = self.bot_state.get("positions", [])
            
            for i, pos in enumerate(positions):
                entry_time_str = pos.get("entry_time")
                if entry_time_str:
                    try:
                        entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                        age_hours = (datetime.now() - entry_time).total_seconds() / 3600
                        
                        # If position is older than 24 hours, log warning
                        if age_hours > 24:
                            self.log_issue(f"Position #{i+1} is {age_hours:.1f} hours old - consider manual review")
                            
                    except:
                        pass
                        
        except Exception as e:
            self.log_issue(f"Stale position check failed: {e}", "error")
            
        return None
    
    def check_and_fix_api_connection(self) -> Optional[str]:
        """Check API connection and reconnect if needed"""
        try:
            # Try to get current price as connection test
            price = self.client.get_current_price()
            if price and price > 0:
                return None  # Connection is fine
                
            self.log_issue("API connection test failed - price returned 0 or None")
            
            # Try to reconnect
            for attempt in range(self.max_retries):
                time.sleep(self.retry_delay)
                try:
                    self.client._init_client()
                    price = self.client.get_current_price()
                    if price and price > 0:
                        fix = f"Reconnected to Binance API after {attempt + 1} attempts"
                        self.log_fix(fix)
                        return fix
                except:
                    continue
                    
            self.log_issue("Failed to reconnect to Binance API after max retries", "error")
            
        except Exception as e:
            self.log_issue(f"API connection check failed: {e}", "error")
            
        return None
    
    def check_and_fix_ghost_positions(self) -> Optional[str]:
        """Check for ghost positions (tracked but no actual BTC)"""
        try:
            position = self.bot_state.get("position")
            actual_btc = self.client.get_balance("BTC")
            
            # Minimum BTC that's considered "dust" (not a real position)
            min_btc = 0.00001
            
            if position and actual_btc < min_btc:
                self.log_issue(f"Ghost position detected: tracked {position.get('quantity', 0)} BTC but only {actual_btc} on Binance")
                
                # Clear the ghost position
                self.bot_state["position"] = None
                self.bot_state["positions"] = []
                
                fix = f"Cleared ghost position (no BTC on exchange)"
                self.log_fix(fix)
                return fix
                
        except Exception as e:
            self.log_issue(f"Ghost position check failed: {e}", "error")
            
        return None
    
    def check_and_fix_orphan_btc(self) -> Optional[str]:
        """Check for orphan BTC (BTC held but no position tracked)"""
        try:
            position = self.bot_state.get("position")
            actual_btc = self.client.get_balance("BTC")
            
            # Minimum BTC to consider as a real position
            min_btc = 0.0001  # About $8 at $80k BTC
            
            if not position and actual_btc >= min_btc:
                self.log_issue(f"Orphan BTC detected: {actual_btc} BTC on Binance but no position tracked")
                
                # Try to recover position from trade history
                fix = self.recover_position_from_history(actual_btc)
                if fix:
                    return fix
                else:
                    # Create a placeholder position with current price as entry
                    current_price = self.client.get_current_price()
                    if current_price:
                        self.bot_state["position"] = {
                            "entry_price": current_price,
                            "quantity": actual_btc,
                            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "amount_usdt": actual_btc * current_price,
                            "regime": "unknown",
                            "indicator_scores": {},
                            "recovered": True
                        }
                        self.bot_state["positions"] = [self.bot_state["position"]]
                        
                        fix = f"Created recovery position for {actual_btc:.8f} BTC at ${current_price:,.2f}"
                        self.log_fix(fix)
                        return fix
                        
        except Exception as e:
            self.log_issue(f"Orphan BTC check failed: {e}", "error")
            
        return None
    
    def recover_position_from_history(self, btc_amount: float) -> Optional[str]:
        """Try to recover position details from trade history"""
        try:
            # Load trade history
            history_file = os.path.join(config.DATA_DIR, "trade_history.json")
            if not os.path.exists(history_file):
                return None
                
            with open(history_file, 'r') as f:
                trades = json.load(f)
                
            # Sort by ID descending (most recent first)
            trades.sort(key=lambda x: x.get('id', 0), reverse=True)
            
            # Find the most recent BUY that doesn't have a matching SELL
            buy_count = 0
            sell_count = 0
            
            for trade in trades:
                if trade.get('type') == 'BUY':
                    buy_count += 1
                    if buy_count > sell_count:
                        # Found unmatched BUY
                        self.bot_state["position"] = {
                            "entry_price": trade.get('price', 0),
                            "quantity": btc_amount,  # Use actual balance
                            "entry_time": trade.get('time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            "amount_usdt": trade.get('amount', 0),
                            "regime": trade.get('regime', 'unknown'),
                            "indicator_scores": {},
                            "recovered": True
                        }
                        self.bot_state["positions"] = [self.bot_state["position"]]
                        
                        fix = f"Recovered position from trade #{trade.get('id')}: entry ${trade.get('price'):,.2f}"
                        self.log_fix(fix)
                        return fix
                elif trade.get('type') == 'SELL':
                    sell_count += 1
                    
        except Exception as e:
            self.log_issue(f"Position recovery failed: {e}", "error")
            
        return None
    
    def fix_sell_quantity(self, requested_qty: float) -> float:
        """
        Fix sell quantity to match actual balance
        
        Args:
            requested_qty: The quantity the bot wants to sell
            
        Returns:
            Corrected quantity that matches actual balance
        """
        try:
            actual_btc = self.client.get_balance("BTC")
            
            if requested_qty > actual_btc:
                self.log_issue(f"Sell quantity mismatch: requested={requested_qty}, actual={actual_btc}")
                self.log_fix(f"Adjusted sell quantity from {requested_qty} to {actual_btc}")
                return actual_btc
                
            return requested_qty
            
        except Exception as e:
            self.log_issue(f"Fix sell quantity failed: {e}", "error")
            return requested_qty
    
    def should_run_health_check(self) -> bool:
        """Check if it's time to run a health check"""
        if self.last_health_check is None:
            return True
            
        elapsed = (datetime.now() - self.last_health_check).total_seconds()
        return elapsed >= self.health_check_interval
    
    def retry_operation(self, operation, *args, **kwargs):
        """
        Retry an operation with exponential backoff
        
        Args:
            operation: Function to retry
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation or None if all retries failed
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = operation(*args, **kwargs)
                if attempt > 0:
                    self.log_fix(f"Operation succeeded after {attempt + 1} attempts")
                return result
            except Exception as e:
                last_error = e
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                self.log_issue(f"Operation failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(delay)
                
        self.log_issue(f"Operation failed after {self.max_retries} attempts: {last_error}", "error")
        return None
    
    def get_health_report(self) -> Dict:
        """Get a summary of recent health issues and fixes"""
        return {
            "last_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "recent_issues": self.issues_detected[-10:],  # Last 10 issues
            "recent_fixes": self.fixes_applied[-10:],     # Last 10 fixes
            "total_issues": len(self.issues_detected),
            "total_fixes": len(self.fixes_applied)
        }


# Standalone function for quick fixes
def quick_fix_balance_mismatch(client, bot_state: Dict) -> bool:
    """Quick fix for balance mismatch issues"""
    try:
        actual_btc = client.get_balance("BTC")
        position = bot_state.get("position")
        
        if position:
            tracked_qty = position.get("quantity", 0)
            if abs(tracked_qty - actual_btc) > 0.00000001:
                logger.info("QUICK-FIX: Adjusting position quantity: %s -> %s", tracked_qty, actual_btc)
                position["quantity"] = actual_btc
                
                # Also update in positions list
                positions = bot_state.get("positions", [])
                for pos in positions:
                    if pos.get("entry_price") == position.get("entry_price"):
                        pos["quantity"] = actual_btc
                        
                return True
                
    except Exception as e:
        logger.error("QUICK-FIX: Error: %s", e)
        
    return False
