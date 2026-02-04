"""
Alert System - Configurable Alerts for Trading Events
Provides real-time alerts for trades, performance, and system events.

Phase 5: Monitoring & Analytics
"""
import os
import json
import logging
from typing import Dict, Optional, List, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

import config

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"


class AlertType(Enum):
    """Types of alerts"""
    TRADE_EXECUTED = "trade_executed"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    DRAWDOWN = "drawdown"
    WIN_STREAK = "win_streak"
    LOSS_STREAK = "loss_streak"
    DAILY_LIMIT = "daily_limit"
    ML_ACCURACY = "ml_accuracy"
    SYSTEM_ERROR = "system_error"
    POSITION_RISK = "position_risk"
    PRICE_ALERT = "price_alert"
    CUSTOM = "custom"


class Alert:
    """Individual alert"""
    
    def __init__(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message: str,
        data: Dict = None
    ):
        self.id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.alert_type = alert_type
        self.level = level
        self.title = title
        self.message = message
        self.data = data or {}
        self.timestamp = datetime.now()
        self.acknowledged = False
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.alert_type.value,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }


class AlertRule:
    """Configurable alert rule"""
    
    def __init__(
        self,
        name: str,
        alert_type: AlertType,
        condition: Callable[[Dict], bool],
        level: AlertLevel = AlertLevel.INFO,
        cooldown_minutes: int = 5,
        enabled: bool = True
    ):
        self.name = name
        self.alert_type = alert_type
        self.condition = condition
        self.level = level
        self.cooldown_minutes = cooldown_minutes
        self.enabled = enabled
        self.last_triggered = None
    
    def check(self, context: Dict) -> Optional[str]:
        """
        Check if rule should trigger.
        Returns alert message if triggered, None otherwise.
        """
        if not self.enabled:
            return None
        
        # Check cooldown
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return None
        
        try:
            if self.condition(context):
                self.last_triggered = datetime.now()
                return context.get('message', f"Rule {self.name} triggered")
        except Exception as e:
            logger.warning(f"Error checking rule {self.name}: {e}")
        
        return None


class AlertSystem:
    """
    Centralized alert system for trading bot.
    Manages alert rules, notifications, and alert history.
    """
    
    CONFIG_FILE = os.path.join(config.DATA_DIR, "alert_config.json")
    HISTORY_FILE = os.path.join(config.DATA_DIR, "alert_history.json")
    MAX_HISTORY = 500
    
    def __init__(self):
        """Initialize alert system"""
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: deque = deque(maxlen=self.MAX_HISTORY)
        self.handlers: List[Callable[[Alert], None]] = []
        
        # Alert thresholds (configurable)
        self.thresholds = {
            'drawdown_warning': 5.0,      # 5% drawdown
            'drawdown_critical': 10.0,    # 10% drawdown
            'loss_streak_warning': 3,     # 3 consecutive losses
            'loss_streak_critical': 5,    # 5 consecutive losses
            'win_streak_notify': 3,       # 3 consecutive wins
            'ml_accuracy_low': 50.0,      # 50% accuracy
            'daily_loss_warning': 2.0,    # 2% daily loss
            'position_age_hours': 4,      # 4 hours
        }
        
        self._load_config()
        self._setup_default_rules()
    
    def _load_config(self):
        """Load alert configuration"""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.thresholds.update(data.get('thresholds', {}))
        except Exception as e:
            logger.warning(f"Could not load alert config: {e}")
    
    def _save_config(self):
        """Save alert configuration"""
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            data = {
                'thresholds': self.thresholds,
                'rules_enabled': {name: rule.enabled for name, rule in self.rules.items()},
                'last_updated': datetime.now().isoformat()
            }
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save alert config: {e}")
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        
        # Drawdown warning
        self.add_rule(AlertRule(
            name="drawdown_warning",
            alert_type=AlertType.DRAWDOWN,
            condition=lambda ctx: ctx.get('drawdown_pct', 0) >= self.thresholds['drawdown_warning'],
            level=AlertLevel.WARNING,
            cooldown_minutes=30
        ))
        
        # Drawdown critical
        self.add_rule(AlertRule(
            name="drawdown_critical",
            alert_type=AlertType.DRAWDOWN,
            condition=lambda ctx: ctx.get('drawdown_pct', 0) >= self.thresholds['drawdown_critical'],
            level=AlertLevel.CRITICAL,
            cooldown_minutes=60
        ))
        
        # Loss streak warning
        self.add_rule(AlertRule(
            name="loss_streak_warning",
            alert_type=AlertType.LOSS_STREAK,
            condition=lambda ctx: ctx.get('consecutive_losses', 0) >= self.thresholds['loss_streak_warning'],
            level=AlertLevel.WARNING,
            cooldown_minutes=15
        ))
        
        # Loss streak critical
        self.add_rule(AlertRule(
            name="loss_streak_critical",
            alert_type=AlertType.LOSS_STREAK,
            condition=lambda ctx: ctx.get('consecutive_losses', 0) >= self.thresholds['loss_streak_critical'],
            level=AlertLevel.CRITICAL,
            cooldown_minutes=30
        ))
        
        # Win streak notification
        self.add_rule(AlertRule(
            name="win_streak",
            alert_type=AlertType.WIN_STREAK,
            condition=lambda ctx: ctx.get('consecutive_wins', 0) >= self.thresholds['win_streak_notify'],
            level=AlertLevel.SUCCESS,
            cooldown_minutes=30
        ))
        
        # ML accuracy low
        self.add_rule(AlertRule(
            name="ml_accuracy_low",
            alert_type=AlertType.ML_ACCURACY,
            condition=lambda ctx: ctx.get('ml_accuracy', 100) < self.thresholds['ml_accuracy_low'],
            level=AlertLevel.WARNING,
            cooldown_minutes=60
        ))
        
        # Daily loss warning
        self.add_rule(AlertRule(
            name="daily_loss_warning",
            alert_type=AlertType.DAILY_LIMIT,
            condition=lambda ctx: ctx.get('daily_loss_pct', 0) >= self.thresholds['daily_loss_warning'],
            level=AlertLevel.WARNING,
            cooldown_minutes=60
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules[rule.name] = rule
    
    def remove_rule(self, name: str):
        """Remove an alert rule"""
        if name in self.rules:
            del self.rules[name]
    
    def enable_rule(self, name: str, enabled: bool = True):
        """Enable or disable a rule"""
        if name in self.rules:
            self.rules[name].enabled = enabled
            self._save_config()
    
    def set_threshold(self, name: str, value: float):
        """Set an alert threshold"""
        self.thresholds[name] = value
        self._save_config()
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler (e.g., Telegram notification)"""
        self.handlers.append(handler)
    
    def check_rules(self, context: Dict):
        """
        Check all rules against current context.
        Triggers alerts for matching rules.
        """
        for name, rule in self.rules.items():
            message = rule.check(context)
            if message:
                self.trigger_alert(
                    alert_type=rule.alert_type,
                    level=rule.level,
                    title=name.replace('_', ' ').title(),
                    message=message,
                    data=context
                )
    
    def trigger_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message: str,
        data: Dict = None
    ):
        """Trigger a new alert"""
        alert = Alert(alert_type, level, title, message, data)
        self.alerts.append(alert)
        
        # Log alert
        log_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.SUCCESS: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.error
        }.get(level, logger.info)
        
        log_method(f"[ALERT] {title}: {message}")
        
        # Call handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        return alert
    
    def trade_alert(
        self,
        trade_type: str,
        price: float,
        quantity: float,
        pnl: float = None,
        pnl_pct: float = None
    ):
        """Quick alert for trade execution"""
        if trade_type.upper() == "BUY":
            self.trigger_alert(
                AlertType.TRADE_EXECUTED,
                AlertLevel.INFO,
                "Trade Executed",
                f"BUY {quantity:.8f} BTC @ ${price:,.2f}"
            )
        elif trade_type.upper() == "SELL":
            if pnl is not None and pnl > 0:
                self.trigger_alert(
                    AlertType.PROFIT_TARGET,
                    AlertLevel.SUCCESS,
                    "Profit Taken",
                    f"SELL @ ${price:,.2f} | P/L: ${pnl:+.2f} ({pnl_pct:+.2f}%)"
                )
            elif pnl is not None and pnl < 0:
                self.trigger_alert(
                    AlertType.STOP_LOSS,
                    AlertLevel.WARNING,
                    "Stop Loss Hit",
                    f"SELL @ ${price:,.2f} | P/L: ${pnl:+.2f} ({pnl_pct:+.2f}%)"
                )
    
    def get_recent_alerts(self, limit: int = 50, level: AlertLevel = None) -> List[Dict]:
        """Get recent alerts"""
        alerts = list(self.alerts)
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        alerts = sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]
        
        return [a.to_dict() for a in alerts]
    
    def get_unacknowledged_count(self) -> int:
        """Get count of unacknowledged alerts"""
        return sum(1 for a in self.alerts if not a.acknowledged)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def acknowledge_all(self):
        """Acknowledge all alerts"""
        for alert in self.alerts:
            alert.acknowledged = True
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent = [a for a in self.alerts if a.timestamp > last_24h]
        
        by_level = {}
        by_type = {}
        for alert in recent:
            by_level[alert.level.value] = by_level.get(alert.level.value, 0) + 1
            by_type[alert.alert_type.value] = by_type.get(alert.alert_type.value, 0) + 1
        
        return {
            'total': len(self.alerts),
            'last_24h': len(recent),
            'unacknowledged': self.get_unacknowledged_count(),
            'by_level': by_level,
            'by_type': by_type
        }


# Singleton instance
_alert_system: Optional[AlertSystem] = None


def get_alert_system() -> AlertSystem:
    """Get or create singleton alert system"""
    global _alert_system
    if _alert_system is None:
        _alert_system = AlertSystem()
    return _alert_system


def setup_telegram_alerts():
    """Setup Telegram as alert handler"""
    try:
        from notifications.telegram_notifier import get_notifier
        
        notifier = get_notifier()
        if notifier:
            def telegram_handler(alert: Alert):
                # Format message based on level
                emoji = {
                    AlertLevel.INFO: "ℹ️",
                    AlertLevel.SUCCESS: "✅",
                    AlertLevel.WARNING: "⚠️",
                    AlertLevel.CRITICAL: "🚨"
                }.get(alert.level, "📢")
                
                message = f"{emoji} *{alert.title}*\n{alert.message}"
                notifier.send_message(message)
            
            alert_system = get_alert_system()
            alert_system.add_handler(telegram_handler)
            logger.info("Telegram alerts configured")
    except Exception as e:
        logger.warning(f"Could not setup Telegram alerts: {e}")
