"""
REST API - External API for Bot Control and Data Access
Provides endpoints for monitoring, control, and data export.

Phase 6: Features & Functionality
"""
import logging
from flask import Blueprint, jsonify, request
from functools import wraps
from datetime import datetime
from typing import Dict, Any

import config

logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# API key authentication (optional)
API_KEY = getattr(config, 'API_KEY', None)


def require_api_key(f):
    """Decorator to require API key for protected endpoints"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if API_KEY:
            key = request.headers.get('X-API-Key') or request.args.get('api_key')
            if key != API_KEY:
                return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated


# =========================================================================
# STATUS ENDPOINTS
# =========================================================================

@api_bp.route('/status', methods=['GET'])
def get_status():
    """Get bot status"""
    try:
        # Import here to avoid circular imports
        from dashboard import bot_state
        
        return jsonify({
            'status': 'ok',
            'running': bot_state.get('running', False),
            'paused': bot_state.get('paused', False),
            'current_price': bot_state.get('current_price', 0),
            'ai_score': bot_state.get('ai_score', 0),
            'decision': bot_state.get('decision', 'WAITING'),
            'market_regime': bot_state.get('market_regime', 'unknown'),
            'positions': len(bot_state.get('positions', [])),
            'balance_usdt': bot_state.get('balance_usdt', 0),
            'total_value': bot_state.get('total_value', 0),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


# =========================================================================
# TRADING ENDPOINTS
# =========================================================================

@api_bp.route('/positions', methods=['GET'])
def get_positions():
    """Get current positions"""
    try:
        from dashboard import bot_state
        
        positions = bot_state.get('positions', [])
        current_price = bot_state.get('current_price', 0)
        
        position_data = []
        for i, pos in enumerate(positions):
            entry = pos.get('entry_price', 0)
            if entry > 0 and current_price > 0:
                pnl_pct = ((current_price - entry) / entry) * 100
                pnl_dollar = (current_price - entry) * pos.get('quantity', 0)
            else:
                pnl_pct = 0
                pnl_dollar = 0
            
            position_data.append({
                'index': i,
                'entry_price': entry,
                'quantity': pos.get('quantity', 0),
                'entry_time': pos.get('time'),
                'current_price': current_price,
                'pnl_percent': round(pnl_pct, 2),
                'pnl_dollar': round(pnl_dollar, 2),
                'regime': pos.get('regime')
            })
        
        return jsonify({
            'positions': position_data,
            'count': len(position_data),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/trades', methods=['GET'])
def get_trades():
    """Get trade history"""
    try:
        from dashboard import bot_state
        
        limit = request.args.get('limit', 50, type=int)
        trade_type = request.args.get('type')  # BUY, SELL, or None for all
        
        trades = bot_state.get('trade_history', [])
        
        if trade_type:
            trades = [t for t in trades if t.get('type') == trade_type.upper()]
        
        # Sort by time descending and limit
        trades = sorted(trades, key=lambda x: x.get('id', 0), reverse=True)[:limit]
        
        return jsonify({
            'trades': trades,
            'count': len(trades),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================================================================
# PERFORMANCE ENDPOINTS
# =========================================================================

@api_bp.route('/performance', methods=['GET'])
def get_performance():
    """Get performance metrics"""
    try:
        from dashboard import bot_state
        from analytics.performance_tracker import get_performance_tracker
        
        tracker = get_performance_tracker()
        metrics = tracker.get_full_metrics()
        
        # Add bot state metrics
        metrics['bot'] = {
            'total_profit': bot_state.get('total_profit', 0),
            'total_trades': bot_state.get('total_trades', 0),
            'win_rate': bot_state.get('win_rate', 0),
            'winning_trades': bot_state.get('winning_trades', 0),
            'losing_trades': bot_state.get('losing_trades', 0)
        }
        
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/performance/equity', methods=['GET'])
def get_equity_curve():
    """Get equity curve data"""
    try:
        from analytics.performance_tracker import get_performance_tracker
        
        limit = request.args.get('limit', 100, type=int)
        tracker = get_performance_tracker()
        
        return jsonify({
            'equity_curve': tracker.get_equity_curve_data(limit),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================================================================
# RISK ENDPOINTS
# =========================================================================

@api_bp.route('/risk', methods=['GET'])
def get_risk_status():
    """Get risk management status"""
    try:
        from core.risk_manager import get_risk_manager
        
        risk_mgr = get_risk_manager()
        status = risk_mgr.get_status()
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================================================================
# AI/ML ENDPOINTS
# =========================================================================

@api_bp.route('/ai/indicators', methods=['GET'])
def get_indicator_accuracy():
    """Get indicator accuracy data"""
    try:
        from dashboard import ai_engine
        
        if ai_engine:
            accuracy = ai_engine.indicator_accuracy
            samples = getattr(ai_engine, 'indicator_samples', {})
            
            return jsonify({
                'accuracy': accuracy,
                'samples': samples,
                'weights': ai_engine.weights,
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({'error': 'AI engine not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/ai/decision', methods=['GET'])
def get_ai_decision():
    """Get current AI decision details"""
    try:
        from dashboard import bot_state
        
        return jsonify({
            'score': bot_state.get('ai_score', 0),
            'decision': bot_state.get('decision', 'WAITING'),
            'reasons': bot_state.get('decision_reasons', []),
            'confluence': bot_state.get('confluence', {}),
            'confidence': bot_state.get('confidence', {}),
            'regime': bot_state.get('market_regime', 'unknown'),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/ml/accuracy', methods=['GET'])
def get_ml_accuracy():
    """Get ML model accuracy"""
    try:
        from ai.ml_predictor import MLPredictor
        
        ml = MLPredictor()
        accuracy = ml.get_accuracy()
        should_retrain, reason = ml.should_retrain()
        
        return jsonify({
            'accuracy': accuracy,
            'should_retrain': should_retrain,
            'retrain_reason': reason,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================================================================
# CONTROL ENDPOINTS (Protected)
# =========================================================================

@api_bp.route('/control/pause', methods=['POST'])
@require_api_key
def pause_trading():
    """Pause trading"""
    try:
        from dashboard import bot_state
        
        bot_state['paused'] = True
        
        return jsonify({
            'status': 'ok',
            'paused': True,
            'message': 'Trading paused',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/control/resume', methods=['POST'])
@require_api_key
def resume_trading():
    """Resume trading"""
    try:
        from dashboard import bot_state
        
        bot_state['paused'] = False
        
        return jsonify({
            'status': 'ok',
            'paused': False,
            'message': 'Trading resumed',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/control/config', methods=['GET', 'POST'])
@require_api_key
def manage_config():
    """Get or update configuration"""
    if request.method == 'GET':
        return jsonify({
            'buy_threshold': getattr(config, 'BUY_THRESHOLD', 0.15),
            'sell_threshold': getattr(config, 'SELL_THRESHOLD', -0.25),
            'stop_loss': getattr(config, 'STOP_LOSS', 0.0075) * 100,
            'profit_target': getattr(config, 'PROFIT_TARGET', 0.015) * 100,
            'trade_amount': getattr(config, 'TRADE_AMOUNT_USDT', 40),
            'max_positions': getattr(config, 'MAX_POSITIONS', 2),
            'paper_mode': getattr(config, 'PAPER_TRADING_MODE', False),
            'timestamp': datetime.now().isoformat()
        })
    
    # POST - update config (limited settings)
    try:
        data = request.get_json() or {}
        
        updates = []
        if 'buy_threshold' in data:
            config.BUY_THRESHOLD = float(data['buy_threshold'])
            updates.append('buy_threshold')
        if 'sell_threshold' in data:
            config.SELL_THRESHOLD = float(data['sell_threshold'])
            updates.append('sell_threshold')
        if 'trade_amount' in data:
            config.TRADE_AMOUNT_USDT = float(data['trade_amount'])
            updates.append('trade_amount')
        
        return jsonify({
            'status': 'ok',
            'updated': updates,
            'message': f'Updated {len(updates)} settings',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================================================================
# PAPER TRADING ENDPOINTS
# =========================================================================

@api_bp.route('/paper/status', methods=['GET'])
def get_paper_status():
    """Get paper trading status"""
    try:
        from core.paper_trader import get_paper_trader, is_paper_mode
        
        if not is_paper_mode():
            return jsonify({
                'enabled': False,
                'message': 'Paper trading not enabled'
            })
        
        trader = get_paper_trader()
        performance = trader.get_performance()
        positions = trader.check_positions(trader.last_price)
        
        return jsonify({
            'enabled': True,
            'performance': performance,
            'positions': positions,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/paper/reset', methods=['POST'])
@require_api_key
def reset_paper_trading():
    """Reset paper trading account"""
    try:
        from core.paper_trader import get_paper_trader
        
        initial_balance = request.get_json().get('initial_balance', 1000) if request.is_json else 1000
        
        trader = get_paper_trader()
        trader.reset(initial_balance)
        
        return jsonify({
            'status': 'ok',
            'message': f'Paper trading reset with ${initial_balance:.2f}',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================================================================
# DATABASE ENDPOINTS
# =========================================================================

@api_bp.route('/db/stats', methods=['GET'])
def get_db_stats():
    """Get database statistics"""
    try:
        from core.database import get_database
        
        db = get_database()
        trade_stats = db.get_trade_stats(days=30)
        ml_accuracy = db.get_ml_accuracy(days=7)
        
        return jsonify({
            'trade_stats_30d': trade_stats,
            'ml_accuracy_7d': ml_accuracy,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    try:
        from core.cache import get_all_cache_stats
        
        return jsonify(get_all_cache_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================================================================
# WEBHOOK ENDPOINT
# =========================================================================

@api_bp.route('/webhook', methods=['POST'])
@require_api_key
def webhook_handler():
    """Handle external webhooks (e.g., TradingView alerts)"""
    try:
        data = request.get_json() or {}
        
        action = data.get('action', '').upper()
        source = data.get('source', 'unknown')
        
        logger.info(f"Webhook received: action={action}, source={source}")
        
        if action == 'BUY':
            # Could trigger a buy signal boost
            return jsonify({
                'status': 'ok',
                'action': 'buy_signal_received',
                'message': 'External buy signal noted'
            })
        elif action == 'SELL':
            # Could trigger a sell signal boost
            return jsonify({
                'status': 'ok',
                'action': 'sell_signal_received',
                'message': 'External sell signal noted'
            })
        elif action == 'ALERT':
            return jsonify({
                'status': 'ok',
                'action': 'alert_received',
                'message': data.get('message', 'Alert received')
            })
        else:
            return jsonify({
                'status': 'ok',
                'action': 'unknown',
                'message': 'Webhook received but no action taken'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def init_api(app):
    """Initialize API with Flask app"""
    app.register_blueprint(api_bp)
    logger.info("REST API initialized at /api/v1")
