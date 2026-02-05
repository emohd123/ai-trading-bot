"""
AI Chat Assistant for Trading Bot
Provides an AI-powered chat interface that can:
- Answer questions about the bot
- View and analyze bot state
- Execute safe commands
- Suggest and apply code changes
"""

import os
import json
import logging
import subprocess
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import config

logger = logging.getLogger(__name__)

# Try to import AI libraries
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class ChatAssistant:
    """AI-powered chat assistant for the trading bot"""
    
    def __init__(self):
        self.provider = None
        self.client = None
        self.conversation_history: List[Dict] = []
        self.max_history = 20  # Keep last 20 messages for context
        
        # Initialize AI client
        self._init_client()
        
        # Available commands the AI can execute
        self.available_commands = {
            "reset_risk": "Reset risk counters (consecutive losses, daily losses)",
            "start_bot": "Start the trading bot",
            "stop_bot": "Stop the trading bot",
            "pause_bot": "Pause trading (monitor only)",
            "resume_bot": "Resume trading",
            "get_state": "Get current bot state",
            "get_performance": "Get performance metrics",
            "get_logs": "Get recent activity logs",
            "sync_balance": "Sync balance with exchange",
            "update_code": "Pull latest code from GitHub",
            "restart_bot": "Update code and restart bot",
            "read_file": "Read a file from the codebase",
            "edit_file": "Edit a file in the codebase",
            "list_files": "List files in a directory",
        }
    
    def _init_client(self):
        """Initialize AI client (Anthropic Claude or OpenAI)"""
        # Try Anthropic first
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and HAS_ANTHROPIC:
            try:
                self.client = anthropic.Anthropic(api_key=anthropic_key)
                self.provider = "anthropic"
                logger.info("Chat assistant initialized with Anthropic Claude")
                return
            except Exception as e:
                logger.warning(f"Failed to init Anthropic: {e}")
        
        # Try OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and HAS_OPENAI:
            try:
                self.client = openai.OpenAI(api_key=openai_key)
                self.provider = "openai"
                logger.info("Chat assistant initialized with OpenAI")
                return
            except Exception as e:
                logger.warning(f"Failed to init OpenAI: {e}")
        
        logger.warning("No AI provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
    
    @property
    def is_available(self) -> bool:
        """Check if AI chat is available"""
        return self.client is not None
    
    def get_system_prompt(self, bot_state: Dict) -> str:
        """Generate system prompt with current bot context"""
        return f"""You are an AI assistant for a cryptocurrency trading bot. You help the user manage and understand their bot.

## Current Bot State
- Running: {bot_state.get('running', False)}
- Symbol: {bot_state.get('current_symbol', 'Unknown')}
- Current Price: ${bot_state.get('current_price', 0):.2f}
- AI Score: {bot_state.get('ai_score', 0):.2f}
- Decision: {bot_state.get('decision', 'Unknown')}
- Market Regime: {bot_state.get('market_regime', 'Unknown')}
- Position: {'Yes' if bot_state.get('position') else 'No'}
- Balance: ${bot_state.get('balance_usdt', 0):.2f} USDT

## Risk Status
- Can Trade: {bot_state.get('risk_status', {}).get('can_trade', True)}
- Consecutive Losses: {bot_state.get('risk_status', {}).get('consecutive_losses', 0)}
- Daily Trades: {bot_state.get('risk_status', {}).get('daily_trades', 0)}

## Available Commands
You can execute commands by responding with a JSON block like this:
```command
{{"action": "command_name", "params": {{}}}}
```

Available commands:
{json.dumps(self.available_commands, indent=2)}

## Guidelines
1. Be helpful and explain trading concepts when asked
2. Warn the user before executing risky commands
3. For code changes, always show the change first and ask for confirmation
4. Keep responses concise but informative
5. If you need to execute a command, explain what it does first

## Code Structure
The bot is organized as:
- config.py - All configuration settings
- dashboard.py - Web UI and main trading loop
- core/ - Exchange client, trader, risk management
- ai/ - AI engine, ML prediction, analysis
- market/ - Market regime, multi-timeframe analysis
- notifications/ - Telegram alerts

When editing code, use the edit_file command with the file path and the exact changes."""
    
    def chat(self, message: str, bot_state: Dict) -> Tuple[str, Optional[Dict]]:
        """
        Process a chat message and return response.
        
        Returns:
            Tuple of (response_text, command_to_execute)
        """
        if not self.is_available:
            return self._fallback_response(message, bot_state), None
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        try:
            if self.provider == "anthropic":
                response = self._chat_anthropic(bot_state)
            else:
                response = self._chat_openai(bot_state)
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Check if response contains a command
            command = self._extract_command(response)
            
            return response, command
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Sorry, I encountered an error: {str(e)}", None
    
    def _chat_anthropic(self, bot_state: Dict) -> str:
        """Chat using Anthropic Claude"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=self.get_system_prompt(bot_state),
            messages=self.conversation_history
        )
        return response.content[0].text
    
    def _chat_openai(self, bot_state: Dict) -> str:
        """Chat using OpenAI"""
        messages = [
            {"role": "system", "content": self.get_system_prompt(bot_state)}
        ] + self.conversation_history
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2048,
            messages=messages
        )
        return response.choices[0].message.content
    
    def _extract_command(self, response: str) -> Optional[Dict]:
        """Extract command from response if present"""
        import re
        
        # Look for ```command ... ``` blocks
        pattern = r'```command\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _fallback_response(self, message: str, bot_state: Dict) -> str:
        """Provide basic responses when AI is not available"""
        message_lower = message.lower()
        
        if "status" in message_lower or "state" in message_lower:
            return f"""**Bot Status**
- Running: {bot_state.get('running', False)}
- Symbol: {bot_state.get('current_symbol', 'Unknown')}
- Price: ${bot_state.get('current_price', 0):.2f}
- Decision: {bot_state.get('decision', 'Unknown')}
- AI Score: {bot_state.get('ai_score', 0):.2f}
- Can Trade: {bot_state.get('risk_status', {}).get('can_trade', True)}

To enable AI chat, set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."""
        
        elif "help" in message_lower:
            return """**Available Commands** (type these exactly):
- `status` - Get bot status
- `reset risk` - Reset risk counters
- `start` - Start the bot
- `stop` - Stop the bot
- `pause` - Pause trading
- `resume` - Resume trading

For full AI assistance, set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."""
        
        elif "reset" in message_lower and "risk" in message_lower:
            return "COMMAND:reset_risk"
        
        elif "start" in message_lower:
            return "COMMAND:start_bot"
        
        elif "stop" in message_lower:
            return "COMMAND:stop_bot"
        
        elif "pause" in message_lower:
            return "COMMAND:pause_bot"
        
        elif "resume" in message_lower:
            return "COMMAND:resume_bot"
        
        else:
            return f"""I understand you said: "{message}"

I'm running in basic mode (no AI API configured).

**Quick Commands:**
- `status` - Bot status
- `reset risk` - Reset risk counters  
- `start/stop/pause/resume` - Control bot

For full AI assistance, add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file."""
    
    def execute_command(self, command: Dict, bot_state: Dict) -> Dict:
        """Execute a command and return result"""
        action = command.get("action", "")
        params = command.get("params", {})
        
        try:
            if action == "reset_risk":
                return self._cmd_reset_risk()
            elif action == "get_state":
                return {"status": "ok", "state": bot_state}
            elif action == "get_logs":
                return {"status": "ok", "logs": bot_state.get("activity_log", [])[-20:]}
            elif action == "read_file":
                return self._cmd_read_file(params.get("path", ""))
            elif action == "edit_file":
                return self._cmd_edit_file(
                    params.get("path", ""),
                    params.get("old_text", ""),
                    params.get("new_text", "")
                )
            elif action == "list_files":
                return self._cmd_list_files(params.get("path", "."))
            else:
                # Return command for dashboard to execute
                return {"status": "pending", "action": action, "params": params}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _cmd_reset_risk(self) -> Dict:
        """Reset risk counters"""
        from core.risk_manager import get_risk_manager
        risk_mgr = get_risk_manager()
        risk_mgr.reset_daily_stats()
        return {"status": "ok", "message": "Risk counters reset successfully"}
    
    def _cmd_read_file(self, path: str) -> Dict:
        """Read a file from the codebase"""
        if not path:
            return {"status": "error", "message": "No path provided"}
        
        # Security: only allow reading from project directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.normpath(os.path.join(project_root, path))
        
        if not full_path.startswith(project_root):
            return {"status": "error", "message": "Access denied: path outside project"}
        
        if not os.path.exists(full_path):
            return {"status": "error", "message": f"File not found: {path}"}
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Limit content size
            if len(content) > 50000:
                content = content[:50000] + "\n\n... (truncated, file too large)"
            
            return {"status": "ok", "path": path, "content": content}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _cmd_edit_file(self, path: str, old_text: str, new_text: str) -> Dict:
        """Edit a file in the codebase (string replacement)"""
        if not path or not old_text:
            return {"status": "error", "message": "Missing path or old_text"}
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.normpath(os.path.join(project_root, path))
        
        if not full_path.startswith(project_root):
            return {"status": "error", "message": "Access denied: path outside project"}
        
        if not os.path.exists(full_path):
            return {"status": "error", "message": f"File not found: {path}"}
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_text not in content:
                return {"status": "error", "message": "old_text not found in file"}
            
            # Count occurrences
            count = content.count(old_text)
            if count > 1:
                return {
                    "status": "error", 
                    "message": f"old_text found {count} times - must be unique. Add more context."
                }
            
            # Make the replacement
            new_content = content.replace(old_text, new_text)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {"status": "ok", "message": f"File {path} updated successfully"}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _cmd_list_files(self, path: str) -> Dict:
        """List files in a directory"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.normpath(os.path.join(project_root, path))
        
        if not full_path.startswith(project_root):
            return {"status": "error", "message": "Access denied: path outside project"}
        
        if not os.path.exists(full_path):
            return {"status": "error", "message": f"Path not found: {path}"}
        
        try:
            items = []
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                items.append({
                    "name": item,
                    "type": "dir" if os.path.isdir(item_path) else "file"
                })
            
            return {"status": "ok", "path": path, "items": sorted(items, key=lambda x: (x["type"] == "file", x["name"]))}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# Singleton instance
_chat_assistant: Optional[ChatAssistant] = None


def get_chat_assistant() -> ChatAssistant:
    """Get or create singleton chat assistant"""
    global _chat_assistant
    if _chat_assistant is None:
        _chat_assistant = ChatAssistant()
    return _chat_assistant
