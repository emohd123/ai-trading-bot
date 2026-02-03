#!/usr/bin/env python3
"""
AI-Powered Crypto Trading Bot
Buy low, sell high with 10% profit target

Usage:
    python main.py              # Run live trading
    python main.py --testnet    # Run on Binance testnet
    python main.py --live      # Run with real money (careful!)
"""
import logging
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.trader import Trader

logger = logging.getLogger(__name__)


def print_warning():
    """Print warning for live trading"""
    logger.warning("!" * 60)
    logger.warning("WARNING: LIVE TRADING MODE - Real money will be used!")
    logger.warning("!" * 60)
    response = input("\nType 'YES' to confirm live trading: ")
    if response != "YES":
        logger.info("Live trading cancelled")
        sys.exit(0)


def main():
    """Main entry point"""
    use_testnet = False

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        if arg in ["--live", "-l"]:
            print_warning()

        elif arg in ["--testnet", "-t"]:
            use_testnet = True
            config.USE_TESTNET = True
            logger.info("TESTNET MODE: Using Binance testnet - no real money")

        elif arg in ["--help", "-h"]:
            print(__doc__)
            sys.exit(0)

        else:
            logger.error("Unknown argument: %s - use --help for usage", arg)
            sys.exit(1)

    # Validate configuration
    if not use_testnet:
        if not config.BINANCE_API_KEY or not config.BINANCE_API_SECRET:
            logger.error("API keys not configured - create .env with BINANCE_API_KEY and BINANCE_API_SECRET")
            sys.exit(1)

    # Create and run trader
    try:
        trader = Trader()
        trader.run()

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")

    except Exception as e:
        logger.exception("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
