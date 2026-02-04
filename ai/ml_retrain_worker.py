"""
ML Retrain Worker: checks for a retrain request file and runs training with safe deploy.
Run via cron or manually: python -m ai.ml_retrain_worker
"""
import os
import sys

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ai.ml_training import RETRAIN_REQUEST_FILE, clear_retrain_request, train_models


def run_retrain_if_requested() -> bool:
    """If ml_retrain_request.json exists, run training with safe_deploy and clear the request. Returns True if a retrain was run."""
    if not os.path.exists(RETRAIN_REQUEST_FILE):
        return False
    try:
        with open(RETRAIN_REQUEST_FILE, "r", encoding="utf-8") as f:
            import json
            req = json.load(f)
        reason = req.get("reason", "requested")
        print(f"Retrain requested: {reason}")
    except Exception:
        pass
    try:
        train_models(horizon="4h", safe_deploy=True)
    except Exception as e:
        print(f"Retrain failed: {e}")
    finally:
        clear_retrain_request()
    return True


if __name__ == "__main__":
    run_retrain_if_requested()
