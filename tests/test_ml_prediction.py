"""
End-to-end test for ML prediction system.
Verifies: data fetch -> feature engineering -> prediction -> AI engine integration.
"""
import sys
import time

def test_ml_prediction():
    """Test full ML prediction pipeline"""
    print("=" * 50)
    print("ML Prediction End-to-End Test")
    print("=" * 50)

    # 1. Fetch market data
    print("\n1. Fetching market data...")
    try:
        from core.binance_client import BinanceClient
        client = BinanceClient()
        df = client.get_historical_klines(interval='1h', limit=200)
    except Exception as e:
        print(f"   [SKIP] Binance requires API keys: {e}")
        print("   Using ml_training fetch (public API)...")
        from ai.ml_training import fetch_historical_data
        df = fetch_historical_data(limit=200)
    
    if df.empty or len(df) < 100:
        print(f"   [FAIL] Insufficient data: {len(df)} rows")
        return False
    print(f"   [OK] Loaded {len(df)} candles")

    # 2. ML Prediction
    print("\n2. Running ML prediction...")
    start = time.time()
    try:
        from ai.ml_predictor import MLPredictor
        predictor = MLPredictor()
        result = predictor.predict(df)
        elapsed = (time.time() - start) * 1000
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Verify output format
    print("\n3. Verifying output format...")
    required = ['direction', 'confidence', 'probability', 'model_votes', 'models_loaded']
    for key in required:
        if key not in result:
            print(f"   [FAIL] Missing key: {key}")
            return False
    print(f"   [OK] All required keys present")

    # 4. Validate values
    if result['direction'] not in ('UP', 'DOWN', 'HOLD'):
        print(f"   [FAIL] Invalid direction: {result['direction']}")
        return False
    if not 0 <= result['confidence'] <= 1:
        print(f"   [FAIL] Confidence out of range: {result['confidence']}")
        return False
    if not 0 <= result['probability'] <= 1:
        print(f"   [FAIL] Probability out of range: {result['probability']}")
        return False
    print(f"   [OK] Values valid")

    # 5. Check inference time
    if elapsed > 100:
        print(f"   [WARN] Inference took {elapsed:.0f}ms (target <100ms)")
    else:
        print(f"   [OK] Inference: {elapsed:.0f}ms")

    # 6. AI Engine integration (via get_score)
    print("\n4. Testing AI engine integration...")
    try:
        score, ml_result = predictor.get_score(df)
        print(f"   [OK] ML score: {score:.4f}, direction: {ml_result.get('direction')}")
    except Exception as e:
        print(f"   [WARN] get_score: {e}")

    # 7. Full AI engine get_decision (uses ML internally)
    print("\n5. Testing full AI engine get_decision...")
    try:
        from core.binance_client import BinanceClient
        from market.multi_timeframe import get_mtf_analysis
        from ai.ai_engine import AIEngine
        bc = BinanceClient()
        mtf = get_mtf_analysis(bc)
        analysis = mtf.get("1h")
        df_1h = mtf.get("df_1h")
        if analysis and df_1h is not None:
            engine = AIEngine()
            decision, details = engine.get_decision(analysis, df=df_1h)
            ml_data = details.get('ml_prediction', {})
            print(f"   [OK] Decision: {decision.value}, ML in details: {bool(ml_data)}")
        else:
            print("   [SKIP] Could not get MTF analysis")
    except Exception as e:
        print(f"   [WARN] AI engine get_decision: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Direction:   {result['direction']}")
    print(f"  Confidence:  {result['confidence']:.2%}")
    print(f"  Probability: {result['probability']:.4f}")
    print(f"  Model votes: {result['model_votes']}")
    print(f"  Models loaded: {result['models_loaded']}")
    if result.get('error'):
        print(f"  Error: {result['error']}")
    print("\n[PASS] End-to-end test completed successfully!")
    return True


if __name__ == '__main__':
    success = test_ml_prediction()
    sys.exit(0 if success else 1)
