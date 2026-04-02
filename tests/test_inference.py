import sys
import os
import numpy as np

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import load_model_resources, model_status

def test_model_inference():
    print("🚀 Starting Model CI Test...")
    
    print("1️⃣ Attempting to load model resources...")
    load_model_resources()
    
    if not model_status["loaded"]:
        print("❌ FAILED: Model did not load successfully.")
        sys.exit(1)
        
    print("✅ Model loaded successfully!")
    
    # After load_model_resources, predictor should be initialized globally in main
    import main
    predictor = main.predictor
    
    if not predictor:
        print("❌ FAILED: Predictor object is None.")
        sys.exit(1)
        
    print("2️⃣ Creating dummy image array (128x128x3)...")
    dummy_image = np.zeros((128, 128, 3))
    
    print("3️⃣ Running prediction inference...")
    try:
        result = predictor.predict_array(dummy_image)
        print(f"✅ Prediction successful: {result}")
        
        if 'predicted_class' not in result:
            print("❌ FAILED: Response missing 'predicted_class'")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ FAILED during prediction array: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("🎉 ALL TESTS PASSED! Output structures look good.")

if __name__ == "__main__":
    test_model_inference()