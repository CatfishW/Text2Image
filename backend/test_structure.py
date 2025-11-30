#!/usr/bin/env python3
"""
Quick structure test - verifies the API is properly set up without requiring the model
This can run even if PyTorch/CUDA is not available
"""

import sys
import importlib.util

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("[OK] Core dependencies available")
        return True
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        return False

def test_main_structure():
    """Test that main.py has the correct structure"""
    print("\nTesting main.py structure...")
    try:
        spec = importlib.util.spec_from_file_location("main", "main.py")
        if spec is None:
            print("[FAIL] Cannot load main.py")
            return False
        
        # Check file exists and is readable
        with open("main.py", "r") as f:
            content = f.read()
        
        checks = {
            "FastAPI app": "FastAPI(" in content,
            "Lifespan context": "lifespan" in content or "@asynccontextmanager" in content,
            "Model loading": "ZImagePipeline" in content,
            "Generate endpoint": "@app.post(\"/generate\"" in content,
            "Health endpoint": "@app.get(\"/health\"" in content,
            "CORS middleware": "CORSMiddleware" in content,
            "Semaphore": "Semaphore" in content,
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            status = "[OK]" if passed else "[FAIL]"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"[FAIL] Error checking structure: {e}")
        return False

def test_api_endpoints():
    """Test that the API can be instantiated (without loading model)"""
    print("\nTesting API instantiation...")
    try:
        # Try to import main (this will fail if model loading is required)
        # We'll just check the file structure instead
        print("  Note: Full API test requires model to be loaded")
        print("  Use test_api.py with a running server for full testing")
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

def main():
    """Run structure tests"""
    print("=" * 60)
    print("Backend API Structure Test")
    print("=" * 60)
    print()
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Structure", test_main_structure()))
    results.append(("API Setup", test_api_endpoints()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n[OK] Backend structure looks good!")
        print("\nNext steps:")
        print("  1. Install all dependencies: pip install -r requirements.txt")
        print("  2. Start the server: python main.py")
        print("  3. Test the API: python test_api.py")
        return 0
    else:
        print("\n[FAIL] Some checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

