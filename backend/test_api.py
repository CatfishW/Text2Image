#!/usr/bin/env python3
"""
Test script for the Text-to-Image API
Tests the API endpoints without requiring the model to be loaded
"""

import requests
import json
import sys
from typing import Optional

API_URL = "http://localhost:8000"


def test_health() -> bool:
    """Test the health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed")
            print(f"  Status: {data.get('status')}")
            print(f"  Model loaded: {data.get('model_loaded')}")
            print(f"  CUDA available: {data.get('cuda_available')}")
            if data.get('cuda_device'):
                print(f"  CUDA device: {data.get('cuda_device')}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to API at {API_URL}")
        print("  Make sure the backend server is running:")
        print("    python main.py")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def test_root() -> bool:
    """Test the root endpoint"""
    print("\nTesting / endpoint...")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Root endpoint works")
            print(f"  {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Root endpoint error: {e}")
        return False


def test_generate(prompt: str = "A beautiful sunset over mountains", skip_if_no_model: bool = True) -> bool:
    """Test the generate endpoint"""
    print(f"\nTesting /generate endpoint...")
    print(f"  Prompt: '{prompt}'")
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "",
        "height": 1024,
        "width": 1024,
        "seed": -1,
        "num_inference_steps": 9,
        "guidance_scale": 0.0
    }
    
    try:
        # First check health to see if model is loaded
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            if not health_data.get('model_loaded') and skip_if_no_model:
                print("  ⚠ Model not loaded, skipping generation test")
                print("  (This is normal if the model is still loading)")
                return True
        
        print("  Sending generation request (this may take a while)...")
        response = requests.post(
            f"{API_URL}/generate",
            json=payload,
            timeout=300  # 5 minute timeout for generation
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Generation successful!")
            print(f"  Seed: {data.get('seed')}")
            print(f"  Size: {data.get('width')}x{data.get('height')}")
            print(f"  Generation time: {data.get('generation_time_ms')}ms")
            print(f"  Image data length: {len(data.get('image_base64', ''))} characters")
            return True
        elif response.status_code == 503:
            error_data = response.json()
            print(f"✗ Service unavailable: {error_data.get('detail')}")
            return False
        else:
            print(f"✗ Generation failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"  Response: {response.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        print("✗ Generation request timed out (this is normal for first generation)")
        return False
    except Exception as e:
        print(f"✗ Generation error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Text-to-Image API Test Suite")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
        global API_URL
        API_URL = api_url
        print(f"Using API URL: {API_URL}")
    
    print(f"\nAPI URL: {API_URL}\n")
    
    results = []
    
    # Test health endpoint
    results.append(("Health Check", test_health()))
    
    # Test root endpoint
    results.append(("Root Endpoint", test_root()))
    
    # Test generate endpoint (with a simple prompt)
    test_prompt = "A red apple on a white background"
    if len(sys.argv) > 2:
        test_prompt = " ".join(sys.argv[2:])
    results.append(("Generate Endpoint", test_generate(test_prompt)))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)

