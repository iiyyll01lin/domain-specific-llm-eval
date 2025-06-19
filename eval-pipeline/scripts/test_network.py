#!/usr/bin/env python3
"""
Network connectivity test script for Docker builds
"""

import sys
import subprocess
import urllib.request
import socket
import time

def test_dns_resolution():
    """Test DNS resolution"""
    print("🔍 Testing DNS resolution...")
    try:
        socket.gethostbyname('pypi.org')
        print("✅ DNS resolution working")
        return True
    except socket.gaierror as e:
        print(f"❌ DNS resolution failed: {e}")
        return False

def test_http_connectivity():
    """Test HTTP connectivity to PyPI"""
    print("🔍 Testing HTTP connectivity to PyPI...")
    try:
        with urllib.request.urlopen('https://pypi.org/simple/', timeout=10) as response:
            if response.status == 200:
                print("✅ HTTP connectivity to PyPI working")
                return True
            else:
                print(f"⚠️ HTTP response code: {response.status}")
                return False
    except Exception as e:
        print(f"❌ HTTP connectivity failed: {e}")
        return False

def test_pip_connectivity():
    """Test pip connectivity"""
    print("🔍 Testing pip connectivity...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--dry-run", "--quiet", "requests"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Pip connectivity working")
            return True
        else:
            print(f"❌ Pip connectivity failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Pip connectivity test timed out")
        return False
    except Exception as e:
        print(f"❌ Pip connectivity test error: {e}")
        return False

def suggest_fixes():
    """Suggest network fixes"""
    print("\n🔧 Suggested fixes:")
    print("1. Check if you're behind a corporate firewall/proxy")
    print("2. Try setting HTTP_PROXY and HTTPS_PROXY environment variables")
    print("3. Use: docker build --build-arg HTTP_PROXY=http://your-proxy:port")
    print("4. Try using a different DNS server (e.g., 8.8.8.8)")
    print("5. Check if PyPI is accessible from your network")

def main():
    print("🌐 Network Connectivity Test")
    print("=" * 40)
    
    tests = [
        ("DNS Resolution", test_dns_resolution),
        ("HTTP Connectivity", test_http_connectivity),
        ("Pip Connectivity", test_pip_connectivity)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        if test_func():
            passed += 1
        time.sleep(1)
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed < len(tests):
        suggest_fixes()
        return 1
    else:
        print("🎉 All network tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
