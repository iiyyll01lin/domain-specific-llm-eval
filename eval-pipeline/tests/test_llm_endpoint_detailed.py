#!/usr/bin/env python3
# filepath: /data/yy/domain-specific-llm-eval/eval-pipeline/test_llm_endpoint_detailed.py

"""
Detailed LLM Endpoint Diagnostic Tool
Analyzes the LLM API endpoint to identify connectivity and configuration issues
"""

import json
import sys
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests


class LLMEndpointDiagnostic:
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LLM-Diagnostic-Tool/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
    def run_comprehensive_test(self):
        """Run all diagnostic tests"""
        print("🔍 LLM Endpoint Comprehensive Diagnostics")
        print("=" * 50)
        print(f"Target: {self.endpoint}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        tests = [
            ("Basic Connectivity", self.test_basic_connectivity),
            ("Service Discovery", self.test_service_discovery),
            ("API Path Testing", self.test_api_paths),
            ("Authentication", self.test_authentication),
            ("API Request", self.test_api_request),
            ("Error Analysis", self.analyze_errors)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"🧪 {test_name}")
            print("-" * 30)
            try:
                result = test_func()
                results[test_name] = result
                print()
            except Exception as e:
                print(f"❌ Test failed: {e}")
                results[test_name] = {"error": str(e)}
                print()
        
        # Summary
        self.print_summary(results)
        return results
    
    def test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic HTTP connectivity"""
        parsed = urlparse(self.endpoint)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        try:
            response = self.session.get(base_url, timeout=10)
            print(f"✅ HTTP connection successful")
            print(f"   Status: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")
            
            # Check if it's an HTML response (indicates wrong service)
            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type:
                print("⚠️  WARNING: Server returned HTML (might be wrong service)")
                if len(response.text) < 1000:
                    print(f"   Response preview: {response.text[:200]}...")
            
            return {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content_type": content_type,
                "is_html": 'html' in content_type
            }
            
        except requests.exceptions.Timeout:
            print("❌ Connection timeout")
            return {"success": False, "error": "timeout"}
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error: {e}")
            return {"success": False, "error": "connection_error", "details": str(e)}
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return {"success": False, "error": "unexpected", "details": str(e)}
    
    def test_service_discovery(self) -> Dict[str, Any]:
        """Discover what service is running"""
        results = {"endpoints_found": []}
        
        # Common API paths to test
        test_paths = [
            "/",
            "/v1",
            "/v1/models",
            "/v1/chat/completions",
            "/api/v1/chat/completions",
            "/health",
            "/status",
            "/docs",
            "/openapi.json"
        ]
        
        parsed = urlparse(self.endpoint)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        for path in test_paths:
            try:
                url = urljoin(base_url, path)
                response = self.session.get(url, timeout=5)
                status = response.status_code
                
                print(f"   {path}: HTTP {status}")
                
                if status < 400:
                    results["endpoints_found"].append({
                        "path": path,
                        "status": status,
                        "content_type": response.headers.get('content-type', ''),
                        "content_length": len(response.text)
                    })
                    
                    # Show preview for interesting responses
                    if status == 200 and len(response.text) < 500:
                        print(f"      Preview: {response.text[:100]}...")
                        
            except requests.exceptions.Timeout:
                print(f"   {path}: Timeout")
            except Exception as e:
                print(f"   {path}: Error - {e}")
        
        return results
    
    def test_api_paths(self) -> Dict[str, Any]:
        """Test specific API paths for LLM service"""
        api_paths = [
            "/v1/chat/completions",
            "/api/v1/chat/completions", 
            "/v1/completions",
            "/chat/completions"
        ]
        
        results = {"api_paths": []}
        parsed = urlparse(self.endpoint)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        for path in api_paths:
            try:
                url = urljoin(base_url, path)
                
                # Test with OPTIONS first (CORS check)
                options_response = self.session.options(url, timeout=5)
                print(f"   OPTIONS {path}: {options_response.status_code}")
                
                # Test with GET
                get_response = self.session.get(url, timeout=5)
                print(f"   GET {path}: {get_response.status_code}")
                
                # Test with POST (no body)
                post_response = self.session.post(url, timeout=5)
                print(f"   POST {path}: {post_response.status_code}")
                
                results["api_paths"].append({
                    "path": path,
                    "options": options_response.status_code,
                    "get": get_response.status_code,
                    "post": post_response.status_code
                })
                
            except Exception as e:
                print(f"   {path}: Error - {e}")
                
        return results
    
    def test_authentication(self) -> Dict[str, Any]:
        """Test authentication methods"""
        api_url = urljoin(self.endpoint, "/v1/chat/completions")
        
        # Test without auth
        try:
            response = self.session.post(api_url, json={}, timeout=5)
            print(f"   No auth: HTTP {response.status_code}")
            no_auth_status = response.status_code
        except Exception as e:
            print(f"   No auth: Error - {e}")
            no_auth_status = None
        
        # Test with Bearer token
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = self.session.post(api_url, json={}, headers=headers, timeout=5)
            print(f"   Bearer token: HTTP {response.status_code}")
            bearer_status = response.status_code
            
            if response.status_code != no_auth_status:
                print("   ✅ Authentication makes a difference")
            else:
                print("   ⚠️  Authentication doesn't change response")
                
        except Exception as e:
            print(f"   Bearer token: Error - {e}")
            bearer_status = None
        
        return {
            "no_auth_status": no_auth_status,
            "bearer_status": bearer_status,
            "auth_matters": no_auth_status != bearer_status
        }
    
    def test_api_request(self) -> Dict[str, Any]:
        """Test actual API request"""
        api_url = urljoin(self.endpoint, "/v1/chat/completions")
        
        payload = {
            "model": "Qwen3-32B",
            "messages": [
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            "temperature": 0.3,
            "max_tokens": 50,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            print(f"   Sending POST to: {api_url}")
            print(f"   Payload: {json.dumps(payload, indent=2)[:200]}...")
            
            response = self.session.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            print(f"   Response status: {response.status_code}")
            print(f"   Response headers: {dict(response.headers)}")
            
            try:
                response_json = response.json()
                print(f"   Response JSON keys: {list(response_json.keys())}")
                
                # Check for expected OpenAI format
                if "choices" in response_json:
                    print("   ✅ OpenAI-compatible response format")
                    if response_json["choices"]:
                        content = response_json["choices"][0].get("message", {}).get("content", "")
                        print(f"   Generated content: {content[:100]}...")
                else:
                    print("   ⚠️  Non-standard response format")
                    print(f"   Response: {json.dumps(response_json, indent=2)[:300]}...")
                    
            except json.JSONDecodeError:
                print("   ❌ Response is not valid JSON")
                print(f"   Raw response: {response.text[:500]}...")
            
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "is_json": "application/json" in response.headers.get("content-type", ""),
                "response_preview": response.text[:200]
            }
            
        except requests.exceptions.Timeout:
            print("   ❌ Request timeout")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze common error patterns"""
        print("   Checking for common issues...")
        
        issues = []
        
        # Test root endpoint for error pages
        try:
            parsed = urlparse(self.endpoint)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            response = self.session.get(base_url, timeout=5)
            
            text = response.text.lower()
            
            if "nginx" in text:
                issues.append("Nginx reverse proxy detected")
            if "apache" in text:
                issues.append("Apache server detected")
            if "error" in text and "proxy" in text:
                issues.append("Proxy error page detected")
            if "404" in text or "not found" in text:
                issues.append("Service not found")
            if "502" in str(response.status_code) or "503" in str(response.status_code):
                issues.append("Backend service unavailable")
                
        except Exception as e:
            issues.append(f"Error analysis failed: {e}")
        
        for issue in issues:
            print(f"   ⚠️  {issue}")
        
        return {"issues": issues}
    
    def print_summary(self, results: Dict[str, Any]):
        """Print diagnostic summary"""
        print("📋 DIAGNOSTIC SUMMARY")
        print("=" * 50)
        
        # Connectivity
        basic_conn = results.get("Basic Connectivity", {})
        if basic_conn.get("success"):
            print("✅ Basic connectivity: WORKING")
        else:
            print("❌ Basic connectivity: FAILED")
            print(f"   Error: {basic_conn.get('error', 'Unknown')}")
        
        # Service type
        if basic_conn.get("is_html"):
            print("⚠️  Service type: Web server (not API)")
        else:
            print("✅ Service type: Likely API endpoint")
        
        # API endpoint
        api_test = results.get("API Request", {})
        if api_test.get("success"):
            print("✅ API functionality: WORKING")
        else:
            print("❌ API functionality: FAILED")
            if api_test.get("status_code"):
                print(f"   HTTP Status: {api_test['status_code']}")
        
        # Issues
        issues = results.get("Error Analysis", {}).get("issues", [])
        if issues:
            print("\n🚨 Identified Issues:")
            for issue in issues:
                print(f"   • {issue}")
        
        print("\n💡 Recommendations:")
        if not basic_conn.get("success"):
            print("   1. Check network connectivity and firewall rules")
            print("   2. Verify the endpoint URL is correct")
        elif basic_conn.get("is_html"):
            print("   1. The endpoint may be misconfigured")
            print("   2. Check if the LLM service is running on the correct port")
            print("   3. Verify the API path (/v1/chat/completions)")
        elif not api_test.get("success"):
            print("   1. Check API authentication (API key)")
            print("   2. Verify the request format is compatible")
            print("   3. Check server logs for detailed errors")
        else:
            print("   ✅ Endpoint appears to be working correctly!")

def main():
    endpoint = "http://llm-proxy.tao.inventec.net/v1/chat/completions"
    api_key = "sk-I5EsSgWMjlgNFFm0DaA79862D07c4b20Be9167520237C4E9"
    
    diagnostic = LLMEndpointDiagnostic(endpoint, api_key)
    results = diagnostic.run_comprehensive_test()
    
    return results

if __name__ == "__main__":
    main()