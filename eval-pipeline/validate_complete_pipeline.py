#!/usr/bin/env python3
"""
Complete Pipeline Validation Script
Tests the entire hybrid RAG evaluation pipeline with custom LLM integration
"""

import sys
import yaml
import json
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path(__file__).parent / "test_logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"pipeline_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging to: {log_file}")
    return logger

def create_test_environment():
    """Create a complete test environment"""
    logger = logging.getLogger(__name__)
    logger.info("🏗️ Creating test environment...")
    
    # Create test directories
    test_dirs = [
        "test_documents",
        "test_outputs",
        "test_configs"
    ]
      for dir_name in test_dirs:
        (Path(__file__).parent / dir_name).mkdir(exist_ok=True)
    
    # Create comprehensive test documents
    test_doc_dir = Path(__file__).parent / "test_documents"
    
    # Document 1: Technical specification
    tech_doc = test_doc_dir / "technical_specification.txt"
    with open(tech_doc, 'w', encoding='utf-8') as f:
        f.write("""
# Redfish API Technical Specification

## Overview
The Redfish API provides a RESTful interface for server management operations. This specification defines the core endpoints, authentication methods, and data models.

## Authentication
Redfish supports multiple authentication mechanisms:

### Basic Authentication
- Username/password credentials sent in HTTP headers
- Simple implementation but less secure
- Recommended only for development environments

### Session Authentication
- Creates a session token after initial authentication
- More secure than basic authentication
- Recommended for production deployments

### Token-Based Authentication
- Uses JWT or similar token standards
- Supports advanced features like role-based access
- Most secure option for enterprise environments

## Core Resources

### Systems Resource
The Systems resource represents physical and virtual computer systems:
- GET /redfish/v1/Systems - List all systems
- GET /redfish/v1/Systems/{id} - Get specific system details
- POST /redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset - Reset system

### Chassis Resource
The Chassis resource represents physical enclosures:
- Contains information about power supplies, cooling fans, and sensors
- Provides environmental monitoring capabilities
- Supports power control operations

### Managers Resource
The Managers resource represents management controllers:
- BMC (Baseboard Management Controller) information
- Management network configuration
- Firmware update capabilities

## Error Handling
All API responses include standardized error information:
- HTTP status codes follow REST conventions
- Error messages provide detailed diagnostic information
- Extended error information available in response body

## Security Considerations
- Always use HTTPS in production environments
- Implement proper certificate validation
- Use strong authentication credentials
- Regular security audits recommended
""")
    
    # Document 2: User guide
    user_doc = test_doc_dir / "user_guide.txt"
    with open(user_doc, 'w', encoding='utf-8') as f:
        f.write("""
# Redfish API User Guide

## Getting Started
This guide helps you get started with the Redfish API for server management.

## Prerequisites
Before using the Redfish API, ensure you have:
- Network access to the server's management interface
- Valid authentication credentials
- A REST client (curl, Postman, or custom application)

## Quick Start Examples

### Example 1: Get System Information
```bash
curl -X GET https://server-bmc/redfish/v1/Systems/1 \
  -H "Accept: application/json" \
  -u username:password
```

This command retrieves detailed information about the server system, including:
- Processor details and specifications
- Memory configuration and status
- Boot settings and options
- Power state and control capabilities

### Example 2: Reset System
```bash
curl -X POST https://server-bmc/redfish/v1/Systems/1/Actions/ComputerSystem.Reset \
  -H "Content-Type: application/json" \
  -d '{"ResetType": "GracefulRestart"}' \
  -u username:password
```

### Example 3: Get Thermal Information
```bash
curl -X GET https://server-bmc/redfish/v1/Chassis/1/Thermal \
  -H "Accept: application/json" \
  -u username:password
```

## Common Use Cases

### Server Health Monitoring
Monitor server health by regularly checking:
- System status and alerts
- Temperature sensors
- Fan speeds and status
- Power consumption metrics

### Firmware Management
Use Redfish for firmware operations:
- Check current firmware versions
- Upload firmware updates
- Schedule firmware installations
- Monitor update progress

### Configuration Management
Manage server configuration through:
- BIOS settings modification
- Boot order configuration
- Network interface setup
- User account management

## Best Practices
- Use session authentication for multiple operations
- Implement proper error handling in applications
- Monitor API rate limits
- Cache frequently accessed data
- Follow security guidelines for production use
""")
    
    # Document 3: FAQ
    faq_doc = test_doc_dir / "faq.txt"
    with open(faq_doc, 'w', encoding='utf-8') as f:
        f.write("""
# Redfish API Frequently Asked Questions

## General Questions

### What is the Redfish API?
The Redfish API is a standards-based interface for server management operations. It provides a RESTful API that enables remote management of servers, storage systems, and networking equipment.

### Why use Redfish instead of IPMI?
Redfish offers several advantages over traditional IPMI:
- Modern RESTful interface instead of binary protocols
- JSON data format for easier parsing and integration
- Standardized resource models across vendors
- Better security with HTTPS and modern authentication
- Extensible architecture for future enhancements

### Which servers support Redfish?
Most modern servers from major vendors support Redfish:
- Dell PowerEdge servers (iDRAC)
- HPE ProLiant servers (iLO)
- Cisco UCS servers
- Lenovo ThinkSystem servers
- Supermicro servers

## Technical Questions

### How do I authenticate with the Redfish API?
Redfish supports multiple authentication methods:
1. Basic Authentication: Send credentials in HTTP headers
2. Session Authentication: Create a session and use session tokens
3. Certificate Authentication: Use client certificates for authentication

### What are the most common HTTP status codes?
- 200 OK: Successful operation
- 201 Created: Resource successfully created
- 400 Bad Request: Invalid request parameters
- 401 Unauthorized: Authentication required or failed
- 404 Not Found: Requested resource does not exist
- 500 Internal Server Error: Server-side error occurred

### How do I handle asynchronous operations?
Some Redfish operations are asynchronous:
1. Initial request returns 202 Accepted
2. Response includes a Location header with task URI
3. Poll the task URI to check completion status
4. Task completes with final result or error

## Troubleshooting

### Connection Issues
If you can't connect to the Redfish API:
- Verify network connectivity to management interface
- Check if HTTPS/HTTP is properly configured
- Ensure management interface is enabled
- Verify correct port number (usually 443 for HTTPS)

### Authentication Problems
For authentication failures:
- Verify username and password are correct
- Check if account is locked or disabled
- Ensure user has necessary privileges
- Try basic authentication first, then session-based

### Performance Issues
To improve API performance:
- Use session authentication for multiple requests
- Implement connection pooling in applications
- Cache static data locally
- Use selective property queries to reduce response size

## Integration Examples

### Python Example
```python
import requests
import json

# Create session
session = requests.Session()
session.auth = ('username', 'password')
session.verify = False  # For testing only

# Get system information
response = session.get('https://server-bmc/redfish/v1/Systems/1')
system_info = response.json()
print(json.dumps(system_info, indent=2))
```

### PowerShell Example
```powershell
# Set credentials
$cred = Get-Credential

# Get system information
$uri = "https://server-bmc/redfish/v1/Systems/1"
$response = Invoke-RestMethod -Uri $uri -Credential $cred -SkipCertificateCheck
$response | ConvertTo-Json -Depth 10
```
""")
    
    logger.info(f"✅ Created {len(list(test_doc_dir.glob('*.txt')))} test documents")
    return test_doc_dir

def create_test_configuration():
    """Create test configuration files"""
    logger = logging.getLogger(__name__)
    logger.info("📋 Creating test configuration...")
    
    # Load base configuration
    base_config_path = Path(__file__).parent / "config" / "pipeline_config.yaml"
    
    if not base_config_path.exists():
        logger.error(f"❌ Base configuration not found: {base_config_path}")
        return None
    
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Modify for testing
    test_doc_dir = Path(__file__).parent / "test_documents"
    test_output_dir = Path(__file__).parent / "test_outputs"
    
    config['data_sources']['documents']['primary_docs'] = [
        str(test_doc_dir / "technical_specification.txt"),
        str(test_doc_dir / "user_guide.txt"),
        str(test_doc_dir / "faq.txt")
    ]
    
    # Reduce sample sizes for testing
    config['testset_generation']['samples_per_document'] = 20
    config['testset_generation']['max_total_samples'] = 50
    
    # Configure for on-premise operation (no external APIs)
    config['testset_generation']['ragas_config']['use_openai'] = False
    config['testset_generation']['ragas_config']['use_custom_llm'] = False
    
    # Mock RAG system endpoint for testing
    config['rag_system']['endpoint'] = "http://localhost:8080/mock-rag"
    
    # Save test configuration
    test_config_path = Path(__file__).parent / "test_configs" / "test_pipeline_config.yaml"
    with open(test_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"✅ Created test configuration: {test_config_path}")
    return test_config_path

def test_document_processing():
    """Test document loading and processing"""
    logger = logging.getLogger(__name__)
    logger.info("📄 Testing document processing...")
    
    try:
        from src.data.document_loader import DocumentLoader
        
        loader = DocumentLoader()
        test_doc_dir = Path(__file__).parent / "test_documents"
        
        for doc_file in test_doc_dir.glob("*.txt"):
            logger.info(f"Processing: {doc_file.name}")
            
            doc_data = loader.load_document(str(doc_file))
            
            if doc_data:
                logger.info(f"✅ Loaded {doc_file.name}: {len(doc_data.get('content', ''))} characters")
            else:
                logger.warning(f"⚠️ Failed to load {doc_file.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")
        return False

def test_hybrid_testset_generation():
    """Test hybrid testset generation"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Testing hybrid testset generation...")
    
    try:
        # Load test configuration
        test_config_path = Path(__file__).parent / "test_configs" / "test_pipeline_config.yaml"
        
        if not test_config_path.exists():
            logger.error("❌ Test configuration not found")
            return False
        
        with open(test_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Import and initialize generator
        from src.data.hybrid_testset_generator import HybridTestsetGenerator
        
        generator = HybridTestsetGenerator(config)
        
        # Set method to configurable (doesn't require external LLM)
        generator.method = 'configurable'
        
        # Generate testset
        test_doc_dir = Path(__file__).parent / "test_documents"
        document_paths = [str(f) for f in test_doc_dir.glob("*.txt")]
        output_dir = Path(__file__).parent / "test_outputs" / "testsets"
        
        results = generator.generate_comprehensive_testset(
            document_paths=document_paths,
            output_dir=output_dir
        )
        
        if results and 'testset' in results:
            testset = results['testset']
            logger.info(f"✅ Generated testset with {len(testset)} samples")
            
            # Analyze testset quality
            if len(testset) > 0:
                logger.info(f"   Sample columns: {list(testset.columns)}")
                logger.info(f"   Generation methods: {testset['generation_method'].value_counts().to_dict()}")
                logger.info(f"   Question types: {testset['question_type'].value_counts().to_dict()}")
            
            return True
        else:
            logger.warning("⚠️ Testset generation returned no results")
            return False
        
    except Exception as e:
        logger.error(f"❌ Hybrid testset generation failed: {e}")
        return False

def test_orchestrator_integration():
    """Test the complete orchestrator"""
    logger = logging.getLogger(__name__)
    logger.info("🎛️ Testing orchestrator integration...")
    
    try:
        # Load test configuration
        test_config_path = Path(__file__).parent / "test_configs" / "test_pipeline_config.yaml"
        
        with open(test_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Import orchestrator
        from src.pipeline.orchestrator import EvaluationOrchestrator
        
        orchestrator = EvaluationOrchestrator(config)
        
        # Test testset generation step
        logger.info("Testing testset generation step...")
        testset_results = orchestrator.generate_testsets()
        
        if testset_results:
            logger.info(f"✅ Orchestrator testset generation: {len(testset_results)} results")
            return True
        else:
            logger.warning("⚠️ Orchestrator testset generation returned no results")
            return False
        
    except Exception as e:
        logger.error(f"❌ Orchestrator integration test failed: {e}")
        return False

def test_configuration_validation():
    """Test configuration file validation"""
    logger = logging.getLogger(__name__)
    logger.info("⚙️ Testing configuration validation...")
    
    try:
        # Test main configuration
        config_path = Path(__file__).parent / "config" / "pipeline_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Required sections
        required_sections = [
            'pipeline',
            'data_sources',
            'testset_generation',
            'rag_system',
            'evaluation'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            logger.error(f"❌ Missing configuration sections: {missing_sections}")
            return False
        
        # Test secrets file
        secrets_path = Path(__file__).parent / "config" / "secrets.yaml"
        if secrets_path.exists():
            with open(secrets_path, 'r', encoding='utf-8') as f:
                secrets = yaml.safe_load(f)
            logger.info("✅ Secrets file loaded successfully")
        
        logger.info("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False

def generate_validation_report(test_results):
    """Generate a comprehensive validation report"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Generating validation report...")
    
    # Create report directory
    report_dir = Path(__file__).parent / "test_outputs" / "validation_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"pipeline_validation_report_{timestamp}.html"
    
    # Calculate summary statistics
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result == "PASS")
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pipeline Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .summary {{ margin: 20px 0; }}
            .test-results {{ margin: 20px 0; }}
            .pass {{ color: green; font-weight: bold; }}
            .fail {{ color: red; font-weight: bold; }}
            .error {{ color: orange; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧪 Pipeline Validation Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="summary">
            <h2>📊 Summary</h2>
            <ul>
                <li>Total Tests: {total_tests}</li>
                <li>Passed: <span class="pass">{passed_tests}</span></li>
                <li>Failed: <span class="fail">{failed_tests}</span></li>
                <li>Success Rate: {success_rate:.1f}%</li>
            </ul>
        </div>
        
        <div class="test-results">
            <h2>🔍 Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Result</th>
                    <th>Status</th>
                </tr>
    """
    
    for test_name, result in test_results.items():
        status_class = "pass" if result == "PASS" else "fail"
        status_icon = "✅" if result == "PASS" else "❌"
        
        html_content += f"""
                <tr>
                    <td>{test_name}</td>
                    <td><span class="{status_class}">{result}</span></td>
                    <td>{status_icon}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="recommendations">
            <h2>💡 Recommendations</h2>
    """
    
    if failed_tests == 0:
        html_content += """
            <p style="color: green; font-weight: bold;">
                🎉 All tests passed! The pipeline is ready for production use.
            </p>
        """
    else:
        html_content += """
            <ul>
                <li>Review failed tests and fix underlying issues</li>
                <li>Check log files for detailed error information</li>
                <li>Verify configuration files are properly set up</li>
                <li>Ensure all dependencies are installed correctly</li>
                <li>Test custom LLM integration if configured</li>
            </ul>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✅ Validation report saved: {report_file}")
    return report_file

def main():
    """Main validation function"""
    logger = setup_logging()
    logger.info("🚀 Starting Complete Pipeline Validation")
    logger.info("=" * 70)
    
    # Preparation steps
    logger.info("🏗️ Setting up test environment...")
    test_doc_dir = create_test_environment()
    test_config_path = create_test_configuration()
    
    if not test_config_path:
        logger.error("❌ Failed to create test configuration")
        return False
    
    # Define validation tests
    validation_tests = [
        ("Configuration Validation", test_configuration_validation),
        ("Document Processing", test_document_processing),
        ("Hybrid Testset Generation", test_hybrid_testset_generation),
        ("Orchestrator Integration", test_orchestrator_integration)
    ]
    
    # Run validation tests
    test_results = {}
    
    for test_name, test_func in validation_tests:
        logger.info(f"\n🔍 Running: {test_name}")
        logger.info("-" * 50)
        
        try:
            result = test_func()
            test_results[test_name] = "PASS" if result else "FAIL"
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.warning(f"⚠️ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            test_results[test_name] = "ERROR"
    
    # Generate validation report
    report_file = generate_validation_report(test_results)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for r in test_results.values() if r == "PASS")
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status_icon = "✅" if result == "PASS" else "❌"
        logger.info(f"{status_icon} {test_name}: {result}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    logger.info(f"📋 Detailed report: {report_file}")
    
    if passed == total:
        logger.info("\n🎉 Complete pipeline validation successful!")
        logger.info("The hybrid RAG evaluation pipeline is ready for use.")
    else:
        logger.warning("\n⚠️ Some validation tests failed.")
        logger.warning("Please review the logs and fix issues before production use.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
