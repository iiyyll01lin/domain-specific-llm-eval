# Custom LLM Integration Guide

This guide explains how to integrate custom or private LLM endpoints with the Hybrid RAG Evaluation Pipeline.

## Overview

The pipeline supports custom LLM integration for:
- **RAGAS Testset Generation**: Using your internal LLM for generating sophisticated questions
- **Secure API Key Management**: Storing credentials safely outside version control
- **Flexible Configuration**: Supporting various authentication methods and endpoints

## Configuration Steps

### 1. Enable Custom LLM in Configuration

Edit `config/pipeline_config.yaml`:

```yaml
testset_generation:
  ragas_config:
    use_openai: false        # Disable OpenAI
    use_custom_llm: true     # Enable custom LLM
    
    custom_llm:
      endpoint: "http://llm-proxy.tao.inventec.net/v1/chat/completions"
      api_key_file: "config/secrets.yaml"
      api_key_path: "inventec_llm.api_key"
      model: "Qwen3-32B"
      
      # Request settings
      temperature: 0.3
      max_tokens: 1000
      stream: false
      
      # Custom headers (optional)
      headers:
        "Accept": "*/*"
        "Content-Type": "application/json"
        "User-Agent": "RAG-Evaluation-Pipeline/1.0"
```

### 2. Set Up Secrets File

Create `config/secrets.yaml` from the template:

```bash
cp config/secrets.yaml.template config/secrets.yaml
```

Edit `config/secrets.yaml` and add your API key:

```yaml
inventec_llm:
  api_key: "your_actual_api_key_here"
```

**Important**: The `secrets.yaml` file is already in `.gitignore` to prevent accidental commits.

### 3. Test the Configuration

Run the test script to validate your setup:

```bash
python test_custom_llm_integration.py
```

## Custom LLM Implementation Details

### API Compatibility

The custom LLM wrapper expects an OpenAI-compatible endpoint that:
- Accepts POST requests to the configured endpoint
- Uses the OpenAI chat completions format
- Returns responses in OpenAI format

### Request Format

```json
{
  "model": "your-model-name",
  "messages": [
    {"role": "user", "content": "Your prompt here"}
  ],
  "temperature": 0.3,
  "max_tokens": 1000,
  "stream": false
}
```

### Response Format

```json
{
  "choices": [
    {
      "message": {
        "content": "Generated response here"
      }
    }
  ]
}
```

### Authentication

The implementation supports:
- Bearer token authentication (most common)
- Custom headers for proprietary authentication
- No authentication (for internal endpoints)

## Usage Examples

### Example 1: Internal Corporate LLM

```yaml
custom_llm:
  endpoint: "https://internal-llm.company.com/v1/chat/completions"
  api_key_file: "config/secrets.yaml"
  api_key_path: "company_llm.api_key"
  model: "company-gpt-4"
  temperature: 0.3
  max_tokens: 1000
  headers:
    "X-Company-Auth": "Bearer"
```

### Example 2: Local LLM Server

```yaml
custom_llm:
  endpoint: "http://localhost:8080/v1/chat/completions"
  api_key_file: ""  # No API key needed
  api_key_path: ""
  model: "local-llama-2"
  temperature: 0.5
  max_tokens: 800
```

### Example 3: Azure OpenAI Service

```yaml
custom_llm:
  endpoint: "https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2023-12-01-preview"
  api_key_file: "config/secrets.yaml"
  api_key_path: "azure_openai.api_key"
  model: "gpt-4"
  headers:
    "api-key": ""  # Will be filled from secrets
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Verify `secrets.yaml` exists and contains the correct key path
   - Check the `api_key_path` configuration matches your secrets file structure

2. **Connection Errors**
   - Verify the endpoint URL is correct and accessible
   - Check firewall and network settings
   - Test the endpoint manually with curl or Postman

3. **Authentication Failures**
   - Verify API key is valid and has necessary permissions
   - Check if custom headers are required
   - Review endpoint documentation for authentication requirements

4. **Response Format Errors**
   - Ensure your LLM endpoint returns OpenAI-compatible responses
   - Check if response parsing needs adjustment

### Debug Mode

Enable debug logging by setting the log level in your test script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Testing

Test your endpoint manually:

```bash
curl -X POST "your-endpoint-url" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Test message"}],
    "temperature": 0.3,
    "max_tokens": 100
  }'
```

## Security Best Practices

### API Key Management

1. **Never commit API keys to version control**
2. **Use environment variables in production**
3. **Rotate API keys regularly**
4. **Restrict API key permissions to minimum required**

### Network Security

1. **Use HTTPS endpoints when possible**
2. **Implement IP whitelisting if available**
3. **Monitor API usage and logs**
4. **Use VPN or private networks for internal endpoints**

### Code Security

1. **Validate endpoint responses**
2. **Implement request timeouts**
3. **Handle authentication errors gracefully**
4. **Log security events appropriately**

## Advanced Configuration

### Custom Headers

For endpoints requiring special authentication:

```yaml
custom_llm:
  headers:
    "X-API-Key": ""           # Filled from secrets
    "X-Client-ID": "pipeline"
    "User-Agent": "RAG-Pipeline/1.0"
```

### Retry Logic

The implementation includes basic error handling, but you can extend it:

```python
# In your custom wrapper
def _call_with_retry(self, prompt, retries=3):
    for attempt in range(retries):
        try:
            return self._call(prompt)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### Load Balancing

For multiple endpoints:

```yaml
custom_llm:
  endpoints:
    - "https://llm1.company.com/v1/chat/completions"
    - "https://llm2.company.com/v1/chat/completions"
  load_balance: "round_robin"  # or "random"
```

## Integration with RAGAS

The custom LLM integrates seamlessly with RAGAS:

1. **Question Generation**: Uses your custom LLM for generating questions
2. **Answer Validation**: Uses your custom LLM for validating answers
3. **Consistency**: Both generator and critic use the same LLM configuration

## Performance Considerations

### Request Optimization

- **Batch Processing**: Group multiple requests when possible
- **Connection Pooling**: Reuse HTTP connections
- **Async Requests**: Use async/await for better throughput

### Cost Management

- **Token Limits**: Set appropriate `max_tokens` limits
- **Caching**: Cache responses for identical prompts
- **Rate Limiting**: Respect API rate limits

### Quality Control

- **Temperature Settings**: Use lower temperature (0.1-0.3) for consistent outputs
- **Response Validation**: Validate generated content quality
- **Fallback Options**: Configure fallback methods if LLM is unavailable

## Support and Maintenance

### Monitoring

- **API Response Times**: Monitor endpoint performance
- **Error Rates**: Track failed requests
- **Token Usage**: Monitor API usage costs
- **Quality Metrics**: Assess generated content quality

### Updates

- **Endpoint Changes**: Monitor for endpoint updates
- **Model Updates**: Test with new model versions
- **Security Updates**: Keep authentication current

For additional support, review the logs generated by `test_custom_llm_integration.py` or check the main pipeline documentation.
