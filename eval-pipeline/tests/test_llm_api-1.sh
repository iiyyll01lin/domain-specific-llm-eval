curl -X POST http://llm-proxy.tao.inventec.net/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-I5EsSgWMjlgNFFm0DaA79862D07c4b20Be9167520237C4E9" \
-d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}'
