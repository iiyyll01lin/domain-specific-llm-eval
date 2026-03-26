curl --request POST \
  --url http://llm-proxy.tao.inventec.net/v1/chat/completions \
  --header 'Accept: */*' \
  --header 'Accept-Encoding: gzip, deflate, br' \
  --header 'Authorization: Bearer sk-I5EsSgWMjlgNFFm0DaA79862D07c4b20Be9167520237C4E9' \
  --header 'Cache-Control: no-cache' \
  --header 'Connection: keep-alive' \
  --header 'Content-Length: 207' \
  --header 'Content-Type: application/json' \
  --header 'Host: llm-proxy.tao.inventec.net' \
  --header 'User-Agent: PostmanRuntime-ApipostRuntime/1.1.0' \
  --data '{
    "messages": [
        {
            "content": "",
            "role": "system"
        },
        {
            "content": "你好",
            "role": "user"
        }
    ],
    "model": "gpt-4o"
}'