#!/bin/bash
# 啟用 LangSmith 可觀測性
echo "LANGCHAIN_TRACING_V2=true" >> .env
echo "LANGCHAIN_PROJECT=domain_specific_llm_eval" >> .env
echo "# 請在這裡填入您的 LANGCHAIN_API_KEY" >> .env
echo "LANGCHAIN_API_KEY=your_api_key_here" >> .env
echo "✅ Observability 環境變數準備完成。"
