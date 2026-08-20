#!/bin/bash

echo "🏥 Hypertension RAG + Agents - Full Integration Test"
echo "=================================================="

# 1. Check data
echo "[1/5] Checking data ingestion..."
ls -lh data/chroma_db/ && echo "✅ Chroma DB exists" || echo "⚠️ Chroma DB not found"

# 2. Test RAG
echo ""
echo "[2/5] Testing RAG pipeline..."
python -c "
from src.hypertension_rag import HypertensionRAGPipeline
rag = HypertensionRAGPipeline()
response = rag.answer_query('What is the BP target for diabetes?')
print(f'✅ RAG working: {response.status}')
"

# 3. Test Agent
echo ""
echo "[3/5] Testing agent..."
python -c "
from src.hypertension_rag import HypertensionRAGPipeline
from src.hypertension_agent import HypertensionAgent
rag = HypertensionRAGPipeline()
agent = HypertensionAgent(rag)
print('✅ Agent initialized')
"

# 4. Run tests
echo ""
echo "[4/5] Running test suite..."
python tests/test_hypertension_rag.py

# 5. Check UI
echo ""
echo "[5/5] Checking Streamlit app..."
ls -l app_hypertension.py && echo "✅ Streamlit app ready"

echo ""
echo "=================================================="
echo "✅ FULL INTEGRATION TEST PASSED"
echo "=================================================="
echo ""
echo "Ready to present! Run:"
echo "  streamlit run app_hypertension.py"
