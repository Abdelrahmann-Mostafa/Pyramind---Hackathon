#!/usr/bin/env python3
"""
Test Hypertension RAG Pipeline
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hypertension_rag import HypertensionRAGPipeline
from src.hypertension_agent import HypertensionAgent

def test_rag_basic():
    """Test basic RAG functionality."""
    print("\n[TEST 1] Basic RAG Query")
    print("="*60)
    
    rag = HypertensionRAGPipeline()
    
    query = "What is the recommended blood pressure target for patients with diabetes?"
    response = rag.answer_query(query)
    
    print(f"Query: {query}")
    print(f"Status: {response.status}")
    print(f"Recommendation: {response.recommendation[:200]}...")
    print(f"Confidence: {response.confidence_level}")
    
    assert response.status == "SUCCESS", "RAG should return SUCCESS"
    assert response.recommendation, "Should have recommendation"
    print("✅ PASS")

def test_rag_refusal():
    """Test RAG refuses out-of-scope queries."""
    print("\n[TEST 2] Out-of-Scope Query Refusal")
    print("="*60)
    
    rag = HypertensionRAGPipeline()
    
    query = "What are the treatment options for lung cancer?"
    response = rag.answer_query(query)
    
    print(f"Query: {query}")
    print(f"Status: {response.status}")
    print(f"Refusal: {response.refusal_reason[:100]}...")
    
    assert response.status == "REFUSED", "RAG should refuse out-of-scope query"
    print("✅ PASS")

def test_agent_reasoning():
    """Test agent reasoning."""
    print("\n[TEST 3] Agent Reasoning")
    print("="*60)
    
    rag = HypertensionRAGPipeline()
    agent = HypertensionAgent(rag)
    
    query = "I'm 62 years old with diabetes. What's my blood pressure target?"
    response = agent.run(query)
    
    print(f"Query: {query}")
    print(f"Agent Response:\n{response[:300]}...")
    
    assert response, "Agent should return response"
    print("✅ PASS")

def test_citations():
    """Test that responses include citations."""
    print("\n[TEST 4] Citation Accuracy")
    print("="*60)
    
    rag = HypertensionRAGPipeline()
    
    query = "What antihypertensive drugs should be used first-line?"
    response = rag.answer_query(query)
    
    print(f"Query: {query}")
    print(f"Num Citations: {len(response.citations)}")
    
    for cit in response.citations:
        print(f"  - {cit.section_number}: {cit.section_title}")
    
    assert len(response.citations) > 0, "Should have citations"
    print("✅ PASS")

if __name__ == "__main__":
    print("\n🧪 Running Hypertension RAG Tests")
    print("="*60)
    
    try:
        test_rag_basic()
        test_rag_refusal()
        test_agent_reasoning()
        test_citations()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
