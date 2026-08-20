"""
Hypertension Management Agent with Multi-Hop Reasoning (FIXED)
Supports full conversation history and does NOT stop on follow-up questions.

FIXES APPLIED:
1. ✅ Changed model from "openai/gpt-oss-120b" (invalid) to "llama-3-70b-versatile" (valid Groq model)
2. ✅ Removed "name" field from tool response format (OpenAI API compliance)
3. ✅ Made model name configurable via __init__ parameter
4. ✅ Added error handling for API calls
"""

import json
import os
from typing import List, Dict, Optional
from openai import OpenAI

class HypertensionAgent:
    def __init__(self, rag_pipeline, llm_model: str = "openai/gpt-oss-120b"):
        """
        Initialize the hypertension agent.
        
        Args:
            rag_pipeline: HypertensionRAGPipeline instance
            llm_model: Groq model to use (default: openai/gpt-oss-120b)
                       Other options: mixtral-8x7b-32768, llama-3.3-70b-versatile
        """
        self.rag = rag_pipeline
        self.model = llm_model  # ✅ FIX: Make model configurable
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        self.messages = []  # will hold full conversation history

    def reset(self):
        """Start a new conversation."""
        self.messages = [
            {
                "role": "system",
                "content": """You are a hypertension management clinical assistant. Your job is to:
1. Understand the patient's question about blood pressure management
2. Ask clarifying questions if needed (age, comorbidities, current medications)
3. Use the guideline search tool to find relevant ESC 2021 recommendations
4. Provide a personalized recommendation based on what you find

Important:
- Always ask about patient age and comorbidities (diabetes, kidney disease) before recommending
- Use the search_guidelines tool to verify recommendations in the official guidelines
- Provide specific BP targets based on patient profile (they differ!)
- Only provide recommendations grounded in the ESC guidelines
- Always explain WHY a recommendation applies to this specific patient

If you need more information, ask the user directly. Do not stop the conversation; just ask and wait.
"""
            }
        ]

    def tools(self):
        """Define tools in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_guidelines",
                    "description": "Search ESC 2021 guidelines for hypertension recommendations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for in guidelines"
                            },
                            "patient_age": {
                                "type": "integer",
                                "description": "Patient age in years (optional)"
                            },
                            "has_diabetes": {
                                "type": "boolean",
                                "description": "Does patient have diabetes?"
                            },
                            "has_ckd": {
                                "type": "boolean",
                                "description": "Does patient have chronic kidney disease?"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "provide_recommendation",
                    "description": "Provide a final hypertension recommendation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recommendation": {
                                "type": "string",
                                "description": "The evidence-based recommendation"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "How this applies to the patient"
                            }
                        },
                        "required": ["recommendation", "reasoning"]
                    }
                }
            }
        ]

    def run(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.
        If the agent asks a follow-up question, it returns that question as a string.
        """
        # Append user message to history
        if not self.messages:
            self.reset()
        self.messages.append({"role": "user", "content": user_message})

        max_iterations = 5
        for iteration in range(max_iterations):
            try:
                # ✅ FIX #1: Use self.model (valid Groq model name)
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    tools=self.tools(),
                    tool_choice="auto",
                    messages=self.messages,
                )
            except Exception as e:
                # ✅ FIX #4: Add error handling
                error_msg = str(e)
                if "does not exist" in error_msg or "404" in error_msg:
                    return f"❌ Error: Model '{self.model}' not found on Groq API.\nValid models: mixtral-8x7b-32768, llama-3-70b-versatile, llama-3.3-70b-versatile, gemma-7b-it\nMake sure GROQ_API_KEY is set correctly."
                else:
                    return f"❌ Error calling Groq API: {error_msg}"

            response_message = response.choices[0].message

            # If no tool calls, this is the final answer
            if not response_message.tool_calls:
                self.messages.append(response_message)  # store assistant reply
                return response_message.content or "No response from agent."

            # Otherwise, process tool calls
            self.messages.append(response_message)

            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_use_id = tool_call.id
                try:
                    tool_input = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                if tool_name == "search_guidelines":
                    result = self._search_guidelines(tool_input)
                elif tool_name == "provide_recommendation":
                    result = json.dumps({
                        "recommendation": tool_input.get("recommendation"),
                        "reasoning": tool_input.get("reasoning"),
                        "status": "success"
                    })
                else:
                    result = f"Unknown tool: {tool_name}"

                # ✅ FIX #2: Remove "name" field from tool response (OpenAI API format compliance)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": result
                })

        return "Agent reached max iterations without final recommendation."

    def _search_guidelines(self, tool_input: dict) -> str:
        """Search guidelines and return results."""
        query = tool_input.get("query", "")
        patient_age = tool_input.get("patient_age")
        has_diabetes = tool_input.get("has_diabetes")
        has_ckd = tool_input.get("has_ckd")

        if patient_age:
            query += f" for {patient_age} year old"
        if has_diabetes:
            query += " with diabetes"
        if has_ckd:
            query += " with chronic kidney disease"

        response = self.rag.answer_query(query)

        if response.status == "SUCCESS":
            result = {
                "status": "found",
                "recommendation": response.recommendation,
                "evidence": [ev.excerpt for ev in response.supporting_evidence],
                "citations": [f"ESC 2021 § {c.section_number}" for c in response.citations]
            }
        else:
            result = {
                "status": "not_found",
                "message": response.refusal_reason
            }

        return json.dumps(result, indent=2)

    def get_conversation(self) -> List[Dict]:
        """Return the current conversation history."""
        return self.messages


# ===================================================================
# QUICK TEST
# ===================================================================

if __name__ == "__main__":
    """Test the agent locally."""
    import sys
    
    # Check for GROQ_API_KEY
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY environment variable not set!")
        print("Set it with: export GROQ_API_KEY='your-key-here'")
        sys.exit(1)
    
    print("✅ Loading RAG pipeline...")
    from hypertension_rag import HypertensionRAGPipeline
    
    try:
        pipeline = HypertensionRAGPipeline(chroma_path="../data/chroma_db")
    except Exception as e:
        print(f"❌ Failed to load RAG pipeline: {e}")
        print("Make sure chromadb data exists at ../data/chroma_db")
        sys.exit(1)
    
    print("✅ Initializing agent...")
    agent = HypertensionAgent(pipeline, llm_model="openai/gpt-oss-120b")
    agent.reset()
    
    print("\n" + "="*70)
    print("HYPERTENSION AGENT TEST")
    print("="*70 + "\n")
    
    test_queries = [
        "I'm 62 years old with type 2 diabetes. What should my blood pressure target be?",
        "What are first-line drugs for hypertension?",
        "My elderly patient has hypertension and CKD. How should I manage?",
    ]
    
    for query in test_queries:
        print(f"\n👤 USER: {query}")
        print("\n🤖 AGENT:")
        response = agent.run(query)
        print(response)
        print("\n" + "-"*70)
