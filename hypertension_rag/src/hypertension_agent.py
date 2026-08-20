"""
Hypertension Management Agent with Multi-Hop Reasoning
Uses Groq (OpenAI format) with tools to personalize recommendations
"""

import json
import os
from typing import Optional
from openai import OpenAI

class HypertensionAgent:
    """Agent that reasons about hypertension management."""
    
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        self.conversation_history = []
    
    def tools(self):
        """Define tools the agent can use in OpenAI format."""
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
                    "name": "ask_followup",
                    "description": "Ask the user a follow-up question to personalize care",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question to ask the user"
                            }
                        },
                        "required": ["question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "provide_recommendation",
                    "description": "Provide a final hypertension recommendation based on guidelines",
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
        """Run agent loop until it provides a recommendation."""
        
        system_prompt = """You are a hypertension management clinical assistant. Your job is to:

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

Start by either understanding the patient's situation or asking clarifying questions."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Call Groq with tools
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                max_tokens=1024,
                tools=self.tools(),
                tool_choice="auto",
                messages=messages,
            )
            
            response_message = response.choices[0].message
            
            # Check if there are tool calls
            if response_message.tool_calls:
                # Add assistant message with tool calls
                messages.append(response_message)
                
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_use_id = tool_call.id
                    
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_input = {}
                    
                    # Execute tool
                    if tool_name == "search_guidelines":
                        result = self._search_guidelines(tool_input)
                    elif tool_name == "ask_followup":
                        return f"Agent asks: {tool_input.get('question', '')}\n\n[Please provide your answer, then we'll continue]"
                    elif tool_name == "provide_recommendation":
                        result = json.dumps({
                            "recommendation": tool_input.get("recommendation"),
                            "reasoning": tool_input.get("reasoning"),
                            "status": "success"
                        })
                    else:
                        result = f"Unknown tool: {tool_name}"
                    
                    # Add tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "name": tool_name,
                        "content": result
                    })
            else:
                # No tool calls, return final response text
                return response_message.content or "No response from agent."
        
        return "Agent reached max iterations without providing recommendation"
    
    def _search_guidelines(self, tool_input: dict) -> str:
        """Search guidelines and return results."""
        query = tool_input.get("query", "")
        
        # Personalize query with patient info if available
        patient_age = tool_input.get("patient_age")
        has_diabetes = tool_input.get("has_diabetes")
        has_ckd = tool_input.get("has_ckd")
        
        if patient_age:
            query += f" for {patient_age} year old"
        if has_diabetes:
            query += " with diabetes"
        if has_ckd:
            query += " with chronic kidney disease"
        
        # Use RAG to search
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


if __name__ == "__main__":
    from hypertension_rag import HypertensionRAGPipeline
    
    # Initialize
    rag = HypertensionRAGPipeline()
    agent = HypertensionAgent(rag)
    
    # Test
    response = agent.run("I'm 65 years old and just diagnosed with high blood pressure. What should I do?")
    print(response)
