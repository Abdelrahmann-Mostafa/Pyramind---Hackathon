"""
Streamlit UI for Hypertension RAG + Agent
"""

import streamlit as st
import os
from src.hypertension_rag import HypertensionRAGPipeline
from src.hypertension_agent import HypertensionAgent

st.set_page_config(
    page_title="💊 Hypertension Decision Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💊 Hypertension Clinical Decision Support")
st.write("""
Evidence-based hypertension management powered by ESC 2021 Guidelines.
Ask questions about blood pressure screening, diagnosis, and treatment.
""")

# Initialize
@st.cache_resource
def load_rag():
    rag = HypertensionRAGPipeline()
    return rag

@st.cache_resource
def load_agent(_rag):
    return HypertensionAgent(_rag)

rag = load_rag()
agent = load_agent(rag)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    use_agent = st.toggle("Use Agent Reasoning", value=True)
    st.info("💡 Agent reasoning: Ask follow-up questions to personalize results")

# Main UI
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area(
        "Ask about hypertension management:",
        placeholder="e.g., 'What is the blood pressure target for a 60-year-old with diabetes?'",
        height=100
    )

with col2:
    search_button = st.button("🔍 Search", use_container_width=True, key="search")

if search_button and user_input:
    with st.spinner("Searching guidelines..."):
        if use_agent:
            # Use agent reasoning
            response = agent.run(user_input)
        else:
            # Use simple RAG
            rag_response = rag.answer_query(user_input)
            response = rag_response.format_for_display()
    
    st.markdown("---")
    st.markdown("### 📋 Response")
    st.write(response)
    
    st.info("✅ Information sourced from ESC 2021 Hypertension Guidelines")

# Example queries
st.markdown("---")
with st.expander("📚 Example Questions"):
    examples = [
        "What is the recommended blood pressure target for patients with diabetes?",
        "How should elderly patients with hypertension be managed?",
        "What are the first-line antihypertensive drugs?",
        "What lifestyle modifications are recommended for hypertension?",
    ]
    for example in examples:
        if st.button(f"Try: {example}", use_container_width=True):
            st.session_state['query'] = example
            st.rerun()

# Footer
st.markdown("---")
st.caption("""
⚠️ **DISCLAIMER**: This tool is for educational purposes. It provides summaries of ESC 2021 guidelines.
Always consult qualified healthcare professionals for patient-specific decisions.
""")
