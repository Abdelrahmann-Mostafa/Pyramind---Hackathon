"""
Streamlit UI for Hypertension RAG + Agent
Supports multi-turn conversation with the agent.
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
st.write("Evidence-based hypertension management powered by ESC 2021 Guidelines.")

# Initialize RAG and Agent (cached)
@st.cache_resource
def load_rag():
    return HypertensionRAGPipeline()

@st.cache_resource
def load_agent(_rag):
    return HypertensionAgent(_rag)

rag = load_rag()
agent = load_agent(rag)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    use_agent = st.toggle("Use Agent Reasoning", value=True)
    st.info("💡 Agent reasoning: asks follow-up questions to personalise results")
    if st.button("🔄 Reset Conversation"):
        agent.reset()
        st.session_state.messages = []
        st.rerun()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about hypertension management..."):
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from agent or RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if use_agent:
                response = agent.run(prompt)
            else:
                rag_response = rag.answer_query(prompt)
                response = rag_response.format_for_display()
        st.markdown(response)

    # Store assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Example queries
with st.expander("📚 Example Questions"):
    examples = [
        "What is the recommended blood pressure target for patients with diabetes?",
        "How should elderly patients with hypertension be managed?",
        "What are the first-line antihypertensive drugs?",
        "I'm 62 with diabetes – what's my BP target?",
    ]
    for example in examples:
        if st.button(f"Try: {example}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()

st.markdown("---")
st.caption("⚠️ **DISCLAIMER**: This tool is for educational purposes. Always consult a healthcare professional.")