import streamlit as st
import os
import sys
import time
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.generation_layer import RAGPipeline

st.set_page_config(
    page_title="Clinical Decision Support RAG",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_name="styles.css"):
    try:
        with open(file_name, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
        <style>
            .hero { background: #f0f4f8; padding: 1.5rem; border-radius: 20px; }
            .citation-box { background: #f8faff; border-left: 4px solid #2e86de; padding: 0.8rem; margin: 0.5rem 0; }
            .confidence-high { color: #1e7e34; font-weight: bold; }
            .confidence-medium { color: #856404; font-weight: bold; }
            .confidence-low { color: #721c24; font-weight: bold; }
            .confidence-insufficient { color: #383d41; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

load_css()

@st.cache_resource
def get_pipeline(api_key: str):
    chroma_path = Path("data/chroma_db")
    if not chroma_path.exists() and Path("../data/chroma_db").exists():
        chroma_path = Path("../data/chroma_db")
    if not chroma_path.exists():
        st.error("❌ Database not initialized. Run ingestion notebook.")
        st.stop()
    try:
        return RAGPipeline(
            chroma_path=str(chroma_path),
            groq_api_key=api_key,
            groq_base_url="https://api.groq.com/openai/v1",
            llm_model="openai/gpt-oss-120b"
        )
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        st.stop()

with st.sidebar:
    st.title("⚙️ RAG Settings")
    api_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    else:
        st.warning("Please enter your Groq API Key.")
    st.markdown("---")
    st.subheader("Retrieval Parameters")
    top_k = st.slider("Top-K Chunks", 1, 10, 5)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.55, 0.05)
    min_chunks = st.slider("Min Required Chunks", 1, 5, 2)
    st.markdown("---")
    st.subheader("🧪 Demo Queries")
    demo_queries = {
        "Select a demo query...": "",
        "In-scope: Analgesia": "What is the recommended analgesia for hip fracture patients upon admission?",
        "In-scope: Imaging": "What imaging is recommended if a hip fracture is suspected but initial X-rays are negative?",
        "Ambiguous: Screening": "Should all elderly women get osteoporosis screening?",
        "Out-of-scope: COVID (Safety Test)": "What treatment do you recommend for COVID-19?"
    }
    selected_demo = st.selectbox("Pre-loaded examples", list(demo_queries.keys()))
    st.markdown("---")
    st.subheader("📜 History")
    if "history" not in st.session_state:
        st.session_state.history = []

st.markdown("""
<div class="hero">
    <h1>⚕️ Clinical Decision Support System</h1>
    <p>Evidence‑based recommendations from hip fracture &amp; osteoporosis guidelines</p>
</div>
""", unsafe_allow_html=True)

col_input, col_button = st.columns([4, 1])
with col_input:
    query_input = st.text_input(
        "Enter your clinical query:",
        value=demo_queries[selected_demo] if selected_demo != "Select a demo query..." else "",
        label_visibility="collapsed",
        placeholder="e.g., What analgesia is recommended postoperatively?"
    )
with col_button:
    submit_button = st.button("Generate Answer", type="primary", use_container_width=True, disabled=not api_key)

if submit_button and query_input:
    if query_input not in st.session_state.history:
        st.session_state.history.append(query_input)
        if len(st.session_state.history) > 10:
            st.session_state.history.pop(0)

if st.session_state.history:
    for q in st.session_state.history[-5:][::-1]:
        st.sidebar.markdown(f'<div class="history-item">{q[:60]}{"..." if len(q)>60 else ""}</div>', unsafe_allow_html=True)

if submit_button and query_input:
    with st.spinner("Processing query through RAG pipeline..."):
        try:
            pipeline = get_pipeline(api_key)
            start_time = time.time()

            response = pipeline.answer_query(
                query=query_input,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
                min_chunks=min_chunks,
                max_tokens=4096
            )

            latency = time.time() - start_time

            st.markdown("### Reasoning Trail")
            with st.expander(f"🔍 Retrieval (Top {top_k} Chunks)", expanded=False):
                if response.retrieval_scores:
                    st.write(f"**Max Similarity Score:** {max(response.retrieval_scores):.2%}")
                    if response.status == "SUCCESS" and response.citations:
                        for i, (score, cit) in enumerate(zip(response.retrieval_scores, response.citations)):
                            st.markdown(
                                f"- **Chunk {i+1}:** Similarity **{score:.2%}** | "
                                f"Chunk ID: `{cit.chunk_id}` | "
                                f"Source: {cit.document_name} § {cit.section_number} - {cit.section_title} (p. {cit.page_number})"
                            )
                    else:
                        for i, score in enumerate(response.retrieval_scores):
                            st.markdown(f"- Chunk {i+1}: Similarity **{score:.2%}**")
                else:
                    st.write("No chunks retrieved.")

            st.markdown("---")

            # ---- Answer Section (NO CARD) ----
            if response.status == "SUCCESS":
                if response.recommendation:
                    conf_map = {
                        "High": "confidence-high",
                        "Medium": "confidence-medium",
                        "Low": "confidence-low",
                        "Insufficient Evidence": "confidence-insufficient"
                    }
                    badge_class = conf_map.get(response.confidence_level, "confidence-insufficient")
                    st.markdown(f'<span class="{badge_class}">🎯 {response.confidence_level} Confidence</span>', unsafe_allow_html=True)

                    st.markdown("### 📋 Recommendation")
                    st.markdown(f"<p style='font-size:1.2rem;'>{response.recommendation}</p>", unsafe_allow_html=True)

                    if response.supporting_evidence:
                        st.markdown("#### 📎 Supporting Evidence")
                        for ev in response.supporting_evidence:
                            st.markdown(f"""
                            <div class="citation-box">
                                <strong>Claim:</strong> {ev.claim}<br/>
                                <strong>Excerpt:</strong> <em>"{ev.excerpt}"</em><br/>
                                <strong>Source Chunk:</strong> <code>{ev.chunk_id}</code>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("The model did not produce a recommendation.")
            else:
                st.markdown("### ❌ Query Refused")
                st.markdown(f"""
                <div class="refusal-box">
                    <strong>Reason:</strong><br/>
                    {response.refusal_reason}
                </div>
                """, unsafe_allow_html=True)
                if response.context_found:
                    st.write(f"**Context found:** {response.context_found}")
                if response.context_lacking:
                    st.write(f"**Context lacking:** {response.context_lacking}")

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"⏱️ **Latency:** {latency:.2f}s")
            with col2:
                st.caption(f"🧠 **Model:** {response.model_used}")
            st.caption(f"_{response.clinical_disclaimer}_")

            with st.expander("🛠️ Debug: Raw JSON Response"):
                st.json(json.loads(response.to_json()))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

st.markdown("""
<div class="footer">
    ⚕️ Disclaimer: This tool provides educational decision support only. Always consult a qualified clinician.
</div>
""", unsafe_allow_html=True)