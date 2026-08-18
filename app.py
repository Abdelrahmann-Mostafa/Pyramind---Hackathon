import streamlit as st
import os
import sys
import time
import json

# Ensure the app can find the 'src' module when run from outside the project directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.generation_layer import RAGPipeline

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================

st.set_page_config(
    page_title="Clinical Decision Support RAG",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better presentation
st.markdown("""
<style>
    .citation-box {
        background-color: #f0f2f6;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    .refusal-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
        color: #856404;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .confidence-insufficient { color: #6c757d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Initialization & Caching
# ==============================================================================

@st.cache_resource
def get_pipeline(api_key: str):
    """Initialize the RAG pipeline once and cache it."""
    # Ensure environment variables are loaded or passed
    return RAGPipeline(
        chroma_path="data/chroma_db",
        groq_api_key=api_key
    )

# ==============================================================================
# Sidebar UI
# ==============================================================================

st.sidebar.title("⚙️ RAG Settings")

# API Key Handling
api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
if api_key:
    os.environ["GROQ_API_KEY"] = api_key
else:
    st.sidebar.warning("Please enter your Groq API Key to continue.")

st.sidebar.markdown("---")

# Retrieval Settings
st.sidebar.subheader("Retrieval Parameters")
top_k = st.sidebar.slider("Top-K Chunks", min_value=1, max_value=10, value=5, help="Number of chunks to retrieve from the vector database.")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.55, step=0.05, help="Minimum similarity score required to attempt an answer.")
min_chunks = st.sidebar.slider("Min Required Chunks", min_value=1, max_value=5, value=2, help="Minimum number of retrieved chunks passing the threshold required to generate an answer.")

st.sidebar.markdown("---")

# Demo Queries
st.sidebar.subheader("🧪 Demo Queries")
demo_queries = {
    "Select a demo query...": "",
    "In-scope: Analgesia": "What is the recommended analgesia for hip fracture patients upon admission?",
    "In-scope: Imaging": "What imaging is recommended if a hip fracture is suspected but initial X-rays are negative?",
    "Ambiguous: Screening": "Should all elderly women get osteoporosis screening?",
    "Out-of-scope: COVID (Safety Test)": "What treatment do you recommend for COVID-19?"
}
selected_demo = st.sidebar.selectbox("Pre-loaded examples", list(demo_queries.keys()))


# ==============================================================================
# Main UI
# ==============================================================================

st.title("⚕️ Clinical Decision Support System")
st.markdown("An evidence-grounded RAG system for medical guidelines (Hip Fracture & Osteoporosis).")

# Determine the query string
query_input = st.text_input("Enter a clinical query:", value=demo_queries[selected_demo] if selected_demo != "Select a demo query..." else "")

submit_button = st.button("Generate Answer", type="primary", disabled=not api_key)

if submit_button and query_input:
    with st.spinner("Processing query through RAG pipeline..."):
        try:
            pipeline = get_pipeline(api_key)
            start_time = time.time()
            
            # Execute Pipeline
            response = pipeline.answer_query(
                query=query_input,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
                min_chunks=min_chunks
            )
            
            latency = time.time() - start_time
            
            # ---------------------------------------------------------
            # Display Results (Reasoning Trail Format)
            # ---------------------------------------------------------
            st.markdown("### Reasoning Trail")
            
            # 1. Retrieval Section
            with st.expander(f"🔍 Retrieval (Top {top_k} Chunks)", expanded=False):
                if response.retrieval_scores:
                    st.write(f"**Max Similarity Score:** {max(response.retrieval_scores):.2%}")
                    for i, score in enumerate(response.retrieval_scores):
                        st.markdown(f"- Chunk {i+1}: Similarity **{score:.2%}**")
                    if response.status == "SUCCESS":
                         for cit in response.citations:
                              st.markdown(f"**Source:** {cit.document_name} § {cit.section_number} - {cit.section_title} (p. {cit.page_number}) [Chunk ID: `{cit.chunk_id}`]")
                else:
                    st.write("No chunks retrieved.")
            
            st.markdown("---")
            
            # 2. Generation & Answer Section
            if response.status == "SUCCESS":
                st.markdown("### ✅ Generated Answer")
                
                # Confidence badge
                conf_class = f"confidence-{response.confidence_level.lower().replace(' ', '-')}"
                st.markdown(f"**Confidence:** <span class='{conf_class}'>{response.confidence_level}</span>", unsafe_allow_html=True)
                
                # Recommendation
                st.info(response.recommendation)
                
                # Citations mapped to claims
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
                # Refusal Section
                st.markdown("### ❌ Query Refused")
                st.markdown(f"""
                <div class="refusal-box">
                    <strong>Reason for Refusal:</strong><br/>
                    {response.refusal_reason}
                </div>
                """, unsafe_allow_html=True)
                
                if response.context_found:
                    st.write(f"**Context found:** {response.context_found}")
                if response.context_lacking:
                    st.write(f"**Context lacking:** {response.context_lacking}")
            
            # 3. Disclaimer and Meta
            st.markdown("---")
            st.caption(f"⏱️ **Latency:** {latency:.2f}s | 🧠 **Model:** {response.model_used}")
            st.caption(f"_{response.clinical_disclaimer}_")
            
            # Debug: Full JSON
            with st.expander("🛠️ Debug: Raw JSON Response"):
                st.json(json.loads(response.to_json()))
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
