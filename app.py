import streamlit as st
from engine import ProductionEntityClassifier
from config import MODEL_PATH

# 1. Page Configuration & Layout
st.set_page_config(
    page_title="Entity AI Classifier",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS for UI Enhancement
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #1E222D;
        border: 1px solid #2E3440;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #88C0D0;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ECEFF4;
    }
    /* Author Footer Badge */
    .author-badge {
        background: linear-gradient(90deg, #4C566A, #3B4252);
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        color: #E5E9F0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Engine Cache Loader
@st.cache_resource
def load_engine():
    return ProductionEntityClassifier(model_path=MODEL_PATH)

# Load Classifier
try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to load engine: {e}")
    st.stop()

# 4. Header & Plain-English Explanation
st.title("⚡ Entity Intelligence Engine")

st.markdown("""
### What does this app do?
Think of this as a **smart digital sorting assistant**. 

When you type a word or sentence into the box below, our AI reads it and instantly tells you whether it is a **Person's Name**, a **Country**, or just **Random Text/Noise**. It helps clean up messy contact lists, user registrations, and web databases automatically!
""")

# 5. Collapsible "Know More" Technical Deep Dive Section
with st.expander("📖 Know More (Technical Details & Architecture)"):
    st.markdown("""
    **Developer:** <span class="author-badge">Created by Nyoh Leonard</span>
    
    #### How It Works Under the Hood:
    1. **Quantized FastText Core:** Uses subword n-gram vector embeddings to understand unknown words, misspellings, and out-of-vocabulary terms with zero latency.
    2. **Deterministic Rules Engine:** Overrides probabilistic machine learning predictions using exact matching dictionary rules for critical entities (e.g., precise country list mapping).
    3. **Confidence Scoring:** Outputs real-time softmax confidence metrics alongside prediction engine metadata (`model` vs. `rule_override`).
    """, unsafe_allow_html=True)

st.divider()

# 6. Interactive Sample Buttons
st.markdown("##### 💡 Try an Example:")
sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if sample_col1.button("👤 Neba"):
    st.session_state.user_input = "Neba"
if sample_col2.button("🌍 Cameroon"):
    st.session_state.user_input = "Cameroon"
if sample_col3.button("❓ asdfghj"):
    st.session_state.user_input = "asdfghj"
if sample_col4.button("👤 Leonard"):
    st.session_state.user_input = "Leonard"

# 7. Text Input Area
text_input = st.text_input(
    "Enter text to analyze:", 
    value=st.session_state.user_input,
    placeholder="Type a name, country, or random string...",
    key="main_input"
)

# 8. Execution and Prediction Display
if st.button("🚀 Analyze Text", type="primary", use_container_width=True) or text_input:
    if not text_input.strip():
        st.warning("Please enter some text or click an example above.")
    else:
        with st.spinner("Analyzing text..."):
            result = engine.predict(text_input)
            
            label = result["label"]
            confidence = float(result["confidence"])
            source = result["source"]
            
        st.markdown("---")
        st.subheader("Classification Results")
        
        # Display Visual Cards
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Label</div>
                <div class="metric-value">{label.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Decision Source</div>
                <div class="metric-value">{source.replace('_', ' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)

        # Progress bar visualizer for confidence score
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**Model Certainty Level:**")
        st.progress(confidence)

        # Expandable JSON Response
        with st.expander("🔍 View Raw JSON Payload"):
            st.json(result)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Built by <b>Nyoh Leonard</b> • FastAPI & Streamlit Microservice Engine</p>", unsafe_allow_html=True)
