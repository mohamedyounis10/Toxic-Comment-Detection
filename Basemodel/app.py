import streamlit as st
import joblib
import re
import spacy
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Toxic Guard AI", page_icon="🚫", layout="wide")

# --- Model Loading ---
@st.cache_resource 
def load_models():
    # Update these paths if you move the files
    model = joblib.load(r'C:\Users\moham\Desktop\NLP\Project\Basemodel\Models\toxic_weighted_model.pkl')
    tfidf = joblib.load(r'C:\Users\moham\Desktop\NLP\Project\Basemodel\Models\tfidf_vectorizer.pkl')
    nlp = spacy.load("en_core_web_sm")
    return model, tfidf, nlp

model, tfidf, nlp = load_models()

# --- Cleaning Logic ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

# --- UI Header ---
st.title("🚫 Toxic Comment Classifier")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter the comment you want to analyze:")
    user_input = st.text_area("", placeholder="Type here...", height=150)
    analyze_btn = st.button("Analyze Sentiment 🔍")

with col2:
    st.info("### About this AI")
    st.write("This model uses **Logistic Regression** with **TF-IDF** to detect 6 types of toxicity in comments.")
    st.write("Developed as part of the NLP project at **KSIU**.")

# --- Analysis Execution ---
if analyze_btn:
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner('AI is thinking...'):
            # 1. Prediction Pipeline
            cleaned_text = clean_text(user_input)
            vectorized_text = tfidf.transform([cleaned_text])
            
            # Accessing estimators for decision_function
            prediction_scores = np.array([est.decision_function(vectorized_text) for est in model.estimators_]).T
            
            # 2. Results UI
            st.success("Analysis Complete!")
            target_cols = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']
            
            st.write("### Detection Results:")
            results_cols = st.columns(3)
            for i, col_name in enumerate(target_cols):
                with results_cols[i % 3]:
                    is_toxic = prediction_scores[0][i] > 0
                    color = "red" if is_toxic else "green"
                    status = "⚠️ DETECTED" if is_toxic else "✅ CLEAN"
                    
                    st.markdown(f"**{col_name}**")
                    st.markdown(f"<p style='color:{color}; font-weight:bold;'>{status}</p>", unsafe_allow_html=True)
                    # Mapping decision score to a 0-1 range for progress bar
                    progress_val = min(max((prediction_scores[0][i] + 1) / 2, 0.0), 1.0)
                    st.progress(progress_val)

            # --- 🔬 Detailed Pipeline Visualization (For Analysis) ---
            st.markdown("---")
            with st.expander("🔍 Show Detailed Text Analysis Pipeline (Explainable AI)"):
                st.write("### 1. Preprocessing Steps")
                
                # Regex Step Visualization
                regex_step = user_input.lower()
                regex_step = re.sub(r"https?://\S+|www\.\S+", "[URL_REMOVED]", regex_step)
                regex_step = re.sub(r"[^a-zA-Z\s]", "[SYMBOL_REMOVED]", regex_step)
                
                st.text(f"Original Input: {user_input}")
                st.text(f"After Regex & Lowercase: {regex_step}")

                st.write("### 2. Token-Level Analysis (NLP Engine)")
                
                # Generate detailed breakdown table
                doc_analysis = nlp(user_input)
                analysis_data = []
                
                for token in doc_analysis:
                    analysis_data.append({
                        "Token": token.text,
                        "Lemma (Root)": token.lemma_,
                        "POS Tag": token.pos_,
                        "Stopword": "Yes" if token.is_stop else "No",
                        "Decision": "Removed" if (token.is_stop or token.is_punct or not token.is_alpha) else "Kept"
                    })
                
                analysis_df = pd.DataFrame(analysis_data)
                
                # Styling the table for the user
                def highlight_kept(val):
                    bg_color = '#d4edda' if val == 'Kept' else '#f8d7da'
                    return f'background-color: {bg_color}; color: black; font-weight: bold;'

                st.dataframe(analysis_df.style.applymap(highlight_kept, subset=['Decision']), use_container_width=True)

                st.write("### 3. Final Model Input")
                st.info(f"The string processed by TF-IDF: **{cleaned_text}**")
                # run on cmd: streamlit run "c:\Users\moham\Desktop\NLP\Project\app.py"
