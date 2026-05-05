import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Toxic Guard AI - DistilBERT", page_icon="🛡️", layout="wide")

# --- Model Loading ---
@st.cache_resource 
def load_models():
    model_path = r"C:\Users\moham\Desktop\NLP\Project\DistilBERT\toxic_model_zip" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

try:
    model, tokenizer = load_models()
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Prediction Logic ---
def predict_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    return probs

# --- UI Header ---
st.title("🛡️ Toxic Comment Classifier (DistilBERT)")
st.markdown("### Advanced NLP Detection System")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter the comment you want to analyze:")
    user_input = st.text_area("", placeholder="Type your comment here...", height=150)
    analyze_btn = st.button("Analyze Sentiment 🔍")

with col2:
    st.info("### About this AI")
    st.write("This model uses **DistilBERT**, a powerful Transformer model fine-tuned for toxic speech detection.")
    st.write("It understands **context** better than traditional models like SVM.")
    st.write("Developed as part of the NLP project at **KSIU**.")

# --- Analysis Execution ---
if analyze_btn:
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner('DistilBERT is analyzing the context...'):
            # 1. Get Predictions
            probabilities = predict_toxicity(user_input)
            
            # 2. Results UI
            st.success("Analysis Complete!")
            target_cols = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']
            
            st.write("### Detection Results (Probability Score):")
            results_cols = st.columns(3)
            
            for i, col_name in enumerate(target_cols):
                with results_cols[i % 3]:
                    prob = probabilities[i]
                    is_detected = prob > 0.5  
                    
                    color = "red" if is_detected else "green"
                    status = "⚠️ DETECTED" if is_detected else "✅ CLEAN"
                    
                    # عرض النتيجة
                    st.markdown(f"**{col_name}**")
                    st.markdown(f"<p style='color:{color}; font-weight:bold; font-size:20px;'>{status} ({prob:.2%})</p>", unsafe_allow_html=True)
                    st.progress(float(prob))

            # --- 🔬 Technical Breakdown ---
            st.markdown("---")
            with st.expander("🔍 Show Model Tokenization Breakdown"):
                st.write("This is how the Transformer model 'sees' your text:")
                tokens = tokenizer.tokenize(user_input)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                
                token_df = pd.DataFrame({
                    "Sub-word Token": tokens,
                    "Token ID": token_ids
                })
                st.table(token_df.T)
                st.caption("DistilBERT uses WordPiece tokenization, breaking words into smaller meaningful units.")

# run on cmd: streamlit run "c:\Users\moham\Desktop\NLP\Project\DistilBERT\application.py"