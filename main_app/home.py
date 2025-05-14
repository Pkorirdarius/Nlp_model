import streamlit as st
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import pandas as pd
from datetime import datetime
import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import altair as alt
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification
)

# Ensure reproducibility in language detection
DetectorFactory.seed = 42

# Constants
FEEDBACK_FILE = "feedback_log.csv"
EXAMPLES_FILE = "example_claims.json"
LABEL_MAP = {0: "MISINFORMATION", 1: "FACTUAL"}
MODEL_CHECKPOINT = "./binary_classifier/results/checkpoint-40"  # Best model from training
LANGUAGES = {
    'en': 'English',
    'sw': 'Swahili'
}

# Load model configuration
def load_config():
    config_path = os.path.join(MODEL_CHECKPOINT, "config.json")
    with open(config_path) as f:
        return json.load(f)

# Load model and tokenizer with proper configuration
@st.cache_resource
def load_model():
    try:
        config = load_config()
        
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_CHECKPOINT,
            config=config,
            local_files_only=True
        )
        
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            MODEL_CHECKPOINT,
            local_files_only=True
        )
        
        model.eval()
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Initialize feedback file if it doesn't exist
def init_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        pd.DataFrame(columns=[
            "timestamp", "input_text", "detected_language", "translated_text",
            "predicted_label", "prediction_confidence", "user_feedback"
        ]).to_csv(FEEDBACK_FILE, index=False)

# Load or create example claims
def load_example_claims():
    if not os.path.exists(EXAMPLES_FILE):
        examples = {
            "factual": [
                "GMO crops undergo rigorous safety testing before market approval.",
                "Some GMO crops have been engineered to require fewer pesticides.",
                "Golden rice is a GMO crop designed to address vitamin A deficiency."
            ],
            "misinformation": [
                "All GMO foods cause cancer and allergies.",
                "GMO crops always contaminate nearby organic farms.",
                "GMO seeds are designed to be sterile so farmers must buy new seeds each year."
            ],
            "factual_sw": [
                "Mazao ya GMO hupitia uchunguzi mkali wa usalama kabla ya kupata idhini ya soko.",
                "Baadhi ya mazao ya GMO yameundwa kupunguza matumizi ya viuatilifu.",
                "Mchele wa dhahabu ni mazao ya GMO yaliyoundwa kukabiliana na upungufu wa vitamini A."
            ],
            "misinformation_sw": [
                "Vyakula vyote vya GMO husababisha saratani na mzio.",
                "Mazao ya GMO daima huchafua mashamba ya kilimo cha kikaboni karibu nayo.",
                "Mbegu za GMO zimeundwa kuwa tasa ili wakulima wanunue mbegu mpya kila mwaka."
            ]
        }
        with open(EXAMPLES_FILE, 'w') as f:
            json.dump(examples, f)
        return examples
    else:
        with open(EXAMPLES_FILE) as f:
            return json.load(f)

# Enhanced prediction function with confidence threshold
def predict_claim(text, model, tokenizer, confidence_threshold=0.7):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512  # Matches config.json's max_position_embeddings
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, pred_label = torch.max(probs, dim=1)
        confidence = confidence.item()
        pred_label = pred_label.item()
        
    # Apply confidence threshold
    if confidence < confidence_threshold:
        return "UNCERTAIN", probs[0].tolist(), confidence
    
    return LABEL_MAP[pred_label], probs[0].tolist(), confidence

# Record feedback
def record_feedback(feedback, user_input, lang_name, translated, predicted_label, confidence):
    new_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "input_text": user_input,
        "detected_language": lang_name,
        "translated_text": translated,
        "predicted_label": predicted_label,
        "prediction_confidence": confidence,
        "user_feedback": feedback
    }

    df = pd.read_csv(FEEDBACK_FILE)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)
    st.success("✅ Thank you! Your feedback has been recorded.")

# Create a feedback summary visualization
def create_feedback_summary():
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        if len(df) > 0:
            # Prepare feedback summary
            feedback_counts = df['user_feedback'].value_counts().reset_index()
            feedback_counts.columns = ['Feedback', 'Count']
            
            # Create a bar chart using Altair
            chart = alt.Chart(feedback_counts).mark_bar().encode(
                x=alt.X('Feedback:N', sort='-y'),
                y='Count:Q',
                color=alt.Color('Feedback:N', scale=alt.Scale(scheme='category10')),
                tooltip=['Feedback', 'Count']
            ).properties(
                title='User Feedback Distribution',
                width=400,
                height=300
            )
            
            return chart
    return None

# Function to create confusion matrix visualization
def create_confusion_matrix():
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        df = df[df['user_feedback'].isin(['Correct', 'Incorrect'])]
        
        if len(df) > 5:  # Only show if we have enough data
            # Create a simple confusion matrix
            correct_predictions = df[df['user_feedback'] == 'Correct'].groupby('predicted_label').size()
            incorrect_predictions = df[df['user_feedback'] == 'Incorrect'].groupby('predicted_label').size()
            
            labels = ["MISINFORMATION", "FACTUAL", "UNCERTAIN"]
            
            # Prepare data for visualization
            conf_data = []
            for label in labels:
                correct = correct_predictions.get(label, 0)
                incorrect = incorrect_predictions.get(label, 0)
                if correct + incorrect > 0:  # Only add if we have data
                    conf_data.append({
                        'Label': label,
                        'Correct': correct,
                        'Incorrect': incorrect,
                        'Accuracy': correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
                    })
            
            if conf_data:
                conf_df = pd.DataFrame(conf_data)
                
                # Create accuracy bar chart
                accuracy_chart = alt.Chart(conf_df).mark_bar().encode(
                    x='Label:N',
                    y='Accuracy:Q',
                    color=alt.Color('Label:N', scale=alt.Scale(scheme='category10')),
                    tooltip=['Label', 'Accuracy', 'Correct', 'Incorrect']
                ).properties(
                    title='Model Accuracy by Prediction Type',
                    width=400,
                    height=300
                )
                
                return accuracy_chart
    return None

# Main UI function
def main():
    model, tokenizer = load_model()
    init_feedback_file()
    examples = load_example_claims()
    
    # App configuration
    st.set_page_config(
        page_title="GMO Misinformation Detector",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #26A69A;
        margin-bottom: 1rem;
    }
    .result-container {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .misinformation {
        background-color: rgba(255, 82, 82, 0.1);
        border: 1px solid rgba(255, 82, 82, 0.3);
    }
    .factual {
        background-color: rgba(38, 166, 154, 0.1);
        border: 1px solid rgba(38, 166, 154, 0.3);
    }
    .uncertain {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #9E9E9E;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar content
    with st.sidebar:
        st.image("https://api.placeholder.com/60/26A69A/FFFFFF?text=GMO", width=60)
        st.markdown("## Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Higher values make the model more conservative in its predictions"
        )
        
        st.markdown("---")
        
        # Example claims section
        st.markdown("## Example Claims")
        
        example_language = st.radio("Language / Lugha:", ["English", "Swahili/Kiswahili"], horizontal=True)
        claim_type = st.radio("Select claim type / Chagua aina ya dai:", ["Factual / Ukweli", "Misinformation / Habari potovu"], horizontal=True)
        
        if example_language == "English":
            if "Factual" in claim_type:
                selected_example = st.selectbox("Choose a factual claim example:", examples["factual"])
            else:
                selected_example = st.selectbox("Choose a misinformation example:", examples["misinformation"])
        else:  # Swahili
            if "Factual" in claim_type:
                selected_example = st.selectbox("Chagua mfano wa dai la ukweli:", examples["factual_sw"])
            else:
                selected_example = st.selectbox("Chagua mfano wa habari potovu:", examples["misinformation_sw"])
        
        if st.button("Use this example / Tumia mfano huu"):
            st.session_state.user_input = selected_example
        
        st.markdown("---")
        
        # Analytics section in sidebar
        st.markdown("## Analytics")
        if st.button("Show Model Performance"):
            st.session_state.show_analytics = True
        
        # About section
        with st.expander("ℹ️ About this app"):
            st.markdown("""
            This application uses a DistilBERT-based model to classify GMO-related claims as either factual or misinformation.
            
            **Model Details:**
            - Architecture: DistilBERT (6-layer distilled version of BERT)
            - Training Epochs: 6 (best checkpoint at epoch 4)
            - Best Validation Loss: 0.3687
            - Sequence Length: 512 tokens
            - Dropout: 20% (classification head)
            
            **Supported Languages:**
            - English (English)
            - Swahili (Kiswahili)
            
            Please provide feedback on predictions to help improve the model.
            """)
    
    # Main content
    st.markdown("<h1 class='main-header'>🧬 GMO Misinformation Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Analyze claims about Genetically Modified Organisms</p>", unsafe_allow_html=True)
    
    # Bilingual subtitle
    st.markdown("<p style='text-align: center; color: #757575;'>English | Kiswahili</p>", unsafe_allow_html=True)
    
    # Initialize session state for user input
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    
    # Initialize analytics view state
    if 'show_analytics' not in st.session_state:
        st.session_state.show_analytics = False
    
                    # Create tabs
    input_tab, results_tab, analytics_tab = st.tabs(["📝 Input", "🔍 Analysis Results", "📊 Analytics"])
    
    # Add Swahili translations as tooltips for accessibility
    st.markdown("""
    <style>
    [data-testid="stTabs"] > div:nth-child(1) > div:nth-child(1) > button:nth-child(1)::after {
        content: " (Ingizo)";
        font-size: 0.8em;
        color: #9E9E9E;
    }
    [data-testid="stTabs"] > div:nth-child(1) > div:nth-child(1) > button:nth-child(2)::after {
        content: " (Matokeo ya Uchambuzi)";
        font-size: 0.8em;
        color: #9E9E9E;
    }
    [data-testid="stTabs"] > div:nth-child(1) > div:nth-child(1) > button:nth-child(3)::after {
        content: " (Takwimu)";
        font-size: 0.8em;
        color: #9E9E9E;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Input Tab
    with input_tab:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter your GMO-related claim / Ingiza dai lako kuhusu GMO:",
                value=st.session_state.user_input,
                height=150,
                placeholder="Example: GMO crops can help increase yield while reducing pesticide use."
            )
        
        with col2:
            st.markdown("#### Options / Chaguo")
            auto_detect = st.checkbox("Auto-detect language / Tambua lugha kiotomatiki", value=True)
            
            if not auto_detect:
                selected_lang = st.selectbox("Select input language / Chagua lugha ya ingizo:", list(LANGUAGES.values()))
                # Get the language code from the selected language name
                lang_code = [k for k, v in LANGUAGES.items() if v == selected_lang][0]
            
            st.markdown("#### Actions / Vitendo")
            analyze_button = st.button("🔍 Analyze Claim / Chambua Dai", type="primary", use_container_width=True)
            clear_button = st.button("🗑️ Clear Input / Futa Ingizo", type="secondary", use_container_width=True)
            
            if clear_button:
                st.session_state.user_input = ""
                st.experimental_rerun()
        
        # Store the current input in session state
        if user_input != st.session_state.user_input:
            st.session_state.user_input = user_input
        
        # Handle the analyze button
        if analyze_button and user_input.strip():
            try:
                # Store results in session state
                st.session_state.analyzed = True
                
                # Language detection and translation
                if auto_detect:
                    detected_lang_code = detect(user_input)
                    lang_name = GoogleTranslator(source='auto', target='en').detect(user_input).get('language')
                else:
                    detected_lang_code = lang_code
                    lang_name = LANGUAGES[lang_code]
                
                # Only translate if not English
                if detected_lang_code != 'en':
                    translated = GoogleTranslator(source='auto', target='en').translate(user_input)
                else:
                    translated = user_input
                
                # Prediction
                predicted_label, confidence_scores, confidence = predict_claim(
                    translated, 
                    model, 
                    tokenizer,
                    confidence_threshold
                )
                
                # Store results in session state
                st.session_state.results = {
                    "input_text": user_input,
                    "detected_language": lang_name,
                    "translated_text": translated,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "confidence_scores": confidence_scores
                }
                
                # Switch to results tab
                results_tab.experimental_set_focus()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Results Tab
    with results_tab:
        if 'analyzed' in st.session_state and st.session_state.analyzed:
            results = st.session_state.results
            
            # Display language info
            st.markdown("### 🌐 Language Information / Taarifa ya Lugha")
            st.info(f"Detected language / Lugha iliyotambuliwa: **{results['detected_language'].capitalize()}**")
            
            if results['input_text'] != results['translated_text']:
                st.markdown("#### 🔁 Translated to English / Imetafsiriwa kwa Kiingereza:")
                st.info(results['translated_text'])
            
            # Display prediction results with appropriate styling
            st.markdown("### 🏷️ Classification Result / Matokeo ya Uainishaji")
            
            result_class = ""
            if results['predicted_label'] == "MISINFORMATION":
                result_class = "misinformation"
                icon = "🚫"
                message_en = "This claim is likely misinformation. Please fact-check it with scientific sources."
                message_sw = "Dai hili linaonekana kuwa habari potovu. Tafadhali hakikisha ukweli wake kwa vyanzo vya kisayansi."
                message = f"{message_en}<br><em>{message_sw}</em>"
            elif results['predicted_label'] == "FACTUAL":
                result_class = "factual"
                icon = "✅"
                message_en = "This claim appears to be factual based on our model."
                message_sw = "Dai hili linaonekana kuwa la kweli kulingana na mfano wetu."
                message = f"{message_en}<br><em>{message_sw}</em>"
            else:
                result_class = "uncertain"
                icon = "⚠️"
                message_en = "The model is uncertain about this claim. Consider manual review."
                message_sw = "Mfano hauna uhakika kuhusu dai hili. Fikiria ukaguzi wa mwongozo."
                message = f"{message_en}<br><em>{message_sw}</em>"
            
            st.markdown(f"<div class='result-container {result_class}'>"
                        f"<h2>{icon} {results['predicted_label']}</h2>"
                        f"<p>{message}</p>"
                        f"<h4>Confidence / Uhakika: {results['confidence']:.2%}</h4>"
                        "</div>", unsafe_allow_html=True)
            
            # Confidence visualization
            st.markdown("### 🔢 Confidence Scores / Alama za Uhakika")
            
            # Create a DataFrame for the confidence scores
            conf_df = pd.DataFrame({
                'Category': ['MISINFORMATION / HABARI POTOVU', 'FACTUAL / UKWELI'],
                'Score': results['confidence_scores']
            })
            
            # Create bar chart with Altair
            conf_chart = alt.Chart(conf_df).mark_bar().encode(
                x=alt.X('Category:N', title=None),
                y=alt.Y('Score:Q', title='Confidence Score / Alama ya Uhakika'),
                color=alt.Color('Category:N', scale=alt.Scale(domain=['MISINFORMATION / HABARI POTOVU', 'FACTUAL / UKWELI'], 
                                                              range=['#FF5252', '#26A69A'])),
                tooltip=['Category', 'Score']
            ).properties(
                title='Model Confidence Scores / Alama za Uhakika za Mfano',
                width=600,
                height=300
            )
            
            st.altair_chart(conf_chart, use_container_width=True)
            
            # Feedback section
            st.markdown("### 🧠 Was this prediction correct? / Je, ubashiri huu ulikuwa sahihi?")
            feedback_cols = st.columns(4)
            
            with feedback_cols[0]:
                if st.button("👍 Correct / Sahihi", help="Prediction was accurate", key="correct_btn"):
                    record_feedback("Correct", results['input_text'], results['detected_language'], 
                                   results['translated_text'], results['predicted_label'], results['confidence'])
            
            with feedback_cols[1]:
                if st.button("👎 Incorrect / Si sahihi", help="Prediction was wrong", key="incorrect_btn"):
                    record_feedback("Incorrect", results['input_text'], results['detected_language'], 
                                   results['translated_text'], results['predicted_label'], results['confidence'])
            
            with feedback_cols[2]:
                if st.button("🤔 Partially / Kiasi", help="Prediction was partially right", key="partial_btn"):
                    record_feedback("Partially Correct", results['input_text'], results['detected_language'], 
                                   results['translated_text'], results['predicted_label'], results['confidence'])
            
            with feedback_cols[3]:
                if st.button("❓ Unsure / Sina uhakika", help="Not certain about the feedback", key="unsure_btn"):
                    record_feedback("Unsure", results['input_text'], results['detected_language'], 
                                   results['translated_text'], results['predicted_label'], results['confidence'])
        else:
            st.info("👈 Enter a claim in the Input tab and click 'Analyze Claim' to see results here. / Ingiza dai katika kichupo cha Ingizo na bofya 'Chambua Dai' ili kuona matokeo hapa.")
    
    # Analytics Tab
    with analytics_tab:
        if st.session_state.show_analytics:
            st.markdown("### 📊 Model Performance Analytics")
            
            try:
                # Load and display training history
                trainer_state_path = os.path.join(MODEL_CHECKPOINT, "../trainer_state.json")
                if os.path.exists(trainer_state_path):
                    with open(trainer_state_path) as f:
                        trainer_state = json.load(f)
                    
                    st.markdown("#### 🏋️ Training Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Best Epoch", f"{trainer_state['epoch']:.1f}")
                    
                    with col2:
                        st.metric("Best Validation Loss", f"{trainer_state['best_metric']:.4f}")
                    
                    with col3:
                        st.metric("Training Steps", f"{trainer_state['global_step']}")
                    
                    # Plot training history
                    history = [x for x in trainer_state['log_history'] if 'loss' in x]
                    if history:
                        df_history = pd.DataFrame(history)
                        
                        # Create an Altair chart for loss over epochs
                        loss_chart = alt.Chart(df_history).mark_line(point=True).encode(
                            x=alt.X('epoch:Q', title='Epoch'),
                            y=alt.Y('loss:Q', title='Loss'),
                            tooltip=['epoch', 'loss']
                        ).properties(
                            title='Training Loss Over Epochs',
                            width=600,
                            height=300
                        )
                        
                        st.altair_chart(loss_chart, use_container_width=True)
                
                # User feedback visualization
                st.markdown("#### 👥 User Feedback Analytics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    feedback_chart = create_feedback_summary()
                    if feedback_chart:
                        st.altair_chart(feedback_chart, use_container_width=True)
                    else:
                        st.info("Not enough feedback data collected yet.")
                
                with col2:
                    accuracy_chart = create_confusion_matrix()
                    if accuracy_chart:
                        st.altair_chart(accuracy_chart, use_container_width=True)
                    else:
                        st.info("Not enough feedback data for accuracy metrics.")
                
                # Display recent feedback entries
                if os.path.exists(FEEDBACK_FILE):
                    df = pd.read_csv(FEEDBACK_FILE)
                    if len(df) > 0:
                        st.markdown("#### 📝 Recent Feedback Entries")
                        st.dataframe(
                            df.sort_values('timestamp', ascending=False)
                            .head(10)
                            [['timestamp', 'input_text', 'predicted_label', 'user_feedback']]
                        )
            
            except Exception as e:
                st.error(f"Error loading analytics: {str(e)}")
        else:
            st.info("Click 'Show Model Performance' in the sidebar to view analytics.")
    
    # Footer
    st.markdown("<div class='footer'>GMO Misinformation Detector v2.0 | DistilBERT Model | English & Swahili Support | Last updated: May 2025</div>", 
               unsafe_allow_html=True)

if __name__ == "__main__":
    main()