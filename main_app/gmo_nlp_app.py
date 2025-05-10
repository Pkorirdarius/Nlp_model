# gmo_nlp_app.py

import streamlit as st
from transformers import pipeline
from googletrans import Translator

# Load pipeline (placeholder: customize with fine-tuned model)
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Translator
translator = Translator()

# Knowledge base (can be extended or linked to a database)
KNOWLEDGE_BASE = {
    "GMOs cause cancer": {
        "label": "False",
        "explanation_en": "There is no evidence that GMOs cause cancer. Decades of research and global health reviews confirm GMO safety.",
        "explanation_sw": "Hakuna ushahidi kwamba GMO husababisha saratani. Utafiti wa miaka mingi unaonyesha usalama wa GMO."
    },
    "Long-term GMO consumption leads to tumors": {
        "label": "False",
        "explanation_en": "No credible long-term studies link GMO consumption to tumors. Scientific consensus supports their safety.",
        "explanation_sw": "Hakuna tafiti za muda mrefu zinazothibitisha GMO husababisha uvimbe. Sayansi inaunga mkono usalama wao."
    }
}

# UI setup
st.set_page_config(page_title="GMO Fact Checker", layout="centered")
st.title("🌽 GMO Fact Checker")
st.write("Enter a claim related to GMOs. You can use English or Swahili.")

# User input
claim = st.text_input("Your claim (e.g., GMOs cause cancer)")

if claim:
    # Translate Swahili to English if needed (for processing)
    detected_lang = translator.detect(claim).lang
    if detected_lang == "sw":
        claim_en = translator.translate(claim, src='sw', dest='en').text
    else:
        claim_en = claim

    # Check against knowledge base (simple exact match for demo)
    response = KNOWLEDGE_BASE.get(claim_en)

    if response:
        st.markdown(f"**Verdict:** {response['label']}")
        explanation = response['explanation_en'] if detected_lang == 'en' else response['explanation_sw']
        st.markdown(f"**Explanation:** {explanation}")
    else:
        # Fallback to model classification (demo placeholder)
        prediction = classifier(claim_en)[0]
        label = "True" if prediction['label'] == "POSITIVE" else "False"
        st.markdown(f"**AI Verdict:** {label} (model-based)")
        st.markdown("_This claim wasn't found in the knowledge base. Result generated using a general model. For more accuracy, extend training with GMO-specific data._")

    st.markdown("---")
    st.markdown(f"**Detected Language:** {'Swahili' if detected_lang == 'sw' else 'English'}")
    st.markdown(f"**Original Claim:** {claim}")
    st.markdown(f"**Processed Claim:** {claim_en}")
