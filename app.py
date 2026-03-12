import streamlit as st
import pandas as pd
import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util

# --- INITIAL UI SETUP ---
st.set_page_config(page_title="AQUARIA", layout="wide")

# --- CONFIGURATION ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3" 
csv_path = "freshwater_aquarium_fish_species.csv"

try:
    # Added .strip() to fix those "401 Unauthorized" errors from hidden spaces
    HF_TOKEN = st.secrets["HF_TOKEN"].strip()
except KeyError:
    st.error("HF_TOKEN not found in Secrets. Please add it to your Streamlit Cloud settings.")
    st.stop()

# --- SIDEBAR (Logo & Branding) ---
with st.sidebar:
    if os.path.exists("aquaria_logo.png"):
        st.image("aquaria_logo.png", use_container_width=True)
    
    st.markdown("---")
    st.info("🚀 **Mode:** API-Accelerated (Swift)")
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- RESOURCES ---
@st.cache_resource
def init_rag():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    # Line 96 Fix: Ensuring this is its own clean block
    df = pd.read_csv(csv_path, encoding="latin1").fillna("Not specified")
    search_cols = ["name", "details", "tank size", "fish compatibility"]
    df["combined_info"] = df[search_cols].astype(str).agg(" | ".join, axis=1)
    embeddings = embedder.encode(df["combined_info"].tolist(), convert_to_tensor=True)
    return embedder, df, embeddings

embedder, df, fish_embeddings = init_rag()
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# --- CHAT INTERFACE ---
st.title("AQUARIA: Your Fishkeeping Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm **AQUARIA** 🐠\n\nHow can I help you today?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("Ask about fish, compatibility, or tank requirements..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. RETRIEVAL
        query_emb = embedder.encode(prompt, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, fish_embeddings, top_k=3)[0]
        context_data = "\n\n".join([df.iloc[h["corpus_id"]]["combined_info"] for h in hits])

        # 2. GENERATION
        placeholder = st.empty()
        full_response = ""
        
        # Mistral tags to ensure it knows its role
        formatted_prompt = f"<s>[INST] You are AQUARIA, a professional expert. Use this data: {context_data}\n\nQuestion: {prompt} [/INST]"

        try:
            # We use text_generation to fix the 'model_not_supported' error
            for token in client.text_generation(
                formatted_prompt,
                max_new_tokens=500,
                stream=True,
                temperature=0.7,
                repetition_penalty=1.2
            ):
                full_response += token
                placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Generation Error: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("AQUARIA | Intelligent Freshwater Database")
