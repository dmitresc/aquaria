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
    # Added .strip() to ensure no hidden spaces break the authentication
    HF_TOKEN = st.secrets["HF_TOKEN"].strip()
except KeyError:
    st.error("HF_TOKEN not found in Secrets. Please add it to your Streamlit Cloud settings.")
    st.stop()

# --- SIDEBAR (Logo & Branding) ---
with st.sidebar:
    if os.path.exists("aquaria_logo.png"):
        st.image("aquaria_logo.png", use_container_width=True)
    
    st.title("🌊 AQUARIA")
    st.markdown("---")
    st.info("🚀 **Mode:** API-Accelerated (Swift)")
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- RESOURCES ---
@st.cache_resource
def init_rag():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
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
        
        # Proper Mistral prompt formatting
        formatted_prompt = f"<s>[INST] You are AQUARIA, a professional expert. Use this data: {context_data}\n\nQuestion: {prompt} [/INST]"

        try:
            # Switched to text_generation to avoid the '400 Bad Request' error
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
            st.write("Tip: If you see a '401', check your token in Streamlit Secrets.")

st.markdown("---")
st.caption("AQUARIA | Intelligent Freshwater Database")    df = pd.read_csv(csv_path, encoding="latin1").fillna("Not specified")
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

        # 2. GENERATION (Fixed logic for Mistral/API compatibility)
        placeholder = st.empty()
        full_response = ""
        
        # Mistral uses special [INST] tags to recognize instructions
        formatted_prompt = f"<s>[INST] You are AQUARIA, a professional freshwater aquarium expert. Use the following FISH DATA to answer accurately. \n\nDATA: {context_data} \n\nQUESTION: {prompt} [/INST]"

        try:
