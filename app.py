import streamlit as st
import pandas as pd
import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util

# --- 1. INITIAL UI SETUP ---
st.set_page_config(page_title="AQUARIA", layout="wide", page_icon="🐠")

# --- 2. CONFIGURATION & SECRETS ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
csv_path = "freshwater_aquarium_fish_species.csv"

try:
    HF_TOKEN = st.secrets["HF_TOKEN"].strip()
except Exception as e:
    st.error("HF_TOKEN missing from Secrets. Please add it to your Streamlit Dashboard.")
    st.stop()

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=120)

# --- 3. DATA & RAG RESOURCES ---
@st.cache_resource
def init_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    df = pd.read_csv(csv_path, encoding="latin1").fillna("Not specified")
    def create_labeled_context(row):
        return (f"Fish: {row['name']} | Scientific: {row['taxonomy']} | "
                f"Details: {row['details']} | Tank: {row['tank size']} | "
                f"pH: {row['phRange']} | Compatibility: {row['fish compatibility']}")
    df["combined_info"] = df.apply(create_labeled_context, axis=1)
    embeddings = embedder.encode(df["combined_info"].tolist(), convert_to_tensor=True, device="cpu")
    return embedder, df, embeddings

with st.sidebar:
    if os.path.exists("aquaria_logo.png"):
        st.image("aquaria_logo.png", use_container_width=True)
    with st.status("📡 Connecting to Fish Database..."):
        embedder, df, fish_embeddings = init_resources()
        st.write("Database Loaded!")
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 5. CHAT INTERFACE ---
st.title("AQUARIA: Your Freshwater Fishkeeping Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm **AQUARIA** 🐠\n\nHow can I help you today?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 6. CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("Ask about fish, compatibility, or tank requirements..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. RETRIEVAL
        with st.spinner("Consulting the fish scrolls..."):
            query_emb = embedder.encode(prompt, convert_to_tensor=True, device="cpu")
            hits = util.semantic_search(query_emb, fish_embeddings, top_k=3)[0]
            context_data = "\n\n".join([df.iloc[h["corpus_id"]]["combined_info"] for h in hits])

        # 2. GENERATION
        placeholder = st.empty()
        full_response = ""
        
        # System instruction formatted for conversational providers
        messages = [
            {"role": "system", "content": f"You are AQUARIA, a freshwater expert. DATA: {context_data}"},
            {"role": "user", "content": prompt}
        ]

        try:
            # Using chat_completion with the stable v0.2 model
            for message in client.chat_completion(
                messages=messages,
                max_tokens=450, 
                stream=True,
                temperature=0.6,
                model="mistralai/Mistral-7B-Instruct-v0.2" # Force the model here
            ):
                token = message.choices[0].delta.content
                if token:
                    full_response += token
                    placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Technical Error: {e}")
            st.info("If this persists, try rebooting the app to clear the provider cache.")
