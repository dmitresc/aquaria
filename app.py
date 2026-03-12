import streamlit as st
import pandas as pd
import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util

# --- INITIAL UI SETUP ---
st.set_page_config(page_title="AQUARIA", layout="wide")

# --- CONFIGURATION ---
# Mistral is very fast and excellent at following instructions via API
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3" 
csv_path = "freshwater_aquarium_fish_species.csv"

# Accessing the token from Streamlit Secrets (Set this up in Streamlit Cloud Settings)
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    st.error("HF_TOKEN not found in Secrets. Please add it to your Streamlit Cloud settings.")
    st.stop()

# --- SIDEBAR (Logo & Branding) ---
with st.sidebar:
    # Ensure aquaria_logo.png is uploaded to your GitHub repo
    if os.path.exists("aquaria_logo.png"):
        st.image("aquaria_logo.png", use_container_width=True)
    
    st.title("🌊 AQUARIA")
    st.markdown("---")
    st.info("🚀 **Mode:** API-Accelerated (Swift)")
    st.write("Expert advice for your freshwater aquarium.")
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("v2.0 | Optimized for Cloud")

# --- RESOURCES (Cached RAG logic) ---
@st.cache_resource
def init_rag():
    # Sentence Transformer runs efficiently on CPU
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load your CSV data
    df = pd.read_csv(csv_path, encoding="latin1").fillna("Not specified")
    
    # Create the search context
    search_cols = ["name", "details", "tank size", "fish compatibility"]
    df["combined_info"] = df[search_cols].astype(str).agg(" | ".join, axis=1)
    
    # Pre-calculate embeddings for the database
    embeddings = embedder.encode(df["combined_info"].tolist(), convert_to_tensor=True)
    return embedder, df, embeddings

# Initialize RAG and the API Client
embedder, df, fish_embeddings = init_rag()
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# --- CHAT INTERFACE ---
st.title("AQUARIA: Your Fishkeeping Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm **AQUARIA** 🐠\n\nI have access to your freshwater fish database. How can I help you today?"}
    ]

# Display conversation history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("Ask about fish, compatibility, or tank requirements..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. RETRIEVAL (Find matching fish data)
        query_emb = embedder.encode(prompt, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, fish_embeddings, top_k=3)[0]
        context_data = "\n\n".join([df.iloc[h["corpus_id"]]["combined_info"] for h in hits])

        # 2. GENERATION (Call the API with streaming)
        placeholder = st.empty()
        full_response = ""
        
        # System instructions to keep the AI in character
        messages = [
            {
                "role": "system", 
                "content": f"You are AQUARIA, a professional freshwater aquarium expert. Use the following FISH DATA to answer the user accurately. If the data is missing, provide safe general advice. FISH DATA: {context_data}"
            },
            {"role": "user", "content": prompt}
        ]

        # Stream the response back to the UI
        try:
            for message in client.chat_completion(
                messages=messages,
                max_tokens=500,
                stream=True,
                temperature=0.7
            ):
                token = message.choices[0].delta.content
                if token:
                    full_response += token
                    placeholder.markdown(full_response + "▌")
            
            # Final update without the cursor
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Generation Error: {e}")
            st.write("Tip: Check your HF_TOKEN in Streamlit Secrets.")

# --- FOOTER ---
st.markdown("---")
st.caption("AQUARIA | Intelligent Freshwater Database")
