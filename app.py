import streamlit as st
import pandas as pd
import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util

# --- 1. INITIAL UI SETUP ---
st.set_page_config(page_title="AQUARIA", layout="wide")

# --- 2. CONFIGURATION & SECRETS ---
# We use the API so we don't crash the Streamlit Cloud RAM
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
csv_path = "freshwater_aquarium_fish_species.csv"

try:
    # .strip() handles hidden spaces from copy-pasting
    HF_TOKEN = st.secrets["HF_TOKEN"].strip()
except KeyError:
    st.error("HF_TOKEN not found in Streamlit Secrets. Please add it in your dashboard settings.")
    st.stop()

# Initialize the API Client
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# --- 3. DATA & RAG RESOURCES ---
@st.cache_resource
def init_resources():
    # Embedding model is small enough for Cloud CPU
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load your specific Colab CSV logic
    df = pd.read_csv(csv_path, encoding="latin1").fillna("Not specified")

    # Your custom Colab formatting function
    def create_labeled_context(row):
        return (f"Fish: {row['name']} | Scientific: {row['taxonomy']} | "
                f"Details: {row['details']} | Tank: {row['tank size']} | "
                f"pH: {row['phRange']} | Compatibility: {row['fish compatibility']}")

    df["combined_info"] = df.apply(create_labeled_context, axis=1)
    
    # Pre-calculate embeddings for fast search
    embeddings = embedder.encode(df["combined_info"].tolist(), convert_to_tensor=True)
    return embedder, df, embeddings

embedder, df, fish_embeddings = init_resources()

# --- 4. SIDEBAR (Your original UI) ---
with st.sidebar:
    if os.path.exists("aquaria_logo.png"):
        st.image("aquaria_logo.png", use_container_width=True)
    
    
    st.markdown("---")
    st.info("🚀 **Mode:** API-Accelerated")
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 5. CHAT INTERFACE ---
st.title("AQUARIA: Your Freshwater Fishkeeping Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm **AQUARIA** 🐠\n\nHow can I help you today?"}]

# Display conversation history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 6. CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("Ask about fish, compatibility, or tank requirements..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Use a spinner so the user knows AQUARIA is working
        with st.spinner("Searching fish database..."):
            # 1. RETRIEVAL
            query_emb = embedder.encode(prompt, convert_to_tensor=True, device="cpu") # Force CPU
            hits = util.semantic_search(query_emb, fish_embeddings, top_k=3)[0]
            context_data = "\n\n".join([df.iloc[h["corpus_id"]]["combined_info"] for h in hits])

        # 2. GENERATION
        placeholder = st.empty()
        full_response = ""
        
        # Optimized prompt for Zephyr-7B
        system_instruction = f"You are AQUARIA, a professional freshwater expert. Only use the following data to answer. If not in data, give a safe general tip. DATA: {context_data}"
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]

        try:
            for message in client.chat_completion(
                messages=messages,
                max_tokens=400, # Shortened tokens = faster response
                stream=True,
                temperature=0.5 # Lower temp = more direct/faster answers
            ):
                token = message.choices[0].delta.content
                if token:
                    full_response += token
                    placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"AQUARIA is currently resting (API Busy). Please try again in 10 seconds.")
