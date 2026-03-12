import streamlit as st
import torch
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- INITIAL UI SETUP ---
st.set_page_config(page_title="AQUARIA", layout="wide")

st.title("🌊 AQUARIA: Your Freshwater Fishkeeping Assistant")
st.write("Ask about fish, tank size, pH, compatibility, etc.")

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_NAME = "microsoft/phi-2"
csv_path = "freshwater_aquarium_fish_species.csv"
embeddings_path = "fish_embeddings.pt"

# --- CACHING FUNCTIONS (The Speed Secret) ---
@st.cache_resource
def load_embedder(device):
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

@st.cache_resource
def load_llm(llm_name):
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained(
        llm_name, 
        device_map="auto", 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    return tokenizer, model

@st.cache_data
def load_and_process_data(_csv_path):
    df = pd.read_csv(_csv_path, encoding="latin1").fillna("Not specified")
    SEARCH_COLS = ["name", "taxonomy", "details", "temprange", "phRange", "tank size", "fish size (cm/inches)", "fish compatibility"]
    df["combined_info"] = df[SEARCH_COLS].astype(str).agg(" | ".join, axis=1)
    return df

@st.cache_data
def get_embeddings(_df, _embedder):
    if os.path.exists(embeddings_path):
        return torch.load(embeddings_path, map_location=DEVICE)
    embeddings = _embedder.encode(_df["combined_info"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    torch.save(embeddings, embeddings_path)
    return embeddings

# --- PRE-LOAD EVERYTHING ---
# By calling these here, they load once when the app starts
with st.spinner("🐠 Initializing AQUARIA... This might take a moment on the first run."):
    tokenizer, model = load_llm(LLM_NAME)
    embedder = load_embedder(DEVICE)
    df = load_and_process_data(csv_path)
    fish_embeddings = get_embeddings(df, embedder)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🌊 AQUARIA")
    st.info(f"Running on: **{DEVICE.upper()}**")
    st.markdown("### Quick Tips:\n- Use specific fish names\n- Ask about tank gallon minimums")
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- CHAT SESSION INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm **AQUARIA** 🐠\n\nHow can I help you with your tank today?"}
    ]

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- LOGIC FUNCTIONS ---
def retrieve_context(query):
    query_embedding = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    hits = util.semantic_search(query_embedding, fish_embeddings, top_k=5)[0]
    contexts = [df.iloc[hit["corpus_id"]]["combined_info"] for hit in hits if hit["score"] > 0.35]
    return "\n\n".join(contexts) if contexts else "No relevant data."

def generate_answer(context, question, chat_history, streamer):
    recent_history = chat_history[-3:] 
    history_str = "".join([f"### {'USER' if m['role']=='user' else 'AQUARIA'}: {m['content']}\n" for m in recent_history])

    prompt = f"""Instruct: You are AQUARIA, a professional freshwater aquarium expert. 
Use the provided FISH DATA to answer the QUESTION. 

FISH DATA:
{context}

{history_str}
QUESTION: {question}
Output: RESPONSE:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=300, # Reduced slightly for even faster responses
        temperature=0.6,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

# --- CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("Ask about fish, tanks, compatibility..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Retrieval is now instant because everything is pre-loaded
        context = retrieve_context(prompt)
        
        # Setup the Streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Start Streaming Generation
        generate_answer(context, prompt, st.session_state.messages, streamer)
        
        placeholder = st.empty()
        full_response = ""
        
        for new_text in streamer:
            full_response += new_text
            clean_text = full_response.replace("RESPONSE:", "").strip()
            placeholder.markdown(clean_text + "▌")
            
        placeholder.markdown(clean_text)
        st.session_state.messages.append({"role": "assistant", "content": clean_text})

# Footer
st.markdown("---")
st.caption("AQUARIA | Optimized for Speed")