import streamlit as st
import pandas as pd
import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util

# --- 1. INITIAL UI SETUP ---
st.set_page_config(page_title="AQUARIA", layout="wide", page_icon="🐠")

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
csv_path = "freshwater_aquarium_fish_species.csv"

# --- 2. LOAD API TOKEN ---
try:
    HF_TOKEN = st.secrets["HF_TOKEN"].strip()
except KeyError:
    st.error("HF_TOKEN not found in Streamlit Secrets.")
    st.stop()

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=180)

# --- 3. LOAD DATABASE + EMBEDDINGS ---
@st.cache_resource
def init_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    df = pd.read_csv(csv_path, encoding="latin1").fillna("Not specified")

    def create_labeled_context(row):
        return (
            f"Fish: {row['name']} | "
            f"Scientific: {row['taxonomy']} | "
            f"Tank Size: {row['tank size']} | "
            f"pH Range: {row['phRange']} | "
            f"Compatibility: {row['fish compatibility']} | "
            f"Details: {row['details']}"
        )

    df["combined_info"] = df.apply(create_labeled_context, axis=1)

    embeddings = embedder.encode(
        df["combined_info"].tolist(),
        convert_to_tensor=True,
        device="cpu"
    )

    return embedder, df, embeddings


# --- 4. SIDEBAR ---
with st.sidebar:

    if os.path.exists("aquaria_logo.png"):
        st.image("aquaria_logo.png", use_container_width=True)

    with st.status("📡 Connecting to Fish Database..."):
        embedder, df, fish_embeddings = init_resources()
        st.write("Database Loaded!")

    st.markdown("---")
    st.info("🚀 Mode: API-Accelerated")

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# --- 5. CHAT UI ---
st.title("AQUARIA: Freshwater Fishkeeping Assistant 🐠")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm **AQUARIA** 🐠\n\nAsk me about freshwater fish, tank setups, or compatibility."
        }
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# --- 6. KEYWORD VALIDATION ---
aquarium_keywords = [
    "fish","aquarium","tank","ph","water","filter","heater",
    "guppy","tetra","betta","cichlid","goldfish","shrimp",
    "compatibility","substrate","plants","aquascape"
]


# --- 7. CHAT INPUT ---
if prompt := st.chat_input("Ask about fish, compatibility, or tank requirements..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        # --- VALIDATION BLOCK ---
        if not any(word in prompt.lower() for word in aquarium_keywords):

            response = "🐠 I specialize in **freshwater aquariums**. Please ask about fish, tanks, water parameters, or compatibility."

            placeholder.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            st.stop()

        # --- RAG RETRIEVAL ---
        with st.spinner("Consulting the fish scrolls..."):

            try:

                query_emb = embedder.encode(
                    prompt,
                    convert_to_tensor=True,
                    device="cpu"
                )

                search_results = util.semantic_search(
                    query_emb,
                    fish_embeddings,
                    top_k=5
                )

                if search_results and len(search_results[0]) > 0:

                    hits = search_results[0]

                    context_data = "\n".join([
                        f"- {df.iloc[h['corpus_id']]['combined_info']}"
                        for h in hits
                    ])

                else:
                    context_data = "No matching fish data found."

            except Exception as e:
                context_data = "Database retrieval error."
                st.warning(e)


        # --- AI GENERATION ---
        messages = [
            {
                "role": "system",
                "content": f"""
You are **AQUARIA**, an expert freshwater aquarium assistant.

RULES:
- Only answer questions about freshwater aquariums, fishkeeping, tanks, water parameters, or fish compatibility.
- If a question is unrelated to aquariums, politely refuse.
- Use the provided fish database information when possible.

Fish Database:
{context_data}
"""
            },
            {"role": "user", "content": prompt}
        ]


        try:

            response_stream = client.chat_completion(
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True
            )

            for message in response_stream:

                if message.choices:

                    delta = message.choices[0].delta

                    if hasattr(delta, "content") and delta.content:

                        token = delta.content
                        full_response += token

                        placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            if full_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
            else:
                st.error("AI returned an empty response.")

        except Exception as e:
            st.error(f"Technical Error: {e}")
