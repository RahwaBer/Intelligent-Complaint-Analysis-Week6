import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from transformers import pipeline

# --- Load model, index, metadata ---
@st.cache_resource
def load_components():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index = faiss.read_index('data/complaints_index.faiss')
    with open('data/complaints_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")
    return embed_model, faiss_index, metadata, llm

embed_model, faiss_index, metadata, llm = load_components()

# --- Prompt Template ---
def build_prompt(question, context_chunks):
    context_text = "\n\n".join(context_chunks)
    return f"""
You are a financial analyst assistant for CrediTrust. Your task is to analyze and summarize customer complaints based on the context provided.

Use only the information found in the retrieved complaint excerpts below to answer the user's question. Do not make assumptions or include information not present in the context. If the context does not contain enough information, respond by saying: "I don't have enough information to answer that."

Context:
{context_text}

Question:
{question}

Answer:
"""

# --- Retrieve Top-k Relevant Chunks ---
def retrieve_chunks(query, k=5):
    query_vec = embed_model.encode([query]).astype('float32')
    _, indices = faiss_index.search(query_vec, k)
    return [metadata[i].get("text", "[No text available]") for i in indices[0]]

# --- Generate Answer ---
def generate_answer(question, context_chunks):
    prompt = build_prompt(question, context_chunks)
    response = llm(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
    return response[0]['generated_text'].split("Answer:")[-1].strip()

# --- Streamlit UI ---
st.set_page_config(page_title="CrediTrust Complaint Assistant", layout="wide")
st.title("📊 CrediTrust Complaint Assistant")
st.markdown("Ask a question based on customer complaints. The assistant will find relevant examples and answer using real complaint data.")

# Input box
question = st.text_input("Type your question below:", placeholder="e.g., Why are users frustrated with BNPL payments?")

# Submit button
if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving and analyzing complaints..."):
            top_chunks = retrieve_chunks(question)
            answer = generate_answer(question, top_chunks)

        # Display result
        st.subheader("🔎 Answer")
        st.write(answer)

        st.subheader("📂 Retrieved Context")
        for i, chunk in enumerate(top_chunks, 1):
            st.markdown(f"**Context {i}:** {chunk}")


# Display result
st.subheader("🔎 Answer")
st.write(answer)

# Display source chunks
st.subheader("📂 Retrieved Complaint Excerpts (Sources)")
for i, chunk in enumerate(top_chunks, 1):
    with st.expander(f"Source {i}"):
        st.write(chunk)
