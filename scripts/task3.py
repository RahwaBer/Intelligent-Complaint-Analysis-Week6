from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from transformers import pipeline

# Load the embedding model (same as Task 2)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAISS index
index = faiss.read_index('data/complaints_index.faiss')

# Load metadata
with open('data/complaints_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

def retrieve_relevant_chunks(question: str, k: int = 5):
    # Embed the question
    question_embedding = model.encode([question]).astype('float32')

    # Perform similarity search
    distances, indices = index.search(question_embedding, k)

    # Retrieve the top-k chunks and their metadata
    results = []
    for i in range(k):
        chunk_index = indices[0][i]
        score = distances[0][i]
        results.append({
            'chunk': metadata[chunk_index],  # metadata includes complaint_id, product, etc.
            'score': float(score)
        })

    return results



# Load the model (e.g., mistralai/Mistral-7B-Instruct-v0.1, or any instruction-tuned model)
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")

def generate_answer(question: str, retrieved_chunks: list, max_tokens: int = 512):
    # Extract the raw complaint text if it's stored separately
    context_texts = [chunk['chunk'].get('text', '') for chunk in retrieved_chunks]
    
    context = "\n\n".join(context_texts)
    
    prompt = prompt_template.format(context=context, question=question)

    # Generate response
    result = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    
    return result[0]['generated_text'].split("Answer:")[-1].strip()


# Sample quesitons
evaluation_questions = [
    "Why are customers frustrated with Buy Now, Pay Later (BNPL)?",
    "What issues do users report about credit card transactions?",
    "Are there frequent complaints about money transfer delays?",
    "Do customers mention any problems with personal loan interest rates?",
    "How do users feel about the savings account features?",
    "What are the most common complaints regarding repayment terms?",
    "Are support and communication frequently mentioned in complaints?",
    "What leads customers to file disputes with the company?",
    "Are customers confused about BNPL repayment schedules?",
    "What kind of fraud or unauthorized activity is reported?"
]
