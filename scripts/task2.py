import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load cleaned data
df = pd.read_csv('filtered_complaints.csv')
df.head()

# Extract the cleaned narratives
texts = df['cleaned_narrative'].dropna().tolist()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Split each narrative into chunks
all_chunks = []
for text in texts:
    chunks = text_splitter.split_text(text)
    all_chunks.extend(chunks)

# Preview result
print(f"Total chunks created: {len(all_chunks)}")
print(all_chunks[:5])  


texts = df['cleaned_narrative'].dropna().tolist()

# Define configurations to test
configs = [
    {'chunk_size': 300, 'chunk_overlap': 50},
    {'chunk_size': 500, 'chunk_overlap': 50},
    {'chunk_size': 700, 'chunk_overlap': 100},
]

# Test each configuration
for config in configs:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    total_chunks = 0
    for text in texts:
        chunks = splitter.split_text(text)
        total_chunks += len(chunks)
    
    print(f"chunk_size={config['chunk_size']}, chunk_overlap={config['chunk_overlap']} → Total Chunks: {total_chunks}")
