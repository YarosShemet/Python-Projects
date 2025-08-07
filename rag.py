import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import requests
from httpx import TimeoutException
import time

#load and preprocess the CSV data
@st.cache_data

def load_data():
    df = pd.read_csv("task1_processed_speeches.xls")  #it's a CSV file despite the extension
    df.dropna(subset=['speech_text'], inplace=True) #exclude missing speeches that are not in English
    df['content'] = df.apply(lambda row: f"{row['person']} ({row['political_group']}) on {row['date']}: {row['speech_text']}", axis=1)
    #Aims to simplify the retrieval
    return df

def embed_and_store(df, collection):
    documents = df['content'].tolist()
    metadatas = df[['date', 'topic', 'person', 'political_group']].to_dict(orient='records') #add description
    ids = [f"doc_{i}" for i in range(len(documents))]
    try:
        batch_size = 100 #add 100 rows at a time
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )
            time.sleep(3)  #small delay to avoid overwhelming the server
    except TimeoutException:
        st.error(f"Timeout while adding batch {i // batch_size + 1}.")

def query_chroma(collection, query_text):
    results = collection.query(query_texts=[query_text], n_results=10) #10 most relevant items
    return results

def generate_response(contexts, query):
    prompt = (
        "Using the following relevant context items, answer the query:\n\n"
        + "\n\n".join(contexts) + "\n\n"
        + f"Query: {query}\n"
        + "Answer:"
    )
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False},
            timeout=60
        )
        return response.json()['response']
    except requests.exceptions.Timeout: #
        return "The request to the LLM timed out. Please try again."

#streamlit App UI
st.title("Speech Explorer - RAG with LLaMA3.2 and ChromaDB")
query = st.text_input("Enter a topic or a speaker's name")

if query:
    #initialize ChromaDB and LLaMA3.2 embedding
    embedding_fn = OllamaEmbeddingFunction(model_name="llama3.2:1b") #llama3.2:latest may be too resource-demanding
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="speeches", embedding_function=embedding_fn) #vector database

    #load data and populate ChromaDB only if not already populated
    if collection.count() == 0:
        with st.spinner("Indexing documents, please wait..."):
            data = load_data() 
            embed_and_store(data, collection) 

    # Perform retrieval
    with st.spinner("Querying documents..."):
        results = query_chroma(collection, query)
        relevant_contexts = results['documents'][0]  # get top 10 documents

    # Generate answer
    with st.spinner("Generating response from LLaMA3.2..."):
        answer = generate_response(relevant_contexts, query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Top 10 Relevant Excerpts")
    for i, context in enumerate(relevant_contexts):
        st.markdown(f"**{i+1}.** {context}")
