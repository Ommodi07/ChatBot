from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
import torch
from dotenv import load_dotenv
import gc

app = Flask(__name__)

load_dotenv()
# Retrieve Google API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Use a more memory-efficient embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Google Gemini LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7, google_api_key=GOOGLE_API_KEY)

# FAISS index path (stored on disk)
FAISS_INDEX_PATH = "faiss_index"
PDF_PATH = "document.pdf"

def process_pdf(pdf_path):
    """Processes the PDF and stores FAISS index on disk."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    vectordb = FAISS.from_documents(texts, embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)
    del texts, documents  # Free memory
    gc.collect()

# Load FAISS index in read-only mode if available
vectordb = None
if os.path.exists(FAISS_INDEX_PATH):
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    process_pdf(PDF_PATH)
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def get_vectordb():
    """Load FAISS database only when needed."""
    global vectordb
    if vectordb is None:
        vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectordb

@app.route('/query', methods=['POST'])
def query_pdf():
    """Handles user queries and retrieves the entire document as context."""
    vectordb = get_vectordb()
    if vectordb is None:
        return jsonify({'error': 'No PDF processed yet'}), 400
    
    data = request.get_json()
    query_text = data.get("query", "").strip()
    if not query_text:
        return jsonify({'error': 'No query provided'}), 400
    
    # Retrieve all document content
    results = vectordb.similarity_search(query_text, k=10)  # Adjust k as needed
    full_context = [res.page_content for res in results]
    
    # Generate response with Gemini
    prompt = f"""Context: {full_context}
                Learn the context properly and answer the query concisely as you are a healthcare chatbot.
                You can reply to greetings but should stick to healthcare-related responses.
                Question: {query_text}
                If you don't know the answer, you can say 'I can't help you with that.'"""
    gemini_response = llm.invoke(prompt)
    
    del results, full_context  # Free memory
    gc.collect()
    
    return jsonify({'response': gemini_response.content})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
