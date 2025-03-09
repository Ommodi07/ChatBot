from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json

app = Flask(__name__)

# Set your Google API key (Replace with your actual key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCOsco3wW-yHA074FTp-Mbz8NgUptGUY_8"

# Load a smaller, memory-efficient embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index from disk if available
vectordb = None
PDF_PATH = "CirrhosisToolkit.pdf"
FAISS_INDEX_PATH = "faiss_index"

def process_pdf(pdf_path):
    global vectordb

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Load only a limited number of pages to reduce memory
    documents = documents[:10]  # Adjust as needed

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    vectordb = FAISS.from_documents(texts, embeddings)
    
    # Save FAISS index to disk
    vectordb.save_local(FAISS_INDEX_PATH)

# Load FAISS index from disk if it exists
if os.path.exists(FAISS_INDEX_PATH):
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    process_pdf(PDF_PATH)

@app.route('/query', methods=['POST'])
def query_pdf():
    global vectordb
    if vectordb is None:
        return jsonify({'error': 'No PDF processed yet'}), 400
    
    data = request.get_json()
    query_text = data.get("query", "")
    if not query_text:
        return jsonify({'error': 'No query provided'}), 400
    
    results = vectordb.similarity_search(query_text, k=3)
    context = "\n".join([res.page_content for res in results])
    
    prompt = f"Using the following document context, answer the query concisely.\n\nContext:\n{context}\n\nQuery: {query_text}" 
    gemini_response = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7).invoke(prompt)
    
    return jsonify({'response': gemini_response.content})

if __name__ == '__main__':
    app.run(debug=True)
