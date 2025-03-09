from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json

app = Flask(__name__)

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCOsco3wW-yHA074FTp-Mbz8NgUptGUY_8"  # Replace with your actual API key

# Use a lightweight embedding model to reduce memory usage
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Google Gemini LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# FAISS index path (stored on disk)
FAISS_INDEX_PATH = "faiss_index"

# PDF path
PDF_PATH = "CirrhosisToolkit.pdf"


def process_pdf(pdf_path):
    """Processes the PDF and stores FAISS index on disk to save memory."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.lazy_load()  # Lazy load to avoid high memory usage

    # Reduce chunk size to save memory
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)

    # Store FAISS on disk to avoid keeping everything in RAM
    vectordb = FAISS.from_documents(texts, embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)


# Load FAISS index if available, else process the PDF
if os.path.exists(FAISS_INDEX_PATH):
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    process_pdf(PDF_PATH)
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


@app.route('/query', methods=['POST'])
def query_pdf():
    """Handles user queries and retrieves relevant document context."""
    if vectordb is None:
        return jsonify({'error': 'No PDF processed yet'}), 400
    
    data = request.get_json()
    query_text = data.get("query", "")
    if not query_text:
        return jsonify({'error': 'No query provided'}), 400
    
    # Perform similarity search with reduced results (k=2) to save memory
    results = vectordb.similarity_search(query_text, k=2)
    context = "\n".join([res.page_content for res in results])
    
    # Generate response with Gemini
    prompt = f"Using the following document context, answer the query concisely.\n\nContext:\n{context}\n\nQuery: {query_text}"
    gemini_response = llm.invoke(prompt)
    
    return jsonify({'response': gemini_response.content})


if __name__ == '__main__':
    app.run(debug=True)
