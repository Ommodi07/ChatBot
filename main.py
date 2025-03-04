import os
import pickle
import faiss
import gdown
import torch  # Import torch for model loading
from flask import Flask, request, jsonify
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Google Drive file IDs
MODEL_FILE_ID = "1q2PO9XUtKePArjemadwwpY6wK_w7IAzH"  # Replace with actual Google Drive file ID
FAISS_INDEX_FILE = "faiss_index.index"
MODEL_PATH = "model.pkl"

# Function to download model file
def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model file already exists, skipping download.")
        return
    
    print("Downloading model from Google Drive...")
    gdown.download(id=MODEL_FILE_ID, output=MODEL_PATH, quiet=False)

# Download the model if not available
download_model()

# Verify the downloaded file and load it on CPU
try:
    with open(MODEL_PATH, "rb") as f:
        vectordb = torch.load(f, map_location=torch.device("cpu"))  # ✅ Load on CPU
        print("Model file loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("The downloaded file may be corrupted or not a valid pickle file.")
    exit(1)

# Load FAISS Index
if not os.path.exists(FAISS_INDEX_FILE):
    print(f"Error: FAISS index file '{FAISS_INDEX_FILE}' not found.")
    exit(1)

index = faiss.read_index(FAISS_INDEX_FILE)

# Load LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# Setup Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Setup LangChain RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get("question")

    if not query:
        return jsonify({"error": "No question provided"}), 400

    # Get response from RAG model
    result = qa_chain.run(query)
    return jsonify({"answer": result})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
