import requests
import pickle
import faiss
from flask import Flask, request, jsonify
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os

app = Flask(__name__)

# Google Drive file ID
FILE_ID = "1q2PO9XUtKePArjemadwwpY6wK_w7IAzH"  # Replace with your actual Google Drive file ID
MODEL_PATH = "model.pkl"

# Function to download model from Google Drive
def download_model():
    if os.path.exists(MODEL_PATH):  # Check if file already exists
        print("Model file already exists, skipping download.")
        return

    print("Downloading model from Google Drive...")
    
    URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    session = requests.Session()
    response = session.get(URL, stream=True)

    # Handle Google Drive warning for large files
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm={value}"
            response = session.get(URL, stream=True)

    # Save the downloaded file
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
    
    print("Model downloaded successfully.")

# Ensure model is downloaded
download_model()

# ✅ Verify if the downloaded file is valid
try:
    with open(MODEL_PATH, "rb") as f:
        vectordb = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    print("The downloaded file may be corrupted or not a valid pickle file.")
    exit(1)

# Load FAISS Index
index = faiss.read_index("faiss_index.index")

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
