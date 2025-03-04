from flask import Flask, request, jsonify
import pickle
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Initialize Flask
app = Flask(__name__)

# Load FAISS Index
index = faiss.read_index("faiss_index.index")

# Load FAISS Store (documents + embeddings)
with open("faiss_store.pkl", "rb") as f:
    vectordb = pickle.load(f)

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
