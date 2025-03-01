from flask import Flask, request, jsonify
from langchain_groq import ChatGroq

app = Flask(__name__)

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key="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"
)

# Define function for chatbot response
def generate_summary_and_quiz(question):
    prompt = f"""
    {question}
    Based on the text above, answer the question.
    You are a medical chatbot designed to help with health concerns related to liver cirrhosis and some basic medical questions.
    If the question is out of this topic, respond with: "I am sorry, I can't help with this question."
    """
    
    response = llm.invoke(prompt)
    return response.content

# Define Flask route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    answer = generate_summary_and_quiz(question)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
