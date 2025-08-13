from flask import Flask, request, jsonify
from model.chatbot import IndoBertChatbot
import os

# Initialize the Flask application
app = Flask(__name__)

# --- IMPORTANT: Load the chatbot model once when the app starts ---
# This saves a lot of time by not reloading the model for every request.
print("Initializing the chatbot...")
try:
    # Use the path relative to the app.py file
    context_path = os.path.join(os.path.dirname(__file__), 'dataset', 'my_descriptions.txt')
    chatbot = IndoBertChatbot(model_name="indolem/indobert-base-uncased", context_file_path=context_path)
    print("Chatbot is ready.")
except Exception as e:
    print(f"Error during chatbot initialization: {e}")
    chatbot = None # Set to None to handle errors gracefully

@app.route('/chat', methods=['POST'])
def chat():
    """
    API endpoint for the chatbot. Expects a JSON payload with a 'question' key.
    """
    if chatbot is None:
        return jsonify({"error": "Chatbot is not initialized."}), 500

    # Ensure the request has JSON data
    if not request.json or 'question' not in request.json:
        return jsonify({"error": "Please provide a 'question' in the request body."}), 400

    question = request.json['question']

    # Get the answer from the chatbot
    answer = chatbot.get_answer(question)

    # Return the answer as a JSON response
    return jsonify({"answer": answer})

@app.route('/', methods=['GET'])
def home():
    """
    A simple home route to confirm the API is running.
    """
    return "The IndoBert Chatbot API is running!"

if __name__ == '__main__':
    # Run the Flask app on host 0.0.0.0 to make it accessible from outside
    # and on a specific port, e.g., 5000
    app.run(host='0.0.0.0', port=5000)