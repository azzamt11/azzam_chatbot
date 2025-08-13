import gradio as gr
from model.chatbot import IndoBertChatbot
import os

# Set a confidence threshold for the chatbot's answers
CONFIDENCE_THRESHOLD = 5.0

# Define the paths for the fine-tuned model and the context file
# Make sure the `fine-tuned-model` folder exists after you run train.py
model_path = "fine-tuned-model"
context_path = os.path.join(os.path.dirname(__file__), 'dataset', 'my_descriptions.txt')

# Initialize the chatbot using the fine-tuned model
print("Initializing the chatbot...")
try:
    chatbot_instance = IndoBertChatbot(model_name=model_path, context_file_path=context_path)
    print("Chatbot is ready.")
except Exception as e:
    print(f"Error during chatbot initialization: {e}")
    chatbot_instance = None

def chat_function(question):
    """
    The function that Gradio will use to power the chatbot UI.
    """
    if not question:
        return "Please enter a question."
    
    if chatbot_instance is None:
        return "Chatbot is not available due to an initialization error."

    answer, confidence = chatbot_instance.get_answer(question)

    if confidence < CONFIDENCE_THRESHOLD:
        return "Maaf, saya tidak memiliki informasi tentang topik tersebut."
    
    return answer

# Create the Gradio interface
iface = gr.Interface(
    fn=chat_function, 
    inputs="text", 
    outputs="text",
    title="IndoBert Chatbot",
    description="Ask me questions about myself in Bahasa Indonesia."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)