import gradio as gr
from model.chatbot import IndoBertChatbot
import os

# Set up the chatbot and confidence threshold
CONFIDENCE_THRESHOLD = 5.0
context_path = os.path.join(os.path.dirname(__file__), 'dataset', 'my_descriptions.txt')
chatbot_instance = IndoBertChatbot(model_name="indolem/indobert-base-uncased", context_file_path=context_path)

def chat_function(question):
    """
    The function that Gradio will use to power the chatbot UI.
    It takes a question and returns the answer.
    """
    if not question:
        return "Please enter a question."

    answer, confidence = chatbot_instance.get_answer(question)

    if confidence < CONFIDENCE_THRESHOLD:
        return "Maaf, saya tidak memiliki informasi tentang topik tersebut."

    return answer

# Create the Gradio interface
iface = gr.Interface(
    fn=chat_function, 
    inputs="text", 
    outputs="text",
    title="Azzam Chatbot",
    description="Tanya saya apapun tentang Azzam"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()