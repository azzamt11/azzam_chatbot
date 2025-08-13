import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class IndoBertChatbot:
    """
    A chatbot that answers questions based on a single, pre-loaded context.
    It uses a fine-tuned IndoBERT model for extractive question-answering.
    """
    def __init__(self, model_name, context_file_path):
        # Load the tokenizer and model.
        # The model_name will be the path to your fine-tuned model folder.
        print(f"Loading tokenizer and model from: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        print("Model and tokenizer loaded successfully.")
        
        # Load the context (the long paragraph) once when the chatbot is initialized
        try:
            with open(context_file_path, 'r', encoding='utf-8') as f:
                self.context = f.read()
            print("Context loaded from file successfully.")
        except FileNotFoundError:
            print(f"Error: Context file '{context_file_path}' not found.")
            self.context = ""

    def get_answer(self, question):
        """
        Takes a question and finds the answer within the pre-loaded context,
        returning the answer and a confidence score.
        
        Args:
            question (str): The question to be answered.
            
        Returns:
            tuple: (str, float) - The extracted answer and its confidence score.
        """
        if not self.context:
            return "Error: Context not available.", 0.0

        inputs = self.tokenizer(question, self.context, return_tensors='pt', truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get start and end logits for the answer
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        answer_start_index = torch.argmax(start_logits)
        answer_end_index = torch.argmax(end_logits)
        
        # Calculate a confidence score from the maximum logits.
        confidence_score = (torch.max(start_logits) + torch.max(end_logits)).item()

        # If the start index is after the end index, the answer is invalid
        if answer_start_index > answer_end_index:
            return "Sorry, I couldn't find a relevant answer in the text.", 0.0
        
        input_ids = inputs['input_ids'].squeeze(0)
        answer_tokens = input_ids[answer_start_index : answer_end_index + 1]
        
        answer = self.tokenizer.decode(answer_tokens)
        
        if not answer or self.tokenizer.decode(answer_tokens).strip() in self.tokenizer.all_special_tokens:
            return "Sorry, I couldn't find a relevant answer in the text.", 0.0
            
        return answer, confidence_score