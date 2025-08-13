import json
from transformers import pipeline

def generate_qa_from_text(raw_text_path, output_json_path, model_name='valhalla/t5-base-qg-hl'):
    """
    Generates a question-answer dataset from a raw text file using a pre-trained model.
    """
    try:
        # Load the raw text
        with open(raw_text_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Initialize the question generation pipeline
        print("Loading question generation model...")
        qg_pipeline = pipeline("text2text-generation", model=model_name)
        print("Model loaded. Generating Q&A pairs...")

        # The model requires a highlight, so we'll use a placeholder for now
        # and then find the answer start and end in the raw text after generation.
        # Note: This is a heuristic. For better quality, a more robust method is needed.
        # We assume the generated answer is a substring of the raw text.
        generated_qa_text = qg_pipeline(f"highlight: The quick brown fox jumps over the lazy dog. context: {raw_text}")[0]['generated_text']

        qa_data = []
        qa_pairs = generated_qa_text.split('<pad>')

        for i, pair in enumerate(qa_pairs):
            pair = pair.strip()
            if not pair:
                continue
            
            parts = pair.split('</s>')
            if len(parts) == 2:
                question = parts[0].strip().replace('<pad>', '').replace('question: ', '').replace('question:', '')
                answer = parts[1].strip()
                
                # Find the answer_start index
                try:
                    answer_start = raw_text.index(answer)
                    qa_data.append({
                        "id": str(i),
                        "context": raw_text,
                        "question": question,
                        "answers": {
                            "text": [answer],
                            "answer_start": [answer_start]
                        }
                    })
                except ValueError:
                    # If the answer isn't an exact substring, skip it
                    continue

        # Save the dataset to a JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2)

        print(f"Dataset with {len(qa_data)} Q&A pairs created and saved to {output_json_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Ensure all required libraries are installed and your text file is correctly formatted.")

if __name__ == "__main__":
    generate_qa_from_text(
        raw_text_path="dataset/my_descriptions.txt",
        output_json_path="dataset/my_qa_data.json"
    )