import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

def train_tinybert_for_qa():
    """
    Fine-tunes a TinyBERT model for extractive question-answering on a custom dataset.
    """
    
    # 1. Load the dataset from the 'dataset' folder
    print("Loading dataset...")
    try:
        # Load the JSON file. Assumes the file is named 'my_qa_data.json'
        # and is inside a 'dataset' folder.
        dataset = load_dataset('json', data_files='dataset/my_qa_data.json', split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure 'dataset/my_qa_data.json' exists and is in the correct format.")
        return
        
    print("Dataset loaded successfully.")

    # 2. Load the tokenizer and model
    model_name = "indolem/indobert-base-uncased"
    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    print("Tokenizer and model loaded.")

    # 3. Preprocess the dataset
    print("Tokenizing the dataset...")
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=512,
            truncation="only_second",
            padding="max_length",
        )
        
        # Tokenize the answers to get the correct start and end positions
        answer_texts = [ans['text'][0] for ans in examples["answers"]]
        answer_starts = [ans['answer_start'][0] for ans in examples["answers"]]

        # Add the 'start_positions' and 'end_positions' labels
        inputs.update({
            'start_positions': answer_starts,
            'end_positions': [start + len(text) for start, text in zip(answer_starts, answer_texts)]
        })
        
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    print("Dataset tokenized.")
    
    # 4. Define training arguments and initialize the Trainer
    output_dir = "./tinybert_qa_finetuned"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",  # Save the model at the end of each epoch
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # 5. Start the fine-tuning process
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    # 6. Save the fine-tuned model
    trainer.save_model(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    train_tinybert_for_qa()