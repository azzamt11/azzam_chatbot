import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

def train_qa_model():
    """
    Fine-tunes a model for extractive question-answering.
    """
    # 1. Load the dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset('json', data_files='dataset/my_qa_data.json', split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
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
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    print("Dataset tokenized.")
    
    # 4. Define training arguments and initialize the Trainer
    output_dir = "./fine-tuned-model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=4, # A smaller batch size for CPU/limited GPU
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
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
    train_qa_model()