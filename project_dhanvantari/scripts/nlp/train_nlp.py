import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Define paths
PROCESSED_DATA_PATH = "C:/Users/INDUSTRY 4.0/PycharmProjects/project_dhanvantari/data/nlp/processed"
MODEL_SAVE_PATH = "C:/Users/INDUSTRY 4.0/PycharmProjects/project_dhanvantari/models/nlp"
MODEL_NAME = "dmis-lab/biobert-base-cased"

# Load dataset (Ensure preprocessed dataset exists)
def load_data():
    dataset = load_dataset("csv", data_files={"train": os.path.join(PROCESSED_DATA_PATH, "train.csv"),
                                               "test": os.path.join(PROCESSED_DATA_PATH, "test.csv")})
    return dataset

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

# Load and tokenize dataset
dataset = load_data()
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(MODEL_SAVE_PATH)
    print("NLP Model training complete and saved!")
