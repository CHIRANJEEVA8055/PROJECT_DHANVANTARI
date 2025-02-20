import os
import pandas as pd
import re
import json

# Define the base project directory
BASE_DIR = "C:/Users/INDUSTRY 4.0/PycharmProjects/project_dhanvantari"

# Define NLP data paths
RAW_NLP_DATA_DIR = os.path.join(BASE_DIR, "data", "nlp", "raw")
PROCESSED_NLP_DATA_DIR = os.path.join(BASE_DIR, "data", "nlp", "processed")
NLP_MODEL_DIR = os.path.join(BASE_DIR, "models", "nlp")

# Ensure directories exist
os.makedirs(PROCESSED_NLP_DATA_DIR, exist_ok=True)
os.makedirs(NLP_MODEL_DIR, exist_ok=True)

print(f"Raw NLP Data Path: {RAW_NLP_DATA_DIR}")
print(f"Processed NLP Data Path: {PROCESSED_NLP_DATA_DIR}")
print(f"NLP Model Path: {NLP_MODEL_DIR}")


# Function to clean text data
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = text.strip()  # Remove leading and trailing spaces
    return text


# Function to load and process text data
def process_nlp_data():
    print("Processing NLP data...")

    for file_name in os.listdir(RAW_NLP_DATA_DIR):
        file_path = os.path.join(RAW_NLP_DATA_DIR, file_name)

        if file_name.endswith(".csv"):
            print(f"Processing CSV file: {file_name}")
            df = pd.read_csv(file_path)
            if "text" in df.columns:
                df["text"] = df["text"].apply(clean_text)
            processed_file_path = os.path.join(PROCESSED_NLP_DATA_DIR, f"processed_{file_name}")
            df.to_csv(processed_file_path, index=False)

        elif file_name.endswith(".json"):
            print(f"Processing JSON file: {file_name}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    if "text" in entry:
                        entry["text"] = clean_text(entry["text"])
            processed_file_path = os.path.join(PROCESSED_NLP_DATA_DIR, f"processed_{file_name}")
            with open(processed_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

        else:
            print(f"Skipping unsupported file format: {file_name}")

    print("NLP data processing completed.")


if __name__ == "__main__":
    process_nlp_data()
