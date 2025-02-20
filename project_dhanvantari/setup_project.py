import os

PROJECT_DIR = "C:\\Users\\INDUSTRY 4.0\\PycharmProjects\\project_dhanvantari"

FOLDERS = [
    "data/cnn/raw", "data/cnn/processed",
    "data/nlp/raw", "data/nlp/processed",
    "models/cnn", "models/nlp", "models/fusion",
    "src/cnn_model", "src/nlp_model", "src/fusion_model", "src/utils",
    "scripts"
]

for folder in FOLDERS:
    os.makedirs(os.path.join(PROJECT_DIR, folder), exist_ok=True)

files = {
    "README.md": "# Project Dhanvantari\n\nThis is a dermatology diagnosis project.",
    "requirements.txt": "# Add required dependencies here",
}

for file, content in files.items():
    with open(os.path.join(PROJECT_DIR, file), "w") as f:
        f.write(content)

print("Project structure created successfully.")