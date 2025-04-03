import os
import json
import urllib.request
import zipfile
from pathlib import Path
import shutil
from datasets import Dataset, Features, Image, Value

# URLs for the validation subset
VAL_SUBSET_URLS = {
    "questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "images": "http://images.cocodataset.org/zips/val2014.zip",
}

# Define paths
data_dir = Path("/mnt/huggingface_cache/vqa_data")
questions_zip = data_dir / "questions.zip"
annotations_zip = data_dir / "annotations.zip"
images_zip = data_dir / "images.zip"

questions_path = data_dir / "v2_OpenEnded_mscoco_val2014_questions.json"
annotations_path = data_dir / "v2_mscoco_val2014_annotations.json"
images_path = data_dir / "val2014"

def download_and_extract():
    """Download and extract the VQA dataset files."""
    print(f"Creating data directory at {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    
    # Download files if they don't exist
    for name, url in VAL_SUBSET_URLS.items():
        zip_path = data_dir / f"{name}.zip"
        if not zip_path.exists():
            print(f"Downloading {name} data from {url}")
            urllib.request.urlretrieve(url, zip_path)
        else:
            print(f"{name} data already downloaded")
    
    # Extract questions
    if not questions_path.exists():
        print("Extracting questions...")
        with zipfile.ZipFile(questions_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    # Extract annotations
    if not annotations_path.exists():
        print("Extracting annotations...")
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    # Extract images only if needed
    if not images_path.exists():
        print("Extracting images (this may take a while)...")
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    else:
        print("Images already extracted")

def load_vqa_subset():
    """Load the VQA subset data."""
    print(f"Loading questions from {questions_path}")
    with open(questions_path, "r") as f:
        questions_data = json.load(f)
    
    print(f"Loading annotations from {annotations_path}")
    with open(annotations_path, "r") as f:
        annotations_data = json.load(f)
    
    # Create a lookup for annotations by question_id
    annotations_lookup = {ann["question_id"]: ann for ann in annotations_data["annotations"]}
    
    examples = []
    count = 0
    max_examples = 600
    
    print("Processing examples...")
    for question in questions_data["questions"]:
        if count >= max_examples:
            break
        
        question_id = question["question_id"]
        if question_id in annotations_lookup:
            annotation = annotations_lookup[question_id]
            
            # Combine question and annotation info
            record = {**question, **annotation}
            
            # Construct the image path
            image_filename = f"COCO_val2014_{record['image_id']:012d}.jpg"
            image_path = str(images_path / image_filename)
            
            # Check if the image actually exists
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                continue
            
            # Store both the image and its path
            record["image"] = image_path
            record["image_file_path"] = image_path
            
            examples.append(record)
            count += 1
    
    print(f"Collected {count} examples")
    return examples

def main():
    print("Starting VQA dataset loading script...")
    download_and_extract()
    examples = load_vqa_subset()
    
    # Define the features for the dataset
    features = Features({
        "question_type": Value("string"),
        "multiple_choice_answer": Value("string"),
        "answers": [
            {
                "answer": Value("string"),
                "answer_confidence": Value("string"),
                "answer_id": Value("int64"),
            }
        ],
        "image_id": Value("int64"),
        "answer_type": Value("string"),
        "question_id": Value("int64"),
        "question": Value("string"),
        "image": Image(),  # This will load the image from the file path
        "image_file_path": Value("string"),  # This will store the original file path as a string
    })
    
    # Create the dataset
    print("Creating dataset...")
    val_subset_dataset = Dataset.from_list(examples, features=features)
    
    print(f"Number of examples loaded: {len(val_subset_dataset)}")
    
    # Access an example
    first_example = val_subset_dataset[0]
    print("\nFirst example:")
    print(f"  Question ID: {first_example['question_id']}")
    print(f"  Question: {first_example['question']}")
    print(f"  Image File Path: {first_example['image_file_path']}")
    print(f"  Loaded Image: {first_example['image']}")
    print(f"  MC Answer: {first_example['multiple_choice_answer']}")

if __name__ == "__main__":
    main()