import random
from DefineEvaluationDataSet import eval_dataset, categories
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define evaluation dataset
# Shuffle the dataset
random.shuffle(eval_dataset)

# Print a few examples to verify
print("Sample entries from eval_dataset:")
for i in range(5):
    print(f"{i+1}. Text: '{eval_dataset[i][0]}'\n   Category: {eval_dataset[i][1]}\n")

print(f"Total samples in eval_dataset: {len(eval_dataset)}")

# Define evaluation function
def evaluate(model, eval_dataset):
    model.eval()
    predictions = []
    labels = []
    
    for text, category in eval_dataset:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1)
        predictions.append(pred.item())
        labels.append(categories.index(category))
    
    accuracy = accuracy_score(labels, predictions)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names=categories))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, predictions))

# Evaluate model
evaluate(model, eval_dataset)

