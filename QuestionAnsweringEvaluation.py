from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import random
from DefineEvaluationDataSet import eval_dataset, categories

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def evaluate(model, eval_dataset):
    model.eval()
    correct = 0
    total = len(eval_dataset)
    
    for text, true_category in eval_dataset:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        # Map the predicted class to your categories
        predicted_category = categories[predicted_class % len(categories)]
        
        if predicted_category == true_category:
            correct += 1
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2f}")

evaluate(model, eval_dataset)