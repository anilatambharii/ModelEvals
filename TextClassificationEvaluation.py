import random
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define evaluation dataset

# Define categories and sample texts
categories = ['Technology', 'Sports', 'Politics', 'Entertainment']

sample_texts = {
    'Technology': [
        "New AI model breaks performance records in image recognition tasks.",
        "The latest smartphone features a foldable screen and 5G capabilities.",
        "Quantum computing breakthrough promises faster drug discovery process.",
        "Cybersecurity experts warn of increasing sophisticated phishing attempts.",
        "Electric vehicle sales surpass traditional cars in major European markets."
    ],
    'Sports': [
        "Underdog team clinches championship in thrilling overtime victory.",
        "Star athlete announces retirement after record-breaking career.",
        "New study reveals long-term effects of concussions in contact sports.",
        "International soccer tournament draws record viewership worldwide.",
        "College basketball player sets new scoring record in conference game."
    ],
    'Politics': [
        "Historic peace agreement signed between long-time rival nations.",
        "Controversial bill passes senate after heated debate.",
        "Local elections see unprecedented turnout among young voters.",
        "Government announces new initiative to combat climate change.",
        "Political scandal leads to high-profile resignations in capital."
    ],
    'Entertainment': [
        "Blockbuster movie breaks box office records on opening weekend.",
        "Popular streaming series renewed for three more seasons.",
        "Legendary musician announces comeback tour after decade-long hiatus.",
        "Virtual reality technology revolutionizes the gaming industry.",
        "Award-winning author releases highly anticipated sequel novel."
    ]
}

# Generate the evaluation dataset
eval_dataset = []

for _ in range(100):  # Generate 100 samples
    category = random.choice(categories)
    text = random.choice(sample_texts[category])
    eval_dataset.append((text, category))

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

