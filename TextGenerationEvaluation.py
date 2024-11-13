from DefineEvaluationDataSet import eval_dataset, categories
import random
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download necessary NLTK data
nltk.download('punkt')

# Load pre-trained model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

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
    scores = []
    smoothie = SmoothingFunction().method1  # Use smoothing
    
    for text, category in eval_dataset:
        input_text = f"summarize: {text}"
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
        
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=150, num_return_sequences=1, do_sample=True)
        
        predicted_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate BLEU score
        reference = [nltk.word_tokenize(text)]  # Tokenize the original text
        candidate = nltk.word_tokenize(predicted_summary)  # Tokenize the predicted summary
        
        # Use lower n-gram order (1-gram and 2-gram only) and smoothing
        bleu_score = sentence_bleu(reference, candidate, weights=(0.5, 0.5), smoothing_function=smoothie)
        scores.append(bleu_score)
    
    average_bleu = sum(scores) / len(scores)
    print("Average BLEU Score:", average_bleu)

# Evaluate model
evaluate(model, eval_dataset)