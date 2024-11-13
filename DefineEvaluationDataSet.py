import random
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
    

# At the end of DefineEvaluationDataSet.py, add:
if __name__ == "__main__":
    # This block will only run if this script is executed directly
    print(f"Generated {len(eval_dataset)} samples")