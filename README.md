# ModelEvals

Gen AI Evaluation

Introduction

  - This repository provides example code for evaluating Gen AI models on three common NLP tasks:

Use Cases
 
  - Text Classification
      - Evaluate the accuracy of a text classification model using the DistilBERT model.
      - Dataset: SST-2 (Stanford Sentiment Treebank)
      - Metric: Accuracy
  
  - Text Generation
      - Evaluate the quality of generated text using the T5 model.
      - Dataset: WikiText-103
      - Metric: BLEU Score
  
  - Question Answering
      - Evaluate the accuracy of a question answering model using the DistilBERT model.
      - Dataset: SQuAD (Stanford Question Answering Dataset)
      - Metric: Accuracy
  
  - Requirements
      - Python 3.8+
      - Transformers library (Hugging Face)
      - PyTorch 1.9+
      - NLTK library (for BLEU score calculation)
  
  - Installation
      - Bash
      - pip install transformers torch nltk
  
  - Usage
      - Clone the repository: git clone https://github.com/your-repo/gen-ai-evaluation.git
      - Navigate to the desired use case directory: cd text-classification, cd text-generation, or cd question-answering
      - Run the evaluation script: python evaluate.py
  
  - Evaluation Metrics
      - Accuracy (Text Classification, Question Answering)
      - BLEU Score (Text Generation)

- License

- MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Additional Resources:
1) Hugging Face Transformers documentation: https://huggingface.co/docs/transformers/index 
2) Gen AI documentations and 
3) Evaluation metrics documentation:https://scikit-learn.org/stable/modules/model_evaluation.html 
