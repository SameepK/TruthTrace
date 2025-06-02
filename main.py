from transformers import pipeline
from transformers import AutoTokenizer

classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

headline = "Breaking: Scientists confirm that unicorns are real"
result = classifier(headline)

# Load the tokenizer from the same model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-fake-news-detection")

# Tokenize it
tokens = tokenizer.tokenize(headline)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Original Text:", headline)
print("Tokens:", tokens)
print("Token IDs:", token_ids)

print("Headline:", headline)
print("Prediction:", result[0]['label'])
print("Confidence:", round(result[0]['score'] * 100, 2), "%")
