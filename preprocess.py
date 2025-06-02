import pandas as pd 
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Load the data
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# Add labels
df_fake["label"] = 1  # FAKE = 1
df_real["label"] = 0  # REAL = 0

# Combine and shuffle
df = pd.concat([df_fake, df_real]).sample(frac=1).reset_index(drop=True)

# Choose either "title" or "text" as input (we'll use title for now)
df = df[["title", "label"]]
df.rename(columns={"title": "text"}, inplace=True)  # rename for consistency

# Split into train/test sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine into one dictionary
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

print(tokenized_dataset)
