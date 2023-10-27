from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel
import matplotlib.pyplot as plt

# Load dataset
emotions = load_dataset('emotion')


# Functions
def print_lbl_distribution(data):
    def label_int2str(row):
        return emotions['train'].features['label'].int2str(row)

    data.set_format(type='pandas')
    df = emotions['train'][:]

    df['label_name'] = df['label'].apply(label_int2str)

    df['label_name'].value_counts(ascending=False).plot.barh()
    plt.title('Frequency of classes')
    plt.show()

# Exploration
print_lbl_distribution(emotions)

#### Tokenization
mdl_ckpt = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(mdl_ckpt)

def tokenize(batch):
    return tokenizer(batch, padding=True, truncation=False)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)




