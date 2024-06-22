from flask import Flask, request, render_template
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from util import FineTunedBERT, getEmbeddings
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset

app = Flask(__name__)

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load precomputed embeddings
print("Loading precomputed embeddings...")
gene_embeddings = pd.read_csv('gene_embeddings.csv')
disease_embeddings = pd.read_csv('disease_embeddings.csv')
print("Embeddings loaded.")

# Function to load model and filter state_dict
def load_model_with_filtered_state_dict(model_class, state_dict_path, device):
    model = model_class(pool="mean", model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", device=device)
    state_dict = torch.load(state_dict_path, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if "gene2vecFusion" not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    return model

# Load model
print("Loading model...")
model = load_model_with_filtered_state_dict(FineTunedBERT, 'state_dict_0.pth', device)
model.eval()
print("Model loaded.")

# Define a function to compute embeddings using util.py
def compute_embeddings(text, model, max_length=512, batch_size=16):
    tokenizer = BertTokenizerFast.from_pretrained(model.model_name)
    tokens = tokenizer.batch_encode_plus(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    model.eval()
    for batch_input_ids, batch_attention_mask in dataloader:
        with torch.no_grad():
            pooled_embeddings, _, _ = model(batch_input_ids.to(device), batch_attention_mask.to(device))
            embeddings.append(pooled_embeddings)

    concat_embeddings = torch.cat(embeddings, dim=0)
    return concat_embeddings

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    embedding = compute_embeddings([text], model).detach().cpu().numpy()

    gene_similarities = cosine_similarity(embedding, gene_embeddings.iloc[:, 1:].values)
    disease_similarities = cosine_similarity(embedding, disease_embeddings.iloc[:, 1:].values)

    top_genes = gene_embeddings.iloc[np.argsort(-gene_similarities[0])[:10]]['Gene name'].tolist()
    top_diseases = disease_embeddings.iloc[np.argsort(-disease_similarities[0])[:10]]['Disease'].tolist()

    return render_template('result.html', top_genes=top_genes, top_diseases=top_diseases)

if __name__ == '__main__':
    app.run(debug=True)
