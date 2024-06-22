import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import collections
import pandas as pd
from tqdm import tqdm

class FineTunedBERT(nn.Module):

    def __init__(self, pool="mean", model_name="bert-base-cased", device="cuda"):
        super(FineTunedBERT, self).__init__()
        self.model_name = model_name
        self.pool = pool
        self.device = device
        
        self.bert = AutoModel.from_pretrained(model_name).to(device)
        self.bert_hidden = self.bert.config.hidden_size
        self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, 1))

    def forward(self, input_ids_, attention_mask_):
        hiddenState, ClsPooled = self.bert(input_ids=input_ids_, attention_mask=attention_mask_).values()

        # Perform pooling on the hidden state embeddings
        if self.pool.lower() == "max":
            embeddings = self.max_pooling(hiddenState, attention_mask_)
        elif self.pool.lower() == "cls":
            embeddings = ClsPooled
        elif self.pool.lower() == "mean":
            embeddings = self.mean_pooling(hiddenState, attention_mask_)
        else:
            raise ValueError('Invalid pooling method.')

        return embeddings, hiddenState, self.pipeline(embeddings)

    def max_pooling(self, hidden_state, attention_mask):
        token_embeddings = hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        pooled_embeddings = torch.max(token_embeddings, 1)[0]
        return pooled_embeddings
    
    def mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        pooled_embeddings = torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return pooled_embeddings

def getEmbeddings(text, model=None, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                  max_length=512, batch_size=1000, gene2vec_flag=False, gene2vec_hidden=200, pool="mean"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(model, FineTunedBERT):
        print("Loading a pretrained model ...")
        model_name = model.model_name            
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    elif isinstance(model, collections.OrderedDict):
        print("Loading a pretrained model from a state dictionary ...")
        state_dict = model.copy() 
        model = FineTunedBERT(pool=pool, model_name=model_name, device=device).to(device)
        model.load_state_dict(state_dict)
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    else:
        print("Creating a new pretrained model ...")
        model = FineTunedBERT(pool=pool, model_name=model_name, device=device).to(device)
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    model = nn.DataParallel(model)
    print("Tokenization ...")
    tokens = tokenizer.batch_encode_plus(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    
    dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Tokenization Done.")
    print("Get Embeddings ...")
    
    embeddings = []
    model.eval()
    for batch_input_ids, batch_attention_mask in tqdm(dataloader):
        with torch.no_grad():
            pooled_embeddings, _, _ = model(batch_input_ids.to(device), batch_attention_mask.to(device))
            embeddings.append(pooled_embeddings)
    
    concat_embeddings = torch.cat(embeddings, dim=0)
    print(concat_embeddings.size())
    return concat_embeddings

def getEmbeddingsWithGene2Vec(dataloader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []
    model.eval()
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch_inputs_a, batch_masks_a, gene2vec_a = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            pooled_embeddings, _, _ = model(batch_inputs_a, batch_masks_a, gene2vec=gene2vec_a)
            embeddings.append(pooled_embeddings)
    
    concat_embeddings = torch.cat(embeddings, dim=0)
    print(concat_embeddings.size())
    return concat_embeddings
