import torch
import ankh
import torch.nn as nn
import math
import os
import pickle
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
# ------------------------------------------------
# PyTorch Model Definitions
# ------------------------------------------------

class AffinityLM(nn.Module):
    def __init__(self, embedding_dim, linear_dim, num_attention_layers, num_heads, dropout_rate):
        super(AffinityLM, self).__init__()
        self.embedding_dim = embedding_dim
        self.protein_projection = nn.Linear(1536, embedding_dim)
        self.molecule_projection = nn.Linear(768, embedding_dim)

        transformer_layers = TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=embedding_dim * 4, dropout=dropout_rate, activation=F.gelu, batch_first = True)
        self.self_attention = TransformerEncoder(transformer_layers, num_attention_layers)
        
        self.affinity_head = nn.Sequential(
            nn.Linear(embedding_dim, linear_dim),
            nn.GELU(),
            nn.Linear(linear_dim, 1)
        )
        self.binding_site_head = nn.Sequential(
            nn.Linear(embedding_dim, linear_dim),
            nn.GELU(),
            nn.Linear(linear_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, protein_embedding, molecule_embedding):
        seq_length = protein_embedding.size(0)
        mol_length = molecule_embedding.size(0)

        # Feature extraction
        protein_embedding = self.protein_projection(protein_embedding)
        molecule_embedding = self.molecule_projection(molecule_embedding)

        # Weight embeddings
        protein_weight = mol_length / seq_length
        molecule_weight = seq_length / mol_length

        weighted_protein_embedding = protein_embedding * protein_weight
        weighted_molecule_embedding = molecule_embedding * molecule_weight

        # Concatenate weighted embeddings
        features = torch.cat((weighted_protein_embedding, weighted_molecule_embedding), dim=0)

        # Self-attention layers
        features = self.self_attention(features)

        # Affinity prediction
        affinity_output = self.affinity_head(features).squeeze().mean()
        print(affinity_output)

        # Drug target interaction prediction
        dti_output = self.dti_head(features).squeeze().mean()
        print(dti_output)

        # Binding site prediction
        binding_site_output = self.binding_site_head(features[:seq_length]).squeeze()
        print(binding_site_output)

        return affinity_output, dti_output, binding_site_output

class AffinityLMSmallModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(AffinityLMSmallModel, self).__init__()
        self.fc1 = nn.Linear(1536, 1536)
        self.fc2 = nn.Linear(768, 768)
        self.fc3 = nn.Linear(1536 + 768, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, prot_embedding, mol_embedding):
        x1 = self.dropout(self.gelu(self.fc1(prot_embedding)))
        x2 = self.dropout(self.gelu(self.fc2(mol_embedding)))
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(self.gelu(self.fc3(x)))
        x = self.gelu(self.fc4(x))
        x = self.dropout(self.gelu(self.fc5(x)))
        x = self.fc6(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class ProteinTokenizer:
    def __init__(self):
        self.vocab = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
            'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
            'U': 21, 'O': 21, # Unusual amino acids
            'X': 21, # Unknown or any amino acid
            'Z': 21, # Glutamic acid or glutamine
            'B': 21, # Asparagine or aspartic acid
            'J': 21, # Leucine or isoleucine
        }
        self.pad_token_id = 0

    def __call__(self, sequences, max_len=None, padding=True, truncation=True):
        tokenized = []
        for seq in sequences:
            tokens = [self.vocab.get(aa, self.vocab['X']) for aa in seq] # 'X' for unknown amino acids
            seq_len = len(tokens)

            if truncation and max_len is not None and seq_len > max_len:
                tokens = tokens[:max_len]
            if padding and max_len is not None:
                tokens += [self.pad_token_id] * (max_len - len(tokens))

            tokenized.append(tokens)

        return torch.tensor(tokenized)

# ------------------------------------------------
# Classes for Model Usage
# ------------------------------------------------

class AffinityLM:
    def __init__(self, model_path='data/AffinityLM3.pt', device=None, protein_cache_size=1000, molecule_cache_size=1000000):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load our affinity prediction model
        self.affinity_model = AffinityLMSmallModel()
        state_dict = torch.load(model_path, map_location=self.device)
        self.affinity_model.load_state_dict(state_dict)
        self.affinity_model.to(self.device)
        
        # Load the Ankh model for protein encoding
        self.ankh_model, self.ankh_tokenizer = ankh.load_large_model()
        self.ankh_model.eval()
        self.ankh_model.to(self.device)
        
        # Load the MolFormer model for molecule encoding
        self.molformer_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True).to(self.device)
        self.molformer_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        
        # Initialize caches for protein and molecule embeddings
        self.protein_cache = {}
        self.molecule_cache = {}
        self.protein_cache_size = protein_cache_size
        self.molecule_cache_size = molecule_cache_size
        
        # Load cached files
        self.load_cache()
    
    def load_cache(self):
        cache_folder = '.cache'
        if os.path.exists(cache_folder):
            protein_cache_file = os.path.join(cache_folder, 'protein_cache.pickle')
            if os.path.exists(protein_cache_file):
                with open(protein_cache_file, 'rb') as f:
                    self.protein_cache = pickle.load(f)
            
            molecule_cache_file = os.path.join(cache_folder, 'molecule_cache.pickle')
            if os.path.exists(molecule_cache_file):
                with open(molecule_cache_file, 'rb') as f:
                    self.molecule_cache = pickle.load(f)
    
    def save_cache(self):
        cache_folder = '.cache'
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        
        protein_cache_file = os.path.join(cache_folder, 'protein_cache.pickle')
        with open(protein_cache_file, 'wb') as f:
            pickle.dump(self.protein_cache, f)
        
        molecule_cache_file = os.path.join(cache_folder, 'molecule_cache.pickle')
        with open(molecule_cache_file, 'wb') as f:
            pickle.dump(self.molecule_cache, f)

    def encode_proteins(self, proteins, batch_size=2):
        embeddings = []
        with tqdm(total=len(proteins), desc="Encoding proteins", unit="protein") as pbar:
            for i in range(0, len(proteins), batch_size):
                batch = proteins[i:i+batch_size]
                cached_embeddings = [self.protein_cache[p].to(self.device) for p in batch if p in self.protein_cache]
                uncached_proteins = [p for p in batch if p not in self.protein_cache]
                if uncached_proteins:
                    tokens = self.ankh_tokenizer.batch_encode_plus(uncached_proteins, padding=True, return_tensors="pt")
                    with torch.no_grad():
                        output = self.ankh_model(input_ids=tokens['input_ids'].to(self.device), 
                                                 attention_mask=tokens['attention_mask'].to(self.device))
                        batch_embeddings = (output.last_hidden_state.mean(dim=1)+4.652981078834273e-05)/0.006146230538994151 # normalize
                        for p, e in zip(uncached_proteins, batch_embeddings):
                            if len(self.protein_cache) < self.protein_cache_size:
                                self.protein_cache[p] = e.cpu()
                    embeddings.extend(cached_embeddings + [e for e in batch_embeddings])
                else:
                    embeddings.extend(cached_embeddings)
                pbar.update(len(batch))
        return torch.stack(embeddings)
    
    def encode_molecules(self, molecules, batch_size=16):
        embeddings = []
        with tqdm(total=len(molecules), desc="Encoding molecules", unit="molecule") as pbar:
            for i in range(0, len(molecules), batch_size):
                batch = molecules[i:i+batch_size]
                cached_embeddings = [self.molecule_cache[m].to(self.device) for m in batch if m in self.molecule_cache]
                uncached_molecules = [m for m in batch if m not in self.molecule_cache]
                if uncached_molecules:
                    tokens = self.molformer_tokenizer.batch_encode_plus(uncached_molecules, padding=True, return_tensors="pt")
                    with torch.no_grad():
                        output = self.molformer_model(input_ids=tokens['input_ids'].to(self.device), 
                                                      attention_mask=tokens['attention_mask'].to(self.device))
                        batch_embeddings = (output.last_hidden_state.mean(dim=1)+0.0020459620282053947)/0.5608618246902549 # normalize
                        for m, e in zip(uncached_molecules, batch_embeddings):
                            if len(self.molecule_cache) < self.molecule_cache_size:
                                self.molecule_cache[m] = e.cpu()
                    embeddings.extend(cached_embeddings + [e for e in batch_embeddings])
                else:
                    embeddings.extend(cached_embeddings)
                pbar.update(len(batch))
        return torch.stack(embeddings)
    
    def __call__(self, proteins, molecules, batch_size=128):
        prot_embeddings = self.encode_proteins(proteins)
        mol_embeddings = self.encode_molecules(molecules)
        affinities = []
        for i in range(0, len(proteins), batch_size):
            with torch.no_grad():
                batch_affinities = self.affinity_model(prot_embeddings[i:i+batch_size], mol_embeddings[i:i+batch_size])
            affinities.extend(batch_affinities)
        return (torch.stack(affinities)*1.5614094578916633+6.51286529169358).cpu().flatten() # denormalize
    
    def score_molecules(self, protein, molecules, batch_size=128, prot_batch_size=2, mol_batch_size=16, save_cache=False):
        prot_embedding = self.encode_proteins([protein], batch_size=prot_batch_size)
        mol_embeddings = self.encode_molecules(molecules, batch_size=mol_batch_size)
        
        affinities = []
        total_molecules = len(molecules)
        
        print("Calculating affinities...")
        with tqdm(total=total_molecules, desc="Scoring molecules", unit="molecule") as pbar:
            for i in range(0, total_molecules, batch_size):
                with torch.no_grad():
                    batch = mol_embeddings[i:i+batch_size]
                    batch_size_actual = len(batch)
                    batch_affinities = self.affinity_model(prot_embedding.repeat(batch_size_actual, 1), batch)
                    affinities.extend(batch_affinities)
                pbar.update(batch_size_actual)
        
        affinities = (torch.stack(affinities)*1.5614094578916633+6.51286529169358).cpu().flatten()  # denormalize
        
        # Create a pandas DataFrame with molecule identifiers and their affinities
        data = {'SMILES': molecules, 'pKd': affinities.numpy()}
        df = pd.DataFrame(data)
        
        print("Sorting results...")
        # Sort the DataFrame by affinity in ascending order (highest affinity first)
        sorted_df = df.sort_values(by='pKd', ascending=False)
        sorted_df.reset_index(drop=True, inplace=True)
        
        if save_cache:
            print("Saving cache...")
            self.save_cache()
        
        return sorted_df