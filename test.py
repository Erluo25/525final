import torch
import torch.nn as nn

def pad_and_stack_vectors(vectors):
    max_len = max([len(vec) for vec in vectors])
    padded_vectors = torch.zeros(len(vectors), max_len)

    for i, vec in enumerate(vectors):
        padded_vectors[i, :len(vec)] = vec

    return padded_vectors

def preprocess_data(data):
    preprocessed_data = []
    for element in data:
        stacked_vectors = pad_and_stack_vectors(element)
        preprocessed_data.append(stacked_vectors)
    return preprocessed_data

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, max_seq_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)  # Add positional encoding
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dim_feedforward)
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, src, src_mask=None):
        embedded = self.embedding(src)
        embedded = embedded + self.pos_encoder(torch.arange(0, src.size(1)).unsqueeze(0).to(src.device))  # Add positional encoding
        transformer_out = self.transformer(embedded, src_mask=src_mask)
        output = self.output_layer(transformer_out)
        return output

input_dim = 8
d_model = 64
nhead = 4
num_layers = 3
dim_feedforward = 256
max_seq_len = 100  # Maximum sequence length for positional encoding

model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, max_seq_len)

data = [
    [torch.randn(3), torch.randn(8)],
    [torch.randn(8), torch.randn(8)],
    [torch.randn(3), torch.randn(3), torch.randn(8)],
   
]
print(data)

preprocessed_data = preprocess_data(data)
print(preprocess_data)

def generate_mask(seq_len, device):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Reshape and concatenate the tensors
single_input = torch.cat([seq.view(-1, input_dim) for seq in preprocessed_data], dim=0).unsqueeze(0)  # Shape: (1, K * max_seq_len, input_dim)
print(single_input.size())

src_mask = generate_mask(single_input.size(1), single_input.device)

outputs = model(single_input, src_mask=src_mask)
print(outputs.size())