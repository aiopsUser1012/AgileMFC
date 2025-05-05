import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict
import math
import random
import warnings
warnings.filterwarnings("ignore")

class CustomDataset(Dataset):
    def __init__(self, data_path, shuffle=False):
        self.data = []
        self.labels = []
        self.label_to_index = {}
        self.word_to_index = defaultdict(lambda: len(self.word_to_index))
        
        with open(data_path, 'r') as f:
            for line in f:
                parts = line.split('\t')
                words = parts[0].split()
                indexed_words = [self.word_to_index[word] for word in words]
                self.data.append(torch.tensor(indexed_words, dtype=torch.int64))

                label_str = parts[1]
                if label_str not in self.label_to_index:
                    self.label_to_index[label_str] = len(self.label_to_index)

                self.labels.append(self.label_to_index[label_str])

        if shuffle:
            self.shuffle()

    def shuffle(self):
        data_labels = [(x, y) for x, y in zip(self.data, self.labels)]
        random.shuffle(data_labels)
        self.data = [i[0] for i in data_labels]
        self.labels = [i[1] for i in data_labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32), 
            self.labels[idx]
        )
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(self.device)
        return self.dropout(X)

class EncoderBlock(nn.Module):
    def __init__(self, input_size, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(input_size, num_heads=8)
        self.dropout = nn.Dropout(0.2)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)

        self.ffn = nn.Sequential(OrderedDict([
                ('hidden_layer',nn.Linear(input_size, forward_expansion * input_size)),
                ('activation',nn.ReLU()),
                ('dropout',nn.Dropout(0.1)),
                ('output_layer',nn.Linear(forward_expansion * input_size, input_size))
        ]))

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)

        x = self.norm1(x + self.dropout(attn_output))
        ffn_outputs = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_outputs[0]))

        return x
    
class Expert(nn.Module):
    def __init__(self, input_size, forward_expansion, num_layers):
        super(Expert, self).__init__()
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(input_size, forward_expansion))
        self.attention_weights = [None] * num_layers

    def forward(self, x):
        for index, blk in enumerate(self.blks):
            x = blk(x)
            self.attention_weights[index] = blk.self_attention.in_proj_weight
        
        out = x.mean(dim=1)
        out = torch.unsqueeze(out, 0)
        return out
    
class Tower(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class LinearFusion(nn.Module):
    def __init__(self, input_dim):
        super(LinearFusion, self).__init__()
        self.fuse = nn.Linear(input_dim * 3, input_dim)
    
    def forward(self, metric_outputs, log_outputs, trace_outputs):
        concatenated = torch.cat((metric_outputs, log_outputs, trace_outputs), dim=2)
        output = self.fuse(concatenated)
        return output

class MMoEv2(nn.Module):
    def __init__(self, config):
        super(MMoEv2, self).__init__()
        self.config = config
        self.input_dim = config["vector_dim"]
        self.num_experts = config["num_experts"]
        self.num_tasks = config["num_tasks"]
        self.expert_out_dim = self.input_dim
        self.towers_hidden_dim = config["towers_hidden_dim"]
        self.softmax = nn.Softmax(dim=1)
        self.emb_dim = config['emb_dim']
        self.forward_expansion = 4
        self.embedding_layer = nn.Embedding(self.emb_dim, self.input_dim)
        self.pos_encoding = PositionalEncoding(self.input_dim, 0.5)
        self.num_layers = config['num_layers'] # transformer block layers
        self.gate_weight = self.config['gate_weight']

        self.experts = nn.ModuleList(
            [Expert(self.input_dim, self.forward_expansion, self.num_layers)
             for _ in range(self.num_experts)])
        
        self.metric_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(self.input_dim, self.num_experts), requires_grad=True)
             for _ in range(self.num_tasks)])
        
        self.log_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(self.input_dim, self.num_experts), requires_grad=True)
             for _ in range(self.num_tasks)])

        self.trace_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(self.input_dim, self.num_experts), requires_grad=True)
             for _ in range(self.num_tasks)])

        self.share_gate = nn.Parameter(torch.randn(self.input_dim, self.num_experts), requires_grad=True)

        self.towers = nn.ModuleList(
            [Tower(self.expert_out_dim, self.towers_hidden_dim, 1)
             for _ in range(self.num_tasks)])
        
        self.fuse = LinearFusion(self.input_dim)

    def forward(self, metric_inputs, log_inputs, trace_inputs):
        metric_inputs = self.embedding_layer(metric_inputs)
        log_inputs = self.embedding_layer(log_inputs)
        trace_inputs = self.embedding_layer(trace_inputs)

        metric_outputs = self.modality_process(metric_inputs, self.metric_gates)
        log_outputs = self.modality_process(log_inputs, self.log_gates)
        trace_outputs = self.modality_process(trace_inputs, self.trace_gates)

        out = self.fuse(metric_outputs, log_outputs, trace_outputs)
        out = out.squeeze(0)

        return out

    def modality_process(self, x, modality_gate):
        x = self.pos_encoding(x * math.sqrt(self.input_dim))

        expert_out = torch.stack([expert(x) for expert in self.experts])
        expert_out = torch.squeeze(expert_out, 1)

        gate_out = [self.softmax((x[:,:,i] @ modality_gate[i]) * (1 - self.gate_weight) +
                                 (x[:,:,i] @ self.share_gate) * self.gate_weight)
                                 for i in range(self.num_tasks)]

        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.expert_out_dim) * expert_out
                       for g in gate_out]

        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        modality_out = [t(ti) for t, ti in zip(self.towers, tower_input)]
        modality_out = torch.stack(modality_out, dim=0).permute(2, 1, 0)

        return modality_out

class LinearClassifierv2(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, n_classes):
        super(LinearClassifierv2, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, n_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.softmax(out)
        return out
    