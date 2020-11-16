import torch
from torch import nn
from torch.nn import functional as F


class AffixPredictor(nn.Module):

    # Pass hyperparameters
    def __init__(self, embedding_matrix_w, input_dim_c, embedding_dim_c, hidden_dim_w, hidden_dim_c, output_dim,
                 dropout):
        super(AffixPredictor, self).__init__()

        # Embedding layers for left and right contexts
        self.embedding_s_1 = nn.Embedding(embedding_matrix_w.shape[0], embedding_matrix_w.shape[1], padding_idx=0)
        self.embedding_s_2 = nn.Embedding(embedding_matrix_w.shape[0], embedding_matrix_w.shape[1], padding_idx=0)

        # Initialize with pretrained embeddings and freeze weights
        self.embedding_s_1.weight.data.copy_(torch.from_numpy(embedding_matrix_w))
        self.embedding_s_1.weight.requires_grad = False
        self.embedding_s_2.weight.data.copy_(torch.from_numpy(embedding_matrix_w))
        self.embedding_s_2.weight.requires_grad = False

        # Embedding layer for base
        self.embedding_b = nn.Embedding(input_dim_c, embedding_dim_c, padding_idx=0)

        # LSTM layers for left and right contexts
        self.lstm_s_1 = nn.LSTM(embedding_matrix_w.shape[1], hidden_dim_w, batch_first=True, bidirectional=True, num_layers=3)
        self.lstm_s_2 = nn.LSTM(embedding_matrix_w.shape[1], hidden_dim_w, batch_first=True, bidirectional=True, num_layers=3)

        # LSTM layer for base
        self.lstm_b = nn.LSTM(embedding_dim_c, hidden_dim_c, batch_first=True, bidirectional=True, num_layers=3)

        # Affine layer 1
        self.linear_1 = nn.Linear(4 * hidden_dim_w + 2 * hidden_dim_c, int((4 * hidden_dim_w + 2 * hidden_dim_c) / 4))

        # Affine layer 2
        self.linear_2 = nn.Linear(int((4 * hidden_dim_w + 2 * hidden_dim_c) / 4), output_dim)

        # Define dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, s_1, s_2, b):
        e_s_1 = self.dropout(self.embedding_s_1(s_1))
        o_s_1, (h_s_1, c_s_1) = self.lstm_s_1(e_s_1)

        e_s_2 = self.dropout(self.embedding_s_2(s_2))
        o_s_2, (h_s_2, c_s_2) = self.lstm_s_2(e_s_2)

        e_b = self.dropout(self.embedding_b(b))
        o_b, (h_b, c_b) = self.lstm_b(e_b)

        h = torch.cat((o_s_1[:, -1, :], o_s_2[:, -1, :], o_b[:, -1, :]), dim=-1)

        t = self.dropout(F.relu(self.linear_1(h)))

        return self.linear_2(t)

