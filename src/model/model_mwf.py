import torch
from segmentation_tools import *
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class MWFPredictor(nn.Module):

    def __init__(self, freeze):
        super(MWFPredictor, self).__init__()

        # BERT layer
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze BERT layer
        for param in self.bert.parameters():
            if freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Linear layer for classification
        self.linear = nn.Linear(768, 1)

        for param in self.linear.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(0.2)

    def forward(self, sents, masks, segs, idxes, device):

        # Perform forward pass
        output = self.dropout(self.bert(sents, attention_mask=masks, token_type_ids=segs)[0])

        # Initialize empty tensor
        output_max = torch.zeros(output.size(0), output.size(-1)).to(device)

        for i, (s, idx) in enumerate(zip(output, idxes)):

            # Pick out hidden states corresponding to word
            w = s[idx[0]:idx[1]]

            # Perform max pooling operation
            output_max[i] = torch.max(w, dim=0)[0]

        return F.sigmoid(self.linear(output_max))
