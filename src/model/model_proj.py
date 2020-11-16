import torch
from segmentation_tools import *
from torch import nn
from transformers import BertForMaskedLM


class AffixPredictor(nn.Module):

    def __init__(self, mode):
        super(AffixPredictor, self).__init__()

        # Store affix type
        self.mode = mode

        # BERT layer
        self.bert_lm = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def forward(self, embs, masks, segs, idxes_mask):

        # Perform forward pass
        output = self.bert_lm(inputs_embeds=embs, attention_mask=masks, token_type_ids=segs)[0]

        # Extract predictions for mask tokens
        if self.mode == 'pfx' or self.mode == 'sfx':
            return output[torch.arange(output.size(0)), idxes_mask]
        elif self.mode == 'both':
            return output[torch.arange(output.size(0)).repeat(2, 1).T, idxes_mask]
