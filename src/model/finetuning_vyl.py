import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import numpy as np
from data_helpers_vyl import *
from evaluation import *
import random
from model import AffixPredictor
from segmentation_tools import *
from torch import nn, optim, softmax
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def train(train_loader, val_loader, model, lr, n_epochs, cuda, lexicon):

    # Define device
    device = torch.device('cuda:{}'.format(cuda) if torch.cuda.is_available() else 'cpu')

    # Define optimizer and training objective
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Move model to CUDA
    model = model.to(device)

    print('Training model...')

    best_acc = None

    for epoch in range(1, n_epochs + 1):

        print('Starting epoch {}...'.format(epoch))

        model.train()

        for i, batch in enumerate(train_loader):

            if batch[0].size(-1) > 512:
                print('Skipping batch {}...'.format(i))
                continue

            if i % 1000 == 0:
                print('Processed {} batches...'.format(i))

            optimizer.zero_grad()

            sents, masks, segs, idxes_mask, labels = batch

            sents, masks, segs, idxes_mask = sents.to(device), masks.to(device), segs.to(device), idxes_mask.to(device)

            # Forward pass
            try:
                output = model(sents, masks, segs, idxes_mask)
            except RuntimeError:
                print('Skipping batch {}...'.format(i))
                continue

            labels = labels.to(device)

            loss = criterion(output, labels)

            # Backpropagate loss
            loss.backward()

            # Update weights
            optimizer.step()

            # Delete tensors to free memory
            del sents, masks, segs, idxes_mask, labels, output

        acc = test_single(val_loader, model, cuda)

        print('Accuracy after epoch {}:\t{:.4f}'.format(epoch, acc))

        with open('results_vyl/results_bert_vyl_{}.txt'.format(lexicon), 'a+') as f:
            f.write('{:.0e}\t{}\n'.format(lr, acc))

        if best_acc is None or acc > best_acc:

            best_acc = acc

            torch.save(model.state_dict(), 'trained_vyl/model_bert_vyl_{}_{:.0e}.torch'.format(lexicon, lr))

    with open('results_vyl/results_bert_vyl_{}.txt'.format(lexicon), 'a+') as f:
        f.write('\n')


def test_single(test_loader, model, cuda):

    tok = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define device
    device = torch.device('cuda:{}'.format(cuda) if torch.cuda.is_available() else 'cpu')

    # Initialize affix list
    afxes = []
    with open(str(Path('../../data/external/affixes_vyl.txt').resolve()), 'r') as f:
        for l in f:
            # Exclude affixes not in BERT vocabulary
            if l.strip() == '' or l.strip() in {'NULL', 'orium', 'tude'}:
                continue
            afxes.append(l.strip().lower())

    # Add affixes not in BERT as unused tokens
    afxes = ['[unused96]', '[unused97]', '[unused98]'] + ['##' + afx for afx in afxes]
    idxes_afx, _ = torch.sort(torch.tensor(tok.convert_tokens_to_ids(afxes)).to(device))

    print('Evaluating model...')

    y_true = list()
    y_pred = list()

    model.eval()

    with torch.no_grad():

        for batch in test_loader:

            sents, masks, segs, idxes_mask, labels = batch

            sents, masks, segs, idxes_mask = sents.to(device), masks.to(device), segs.to(device), idxes_mask.to(device)

            # Forward pass
            output = model(sents, masks, segs, idxes_mask)

            # Filter affixes
            output_afx = torch.index_select(output, -1, idxes_afx)

            # Rank predictions
            vals_afx, preds_afx = torch.topk(output_afx, k=output_afx.size(-1), dim=-1)

            labels = labels.to(device)

            # Store labels and predictions
            y_true.extend([l.item() for l in labels])
            y_pred.extend([[idxes_afx[p].item() for p in list(l)][0] for l in preds_afx])

            # Delete tensors to free memory
            del sents, masks, segs, idxes_mask, labels, output

    return len([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)


def main():

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    parser.add_argument('--lexicon', default=None, type=str, required=True, help='Lexicon setting')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--cuda', default=None, type=int, required=True, help='Selected CUDA.')

    args = parser.parse_args()

    print('Lexicon setting: {}'.format(args.lexicon))
    print('Batch size: {}'.format(args.batch_size))
    print('Learning rate: {}'.format(args.lr))
    print('Number of epochs: {}'.format(args.n_epochs))

    # Define poath to data
    inpath = str(Path('../../data/final').resolve())

    if args.lexicon == 'shared':
        train_path = '{}{}sents_vyl_train.txt'.format(inpath, os.sep)
        val_path = '{}{}sents_vyl_dev.txt'.format(inpath, os.sep)

    elif args.lexicon == 'split':
        train_path = '{}{}sents_vyl_train_split.txt'.format(inpath, os.sep)
        val_path = '{}{}sents_vyl_dev_split.txt'.format(inpath, os.sep)

    # Initialize train loader
    print('Load training data...')
    train_data = AffixDataset(train_path)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sents)

    # Initialize val loader
    print('Load validation data...')
    val_data = AffixDataset(val_path)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_sents)

    # Initialize model
    affix_predictor = AffixPredictor('sfx', freeze=False)

    train(train_loader, val_loader, affix_predictor, args.lr, args.n_epochs, args.cuda, args.lexicon)


if __name__ == '__main__':
    main()
