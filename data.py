import pickle
import random

import nltk

PAD = 0
GO = 1
EOS = 2
UNK = 3

def load_dataset(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        word2id = data['word2id']
        id2word = data['id2word']
        dialogs = data['trainingSamples']
    # dialoges.shape: [n, 2]
    print('dialogs lenght is {}'.format(len(dialogs)))
    return word2id, id2word, dialogs[0:1000]


def generate_batches(dialogs, batch_size):
    random.shuffle(dialogs)
    batches = []
    
    n_samples = len(dialogs)

    for i in range(0, n_samples, batch_size):
        batch_dialogs = dialogs[i:min(i+batch_size, n_samples)]
        batch = create_single_batch(batch_dialogs)
        batches.append(batch)
    return batches

class Batch:
    def __init__(self):
        self.batch_size = 0
        self.encoder_sources = []
        self.encoder_sources_length_list = []
        self.decoder_targets = []
        self.decoder_targets_length_list = []

def create_single_batch(batch_dialogs): 
    batch = Batch()
    #  batch_dialogs = dialogs[i:min(i+batch_size, n_samples)]
    batch.batch_size = len(batch_dialogs)
    batch.encoder_sources_length_list = [len(dialog[0]) for dialog in batch_dialogs]
    batch.decoder_targets_length_list = [len(dialog[1]) for dialog in batch_dialogs]

    max_sources_length = max(batch.encoder_sources_length_list)
    max_targets_length = max(batch.decoder_targets_length_list)

    for dialog in batch_dialogs:
        # for source sequence: reverse the inputs
        source = list(reversed(dialog[0]))
        source_pad = [PAD] * (max_sources_length - len(dialog[0]))
        batch.encoder_sources.append(source_pad + source)

        # target sequence
        target = dialog[1]
        target_pad = [PAD] * (max_targets_length - len(dialog[1]))
        batch.decoder_targets.append(target + target_pad)

    return batch

## sentence2batch is used to convert sentence to a data batch
def query2batch(sentence, word2idx):
    tokens = nltk.word_tokenize(sentence) 
    query = []
    for token in tokens:
        # idx = word2idx.get(token, default=UNK)
        idx = word2idx.get(token, UNK) 
        query.append(idx)

    batch_dialogs = []
    batch_dialogs.append([query, []])

    ## batch_dialogs is depth = 3 list
    ## first depth: No. of dialogs, len(first_depth) = batch_size
    ## second depth: first element is query, second element is response, len(second_depth) = 2
    ## third depth: no of each word index in query or response

    return create_single_batch(batch_dialogs)
    



