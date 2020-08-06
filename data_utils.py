import json
import numpy as np


def pad_data(file_name, tokenizer, max_len):
    """
        This function is used as the Dataset Class in PyTorch
    """
    file_content = json.load(open(file_name, encoding='utf-8'))
    data = []
    for line in file_content:
        person, element = line['被告人'], line['element']
        sentence = line['sentence']
        label = line['label']

        person += element   # concatenate '#person#' and 'sentence' as the input to ernie
        src_id, sent_id = tokenizer.encode(person, sentence, truncate_to=max_len-3)  # 3 special tokens

        # pad src_id and sent_id (with 0 and 1 respectively)
        src_id = np.pad(src_id, [0, max_len-len(src_id)], 'constant', constant_values=0)
        sent_id = np.pad(sent_id, [0, max_len-len(sent_id)], 'constant', constant_values=1)

        data.append((src_id, sent_id, label))
    return data


def make_batches(data, batch_size, shuffle=True):
    """
        This function is used as the DataLoader Class in PyTorch
    """
    if shuffle:
        np.random.shuffle(data)
    loader = []
    for j in range(len(data)//batch_size):
        one_batch_data = data[j * batch_size:(j + 1) * batch_size]
        src_id, sent_id, label = zip(*one_batch_data)

        src_id = np.stack(src_id)
        sent_id = np.stack(sent_id)
        label = np.stack(label).astype(np.float32)  # change the data type to compute BCELoss conveniently

        loader.append((src_id, sent_id, label))
    return loader
