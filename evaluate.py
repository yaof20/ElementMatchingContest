import paddle.fluid.dygraph as D
import paddle.fluid as F
import numpy as np
import json
from data_utils import pad_data, make_batches
from model import ErnieForElementClassification
from ernie.tokenizing_ernie import ErnieTokenizer
import logging


def evaluate(model, model_name, dataset, batch_size=1, threshold=0.5):
    # load trained parameters
    state_dict, _ = F.load_dygraph(model_name)
    model.load_dict(state_dict)

    model.eval()
    logging.info('start evaluating process!')

    # load the data into data_loader
    eval_loader = make_batches(dataset, batch_size=batch_size, shuffle=False)

    all_pred = np.array([])
    all_label = np.array([])
    for i, data in enumerate(eval_loader):
        # prepare inputs for the model
        src_ids, sent_ids, labels = data

        # convert numpy variables to paddle variables
        src_ids = D.to_variable(src_ids)
        sent_ids = D.to_variable(sent_ids)

        # feed into the model
        outs = ernie_model(src_ids, sent_ids)

        # make prediction
        pred = outs.numpy() > threshold

        # add to the final result list
        all_pred = np.append(all_pred, pred)
        all_label = np.append(all_label, labels)

    all_pred = all_pred.astype(bool)
    all_label = all_label.astype(bool)

    tp = (all_pred & all_label).tolist().count(True)
    tp_plus_fp = all_pred.tolist().count(True)
    tp_plus_fn = all_label.tolist().count(True)

    p = (tp / tp_plus_fp) if tp_plus_fp != 0 else 0   # precision
    r = (tp / tp_plus_fn) if tp_plus_fn != 0 else 0   # recall
    f = (2 * p * r / (p + r)) if (p + r) != 0 else 0  # f1
    return p, r, f


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    place = F.CUDAPlace(0) if F.is_compiled_with_cuda() else F.CPUPlace()
    # use dynamic computation graph
    D.guard(place).__enter__()
    print('evaluating model on: ', place)

    # load raw model
    config = json.load(open('origin_config.json'))
    ernie_model = ErnieForElementClassification(config)
    logging.info('load raw model successfully!')

    # load tokenizer
    ernie_tokenizer = ErnieTokenizer.from_pretrained('./model-ernie1.0.1')
    logging.info('load tokenizer successfully!')

    # load dataset
    padded_dataset = pad_data('./data/data4/dev.json', ernie_tokenizer, 512)

    logging.info('load dataset successfully!')
    best_f1 = 0
    model_names = ['model_' + str(i) + 'epoch' for i in range(1, 11)]
    for model_name in model_names:
        for i in range(50, 56, 5):
            precision, recall, f1 = evaluate(ernie_model, './saved/resultall_4/'+model_name, padded_dataset, batch_size=8, threshold=i * 0.01)
            if f1 > best_f1:
                best_f1 = f1
            print('f1: %.5f, precision: %.5f, recall: %.5f,\nmodel_name: %s, bert_f1:, %.5f, threshold:, %.2f\n' % (f1, precision, recall, model_name, best_f1, i*0.01))
            # print(' current best_f1: ', best_f1, ' model_name: ', model_name, 'threshold', i*0.01)
            # print(' precision: ', precision, ' recall: ', recall, ' f1 score: ', f1, '\n')




