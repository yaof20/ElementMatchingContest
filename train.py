# coding = utf-8
import paddle.fluid.dygraph as D
import paddle.fluid as F
from ernie.optimization import AdamW, LinearDecay
from data_utils import pad_data, make_batches
from model import ErnieForElementClassification
from ernie.tokenizing_ernie import ErnieTokenizer
import logging


def train(model, dataset, lr=1e-5, batch_size=1, epochs=10):

    max_steps = epochs * (len(dataset) // batch_size)
    # max_train_steps = args.epoch * num_train_examples // args.batch_size  // dev_count
    optimizer = AdamW(
        # learning_rate=LinearDecay(lr, int(0), max_steps),
        learning_rate= lr,
        parameter_list=model.parameters(),
        weight_decay=0)

    model.train()
    logging.info('start training process!')
    for epoch in range(epochs):
        # shuffle the dataset every epoch by reloading it
        data_loader = make_batches(dataset, batch_size=batch_size, shuffle=True)

        running_loss = 0.0
        for i, data in enumerate(data_loader):
            # prepare inputs for the model
            src_ids, sent_ids, labels = data

            # convert numpy variables to paddle variables
            src_ids = D.to_variable(src_ids)
            sent_ids = D.to_variable(sent_ids)
            labels = D.to_variable(labels)

            # feed into the model
            outs = model(src_ids, sent_ids, labels=labels)

            loss = outs[0]
            loss.backward()
            optimizer.minimize(loss)
            model.clear_gradients()

            running_loss += loss.numpy()[0]
            if i % 10 == 9:
                print('epoch: ', epoch + 1, '\n\tstep: ', i + 1, '\n\trunning_loss: ', running_loss)
                running_loss = 0.0
        if epoch+1 == 1:
            state_dict = model.state_dict()
            F.save_dygraph(state_dict, './saved/resultall_4/model_1epoch')
        if epoch+1 == 2:
            state_dict = model.state_dict()
            F.save_dygraph(state_dict, './saved/resultall_4/model_2epoch')
        if epoch+1 == 3:
            state_dict = model.state_dict()
            F.save_dygraph(state_dict, './saved/resultall_4/model_3epoch')
        if epoch+1 == 4:
            state_dict = model.state_dict()
            F.save_dygraph(state_dict, './saved/resultall_4/model_4epoch')
        if epoch+1 == 5:
            state_dict = model.state_dict()
            F.save_dygraph(state_dict, './saved/resultall_4/model_5epoch')
        if epoch+1 == 6:
            state_dict = model.state_dict()
            F.save_dygraph(state_dict, './saved/resultall_4/model_6epoch')
        if epoch+1 == 7:
            state_dict = model.state_dict()
            F.save_dygraph(state_dict, './saved/resultall_4/model_7epoch')
        if epoch+1 == 8:
            state_dict = model.state_dict()
            F.save_dygraph(state_dict, './saved/resultall_4/model_8epoch')
        if epoch+1 == 9:
            state_dict = model.state_dict()
            F.save_dygraph(state_dict, './saved/resultall_4/model_9epoch')


    state_dict = model.state_dict()
    logging.info('start saving trained model parameters')

    F.save_dygraph(state_dict, './saved/resultall_4/model_10epoch')

    logging.info('model parameters saved!')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    place = F.CUDAPlace(0) if F.is_compiled_with_cuda() else F.CPUPlace()
    # use dynamic computation graph
    D.guard(place).__enter__()
    print('training model on: ', place)

    # load model and tokenizer
    ernie_model = ErnieForElementClassification.from_pretrained('./model-ernie1.0.1', num_labels=1)
    ernie_tokenizer = ErnieTokenizer.from_pretrained('./model-ernie1.0.1')
    logging.info('load model and tokenizer successfully!')

    # load dataset
    padded_dataset = pad_data('./data/train&dev_4.json', ernie_tokenizer, 512)

    logging.info('load dataset successfully!')

    train(ernie_model, padded_dataset, lr=1e-5, batch_size=8, epochs=10)



