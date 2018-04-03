import tarfile, io
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import autograd, nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from korean_character_parser import decompose_str_as_one_hot
from dataset import load_batch_input_to_memory, get_dataloaders, read_test_file

import nsml
from nsml import DATASET_PATH, IS_DATASET, GPU_NUM


def bind_model(model, wv_model):
    def save(filename, *args):
        # save the model with 'checkpoint' dictionary.
        checkpoint = {
            'model': model.state_dict()
        }
        torch_file = io.BytesIO()
        torch.save(checkpoint, torch_file)
        torch_info = tarfile.TarInfo("torch")
        torch_info.size = len(torch_file.getbuffer())

        wv_file = io.BytesIO()
        wv_model.save_to(wv_file)
        wv_info = tarfile.TarInfo("word2vec")
        wv_info.size = len(wv_file.getbuffer())

        tarball = tarfile.open(filename, 'w')
        tarball.add(wv_info, wv_file)
        tarball.add(torch_info, torch_file)
        tarball.close()

    def load(filename, *args):
        tarball = tarfile.open(filename, 'r')
        for mem in tarball.getmembers():
            file = extractfile(mem)
            if mem.name == 'word2vec':
                wv_model.load_from(file)
                print('Word2Vec Model loaded')

            elif mem.name == 'torch':
                checkpoint = torch.load(file)
                model.load_state_dict(checkpoint['model'])
                print('PyTorch Model loaded')


    def infer(raw_data, **kwargs):
        data = raw_data['data']
        data = wv_model.preprocess_train(data)
        
        model.eval()
        output_predictions = model(data)
        output_predictions = output_predictions.squeeze()
        prob = output_predictions.data
        prediction = np.where(prob > 0.5, 1, 0)
        return list(zip(prob, prediction.tolist()))

    # function in function is just used to divide the namespace.
    nsml.bind(save, load, infer)


def data_loader(dataset_path, train=False, batch_size=200,
                ratio_of_validation=0.1, shuffle=True):
    if train:
        return get_dataloaders(dataset_path=dataset_path, batch_size=batch_size,
                               ratio_of_validation=ratio_of_validation,
                               shuffle=shuffle)
    else:
        data_dict = {'data': read_test_file(dataset_path=DATASET_PATH)}
        return data_dict


class LSTMRegression(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, character_size, output_dim,
                 minibatch_size):
        super(LSTMRegression, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.character_size = character_size
        self.output_dim = output_dim
        self.minibatch_size = minibatch_size
        # this embedding is a table to handle sparse matrix instead of one-hot coding. so we just feed a list of indexes.
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers)
        # non-linear function is defined later.
        self.hidden2score = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self):
        initializer_1 = autograd.Variable(
            torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim))
        initializer_2 = autograd.Variable(
            torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim))
        if GPU_NUM:
            initializer_1 = initializer_1.cuda()
            initializer_2 = initializer_2.cuda()
        return initializer_1, initializer_2

    def forward(self, data):  # define inter-layer operations
        # re-pruning. due to dataparallel.

        # correct data format of input. list of list
        # [
        # ['미세먼지를 마시면\t미세먼지는..', 0],
        # ['버스안내방송\t버스안내방송 질문!', 1],
        # ['고려의 왕 이름\t고려왕들의 이름', 1],
        # ['세상에서 제일~!\t세상에서 제일', 1],
        # ['지표생물?\t지표생물의 뜻...', 1],
        # ['모글이 뭔가요\t당이뭔가요???', 0],
        # ['잠바에 묻은 얼룩\t잠바 얼룩', 1],
        # ['크로와상\t14co2', 0] ...
        # ]

        preprocessed = [decompose_str_as_one_hot(datum[0], warning=False) for datum in data]
        preprocessed.sort(key=lambda x: len(x), reverse=True)

        var_seqs, var_lengths = load_batch_input_to_memory(preprocessed, has_targets=False)

        var_seqs = autograd.Variable(var_seqs)
        var_lengths = autograd.Variable(var_lengths)
        if GPU_NUM:
            var_seqs = var_seqs.cuda(async=True)
            var_lengths = var_lengths.cuda(async=True)

        var_seqs = var_seqs[:, :var_lengths.data.max()]

        self.minibatch_size = len(var_lengths)

        # Zero padded maxtrix shaped (Batch, Time) ->  Tensor shaped (Batch, Time, Embeded_Feature)
        embeds = self.embeddings(var_seqs)

        # (Batch X Compact_Time , Embeded_Feature)
        packed_x = pack_padded_sequence(embeds, var_lengths.data.cpu().numpy(), batch_first=True)
        # Compact_time means a tensor without pads. So this is a concatenated tensor with only useful sequence.
        # Ex) [[53, 16], [40,16]] --> [53+40, 16]

        # This makes the memory the parameters and its grads occupied contiguous for efficiency of memory usage..
        self.lstm.flatten_parameters()

        # _hidden is not important, the output is important.
        packed_output, _hidden = self.lstm(packed_x, self.init_hidden())

        # Reverse operation of pack_padded_sequence. as (Time, Batch, Concatenation of 2 directional hidden's output).
        lstm_outs, _ = pad_packed_sequence(packed_output)

        # Implementation of last relevant output indexing.
        if GPU_NUM:
            idx = ((var_lengths - 1).view(-1, 1).expand(lstm_outs.size(1),
                                                        lstm_outs.size(2)).unsqueeze(0)).cuda()  # async=True
        else:
            idx = ((var_lengths - 1).view(-1, 1).expand(lstm_outs.size(1),
                                                        lstm_outs.size(2)).unsqueeze(0))
        # squeeze remove all ones, so it breaks when batch size is 1. dim=0 should be added to avoid it
        last_lstm_outs = lstm_outs.gather(0, idx).squeeze(dim=0)
        output_activation = self.hidden2score(last_lstm_outs)

        output_pred = F.sigmoid(output_activation)
        return output_pred


def inference_loop(data_loader, model, loss_function, optimizer, threshold, learning=True):  # , without_training=False
    if learning:
        model.train()  # select train mode
    else:
        model.eval()

    sum_loss = 0.0
    num_of_instances = 0
    acc_sum = 0.0
    for i, (data, label) in enumerate(data_loader):
        # we need to clear out the hidden state of the LSTM, detaching it from its history on the last instance.
        # Tensors not supported in DataParallel. You should put Variable to use data_parallel before call forward().
        if GPU_NUM:
            var_targets = autograd.Variable(label.float(), requires_grad=False).cuda(async=True)
        else:
            var_targets = autograd.Variable(label.float(), requires_grad=False)
        output_predictions = model(data)
        output_predictions = output_predictions.squeeze()

        loss = loss_function(output_predictions, var_targets)
        sum_loss += loss.data[0] * len(label)
        num_of_instances += len(label)
        acc = accuracy_score(label, np.where(output_predictions.data > threshold, 1, 0))
        acc_sum += acc * len(label)

        if learning:
            # Remember that Pytorch accumulates gradients. So we need to clear them out before each instance
            # model.zero_grad(). this is also same if optimizer = optim.Optimizer(model.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Batch : ', i + 1, '/', len(data_loader), ', BCE in this minibatch: ', loss.data[0])

    return sum_loss / num_of_instances, acc_sum / num_of_instances  # return mean loss


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--optimizer', type=str, default='adam')  # optimizer
    args.add_argument('--epochs', type=int, default=100)  # 100
    args.add_argument('--batch', type=int, default=200)  # 200
    args.add_argument('--embedding', type=int, default=8)  # 8
    args.add_argument('--hidden', type=int, default=512)  # 512
    args.add_argument('--threshold', type=float, default=0.5)  # 0.5
    args.add_argument('--layers', type=int, default=2)  # 2
    args.add_argument('--initial_lr', type=float, default=0.01)  # default : 0.01 (initial learning rate)
    args.add_argument('--char', type=int, default=250)  # Do not change this
    args.add_argument('--output', type=int, default=1)  # Do not change this
    args.add_argument('--mode', type=str, default='train')  # 'train' or 'test' (for nsml)
    args.add_argument('--pause', type=int, default=0)  # Do not change this (for nsml)
    args.add_argument('--iteration', type=str, default='0')  # Do not change this (for nsml)

    initial_time = time.time()
    config = args.parse_args()
    random_seed = 1234
    np.random.seed(random_seed)
    if GPU_NUM:
        torch.cuda.manual_seed(random_seed)

    model = LSTMRegression(config.embedding, config.hidden, config.layers,
                            config.char, config.output, config.batch)
    loss_function = nn.BCELoss()
    if GPU_NUM:
        model = model.cuda()
        loss_function = loss_function.cuda()
    threshold = config.threshold

    if config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.initial_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)

    bind_model(model)
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        if IS_DATASET:
            train_loader, val_loader = data_loader(dataset_path=DATASET_PATH, train=True,
                                                   batch_size=config.batch, ratio_of_validation=0.1,
                                                   shuffle=True)
        else:
            data_path = '../dummy/kin_data/'  # NOTE: load from local PC
            train_loader, val_loader = data_loader(dataset_path=data_path, train=True,
                                                   batch_size=config.batch, ratio_of_validation=0.1,
                                                   shuffle=True)

        min_val_loss = np.inf
        for epoch in range(config.epochs):
            # train on train set
            train_loss, train_acc = inference_loop(train_loader, model, loss_function, optimizer, threshold,
                                                   learning=True)
            # evaluate on validation set
            val_loss, val_acc = inference_loop(val_loader, model, loss_function,
                                               None, threshold, learning=False)
            print('epoch:', epoch, ' train_loss:', train_loss, 'train_acc:', train_acc,
                  ' val_loss:', val_loss, ' min_val_loss:', min_val_loss,
                  'val_acc:', val_acc)

            nsml.report(summary=True, scope=locals(), epoch=epoch,
                        total_epoch=config.epochs, val_acc=val_acc,
                        train_acc=train_acc, train__loss=train_loss,
                        val__loss=val_loss, min_val_loss=min_val_loss,
                        step=epoch)

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                nsml.save(epoch)
            else:  # default save
                if epoch % 30 == 0:
                    nsml.save(epoch)
