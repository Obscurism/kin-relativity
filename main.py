import argparse
import io
import nsml
import numpy as np
import os
import torch
import tarfile
import time

from dataset import load_batch_input_to_memory, get_dataloaders, read_test_file, word2vec_mapped_size
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM
from preprocess import Preprocessor
from progress import create_progressbar, finish_progressbar
from sklearn.metrics import accuracy_score

from torch import cat, nn, optim, Tensor
from torch.autograd import Variable
from torch.nn.functional import relu, sigmoid
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# May the table be with you

# ===SHAPE TABLE===
# Vocab := array(50,)
# Sentence := [Vocab...]
# SentenceSequence := [Sentence...]
# Input := [SentenceSequence, SentenceSequence]
# InputSet := [Input...]
# Ouput := int
# Data := [Input, Output]
# Dataset := [Data...]

def bind_model(comp_model, config):
    model, wv_model = comp_model

    def save(filename, *args):
        # save the model with 'checkpoint' dictionary.
        print("[Save] Saving...")
        tarball = tarfile.open(filename, 'w')

        print("[Save] Saving torch file...")
        checkpoint = {
            'model': model.state_dict()
        }
        torch_file = open('torch_model.tmp', 'wb+')
        torch.save(checkpoint, torch_file)
        torch_file.seek(0)
        torch_info = tarball.gettarinfo(fileobj=torch_file)

        print("[Save] Saving word2vec file...")
        wv_file = open('wv_model.tmp', 'wb+')
        wv_model.save_to(wv_file)
        wv_file.seek(0)
        wv_info = tarball.gettarinfo(fileobj=wv_file)

        print("[Save] Compressing files into tar file...")
        tarball.addfile(wv_info, wv_file)
        tarball.addfile(torch_info, torch_file)
        tarball.close()

        torch_file.close()
        wv_file.close()

        print("[Save] Removing temporary files...")
        os.remove('torch_model.tmp')
        os.remove('wv_model.tmp')

        print("[Save] Saving complete!")

    def load(filename, *args):
        print("[Load] Loading...")
        tarball = tarfile.open(filename, 'r')
        for mem in tarball.getmembers():
            tarball.extract(mem)

            if mem.name == 'wv_model.tmp':
                wv_model.load_from(mem.name)
                print('[Load] Word2Vec Model loaded.')

            elif mem.name == 'torch_model.tmp':
                extr_file = open(mem.name, 'rb')
                checkpoint = torch.load(extr_file)
                model.load_state_dict(checkpoint['model'])
                extr_file.close()

                print('[Load] PyTorch Model loaded.')

            os.remove(mem.name)


    def infer(raw_data, **kwargs):
        model.eval()

        data = raw_data
        data = wv_model.preprocess_test_all(data)
        # data: Dataset

        output_predictions = model(data)
        output_predictions = output_predictions.squeeze()
        prob = output_predictions.data

        output_data = [[prob, 1 if prob > 0.5 else 0] for prob in output_predictions.data]
        return output_data

    nsml.bind(save, load, infer)

    return save, load, infer


class LSTMRegression(nn.Module):
    def __init__(
        self, hidden_dim, hidden_dense, num_layers,
        output_dim, minibatch_size, drop_prob
    ):
        super(LSTMRegression, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dense = hidden_dense
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.minibatch_size = minibatch_size
        self.drop_prob = drop_prob

        self.lstm0_l0 = nn.LSTM(word2vec_mapped_size, self.hidden_dim, self.num_layers, dropout=self.drop_prob)
        self.lstm1_l0 = nn.LSTM(word2vec_mapped_size, self.hidden_dim, self.num_layers, dropout=self.drop_prob)
        self.dense_l1 = nn.Linear(2 * self.hidden_dim, self.hidden_dense)
        self.dense_l2 = nn.Linear(self.hidden_dense, self.output_dim)

    def init_hidden(self):
        initializer_1 = Variable(
            torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim))
        initializer_2 = Variable(
            torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim))
        if GPU_NUM:
            initializer_1 = initializer_1.cuda()
            initializer_2 = initializer_2.cuda()
        return initializer_1, initializer_2

    def forward(self, data):
        # data: Dataset
        # datum: Data
        # preprocessed: InputSet
        preprocessed = [datum[0] for datum in data]
        preprocessed.sort(key=lambda x: len(x[0]), reverse=True)

        # var_seqs: InputSet
        # var_lengths: array(None,)
        var_seqs, var_lengths = load_batch_input_to_memory(preprocessed, has_targets=False)

        var_seqs = Variable(var_seqs)
        var_lengths = Variable(var_lengths)

        if GPU_NUM:
            var_seqs = var_seqs.cuda(async=True)
            var_lengths = var_lengths.cuda(async=True)

        self.minibatch_size = len(var_lengths)

        var_seqs_x0 = var_seqs[:, 0]
        var_seqs_x1 = var_seqs[:, 1]

        packed_x0 = pack_padded_sequence(var_seqs_x0, var_lengths.data.cpu().numpy(), batch_first=True)
        packed_x1 = pack_padded_sequence(var_seqs_x1, var_lengths.data.cpu().numpy(), batch_first=True)

        # This makes the memory the parameters and its grads occupied contiguous for efficiency of memory usage..
        self.lstm0_l0.flatten_parameters()
        self.lstm1_l0.flatten_parameters()

        x0_out, _hidden0 = self.lstm0_l0(packed_x0, self.init_hidden())
        x1_out, _hidden1 = self.lstm1_l0(packed_x1, self.init_hidden())

        # Reverse operation of pack_padded_sequence. as (Time, Batch, Concatenation of 2 directional hidden's output).
        x0_out, _ = pad_packed_sequence(x0_out)
        x1_out, _ = pad_packed_sequence(x1_out)

        # Implementation of last relevant output indexing.
        if GPU_NUM:
            idx = ((var_lengths - 1).view(-1, 1).expand(x0_out.size(1),
                                                        x0_out.size(2)).unsqueeze(0)).cuda()  # async=True
        else:
            idx = ((var_lengths - 1).view(-1, 1).expand(x0_out.size(1),
                                                        x0_out.size(2)).unsqueeze(0))

        # squeeze remove all ones, so it breaks when batch size is 1. dim=0 should be added to avoid it
        x0_out = x0_out.gather(0, idx).squeeze(dim=0)
        x1_out = x1_out.gather(0, idx).squeeze(dim=0)

        # CAT IS SO CUTE, unless the cat is 'concat'. Nyan Nyan Nyan
        lstm_out = cat((x0_out, x1_out), 1)

        dense_out = relu(self.dense_l1(lstm_out))
        output_pred = sigmoid(self.dense_l2(dense_out))

        return output_pred


def inference_loop(data_loader, model, loss_function, optimizer, threshold, learning=True):  # , without_training=False
    if learning:
        model.train()  # select train mode
    else:
        model.eval()

    sum_loss = 0.0
    num_of_instances = 0
    acc_sum = 0.0

    # progbar = create_progressbar(len(data_loader))

    for i, (data, label) in enumerate(data_loader):
        # we need to clear out the hidden state of the LSTM, detaching it from its history on the last instance.
        # Tensors not supported in DataParallel. You should put Variable to use data_parallel before call forward().
        if GPU_NUM:
            var_targets = Variable(label.float(), requires_grad=False).cuda(async=True)
        else:
            var_targets = Variable(label.float(), requires_grad=False)
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

        # progbar.update(i)
        if i % 5 == 0:
            print('Batch : ', i + 1, '/', len(data_loader), ', BCE in this minibatch: ', loss.data[0])

    # finish_progressbar(progbar)
    return sum_loss / num_of_instances, acc_sum / num_of_instances  # return mean loss


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--optimizer', type=str, default='adam')  # optimizer
    args.add_argument('--epochs', type=int, default=20)  # 20
    args.add_argument('--batch', type=int, default=128)  # 200
    args.add_argument('--hidden', type=int, default=512)  # 512
    args.add_argument('--hidden-dense', type=int, default=256)  # 256
    args.add_argument('--dropout', type=float, default=0.15) # 0.15
    args.add_argument('--threshold', type=float, default=0.5)  # 0.5
    args.add_argument('--layers', type=int, default=2)  # 2
    args.add_argument('--initial_lr', type=float, default=0.01)  # default : 0.01 (initial learning rate)
    args.add_argument('--output', type=int, default=1)  # Do not change this
    args.add_argument('--mode', type=str, default='train')  # 'train' or 'test' (for nsml)
    args.add_argument('--pause', type=int, default=0)  # Do not change this (for nsml)
    args.add_argument('--iteration', type=str, default='0')  # Do not change this (for nsml)

    initial_time = time.time()
    config = args.parse_args()

    model = LSTMRegression(
        config.hidden, config.hidden_dense, config.layers,
        config.output, config.batch, config.dropout
    )

    wv_model = Preprocessor()


    def kin_data_loader(dataset_path, train=False, batch_size=200,
                    ratio_of_validation=0.1, shuffle=True):
        if train:
            return get_dataloaders(dataset_path=dataset_path, batch_size=batch_size,
                                   ratio_of_validation=ratio_of_validation,
                                   shuffle=shuffle, preprocessor=wv_model)
        else:
            data_dict = {'data': read_test_file(dataset_path=DATASET_PATH, preprocessor=wv_model)}
            return data_dict

    loss_function = nn.BCELoss()
    if GPU_NUM:
        model = model.cuda()
        loss_function = loss_function.cuda()
    threshold = config.threshold

    if config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.initial_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)

    comp_model = (model, wv_model)

    bind_model(comp_model, config)

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        data_path = "./dummy/kin/"

        if HAS_DATASET:
            data_path = DATASET_PATH

        train_loader, val_loader = kin_data_loader(dataset_path=data_path, train=True,
                                                   batch_size=config.batch, ratio_of_validation=0.1,
                                                   shuffle=True)

        min_val_loss = np.inf
        for epoch in range(config.epochs):
            print('[Train #%d]' % epoch)
            print()
            # train on train set
            train_loss, train_acc = inference_loop(train_loader, model, loss_function,
                                                   optimizer, threshold, learning=True)
            # evaluate on validation set
            val_loss, val_acc = inference_loop(val_loader, model, loss_function,
                                               None, threshold, learning=False)

            print()
            print('train_loss: %.5f' % train_loss, 'train_acc   : %.5f' % train_acc)
            print('val_loss  : %.5f' % val_loss,   'min_val_loss: %.5f' % min_val_loss, 'val_acc: %.5f' % val_acc)
            print()
            print()

            nsml.report(summary=True, scope=locals(), epoch=epoch,
                        total_epoch=config.epochs, val_acc=val_acc,
                        train_acc=train_acc, train__loss=train_loss,
                        val__loss=val_loss, min_val_loss=min_val_loss,
                        step=epoch)

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                nsml.save(epoch)

            else:  # default save
                if epoch % 10 == 0:
                    nsml.save(epoch)
