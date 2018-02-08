import io
import tempfile
import pickle

import numpy as np
import torch
from torch.autograd import Variable

import vocab

import sys
sys.path.append('/mnt/matylda5/ibenes/projects/santosh-lm/smm-pytorch/')
from smm import SMM, update_ws 

class IvecExtractor():
    def __init__(self, model, nb_iters, lr, tokenizer):
        self._model = model
        self._nb_iters = nb_iters
        self._lr = lr
        self._tokenizer = tokenizer

    def __call__(self, sentence):
        """ Extract i-vectors given the model and stats """
        data = self._tokenizer.transform([sentence])
        data = torch.from_numpy(data.A.astype(np.float32))

        self._model.reset_w(data.size(0))  # initialize i-vectors to zeros
        opt_w = torch.optim.Adagrad([self._model.W], lr=self._lr)

        if self._model.cuda:
            data = data.cuda()

        X = Variable(data.t())
        loss = self._model.loss(X)

        for i in range(self._nb_iters):
            loss = update_ws(self._model, opt_w, loss, X)

        return self._model.W.data.t()

    def save(self, f):
        tmp_f = tempfile.TemporaryFile()
        # self._model.cpu()
        torch.save(self._model, tmp_f)
        tmp_f.seek(0)
        model_bytes = io.BytesIO(tmp_f.read())

        nb_iters_bytes = io.BytesIO()
        pickle.dump(self._nb_iters, nb_iters_bytes)

        lr_bytes = io.BytesIO()
        pickle.dump(self._lr, lr_bytes)

        tokenizer_byters = io.BytesIO()
        pickle.dump(self._tokenizer, tokenizer_byters)

        complete_smm = {'model': model_bytes, 'tokenizer': tokenizer_byters,
                        'lr': lr_bytes, 'nb_iters': nb_iters_bytes}
        pickle.dump(complete_smm, f)


def load_smm(f):
    complete_lm = pickle.load(f)

    model_bytes = complete_lm['model']
    tmp_f = tempfile.TemporaryFile()
    tmp_f.write(model_bytes.getvalue())
    tmp_f.seek(0)
    model = torch.load(tmp_f)

    tokenizer_bytes = complete_lm['tokenizer']
    tokenizer_bytes.seek(0)
    tokenizer = pickle.load(tokenizer_bytes)

    lr_bytes = complete_lm['lr']
    lr_bytes.seek(0)
    lr = pickle.load(lr_bytes)

    nb_iters_bytes = complete_lm['nb_iters']
    nb_iters_bytes.seek(0)
    nb_iters = pickle.load(nb_iters_bytes)

    return IvecExtractor(model, nb_iters, lr, tokenizer)
