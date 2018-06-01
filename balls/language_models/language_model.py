import io
import pickle
import tempfile
import torch



class LanguageModel():
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def save(self, f):
        tmp_f = tempfile.TemporaryFile()
        was_on_cuda = next(self.model.parameters()).is_cuda
        self.model.cpu()
        torch.save(self.model, tmp_f)
        tmp_f.seek(0)
        model_bytes = io.BytesIO(tmp_f.read())
        if was_on_cuda:
            self.model.cuda()

        vocab_bytes = io.BytesIO()
        pickle.dump(self.vocab, vocab_bytes)

        complete_lm = {'model': model_bytes, 'vocab': vocab_bytes}
        pickle.dump(complete_lm, f)


def load(f):
    complete_lm = pickle.load(f)

    model_bytes = complete_lm['model']
    tmp_f = tempfile.TemporaryFile()
    tmp_f.write(model_bytes.getvalue())
    tmp_f.seek(0)
    model = torch.load(tmp_f)

    vocab_bytes = complete_lm['vocab']
    vocab_bytes.seek(0)
    vocab = pickle.load(vocab_bytes)

    return LanguageModel(model, vocab)
