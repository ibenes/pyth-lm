class IndexGenerator():
    def __init__(self, assigned):
        self.next_ = 0
        self.assigned_ = assigned

    def next(self):
        while self.next_ in self.assigned_:
            self.next_ += 1

        retval = self.next_
        self.next_ += 1
        return retval

class Vocabulary:
    def __init__(self, unk_word, unk_index):
        self.w2i_ = {unk_word:unk_index}
        self.i2w_ = {unk_index:unk_word}
        self.unk_index_ = unk_index
        self.unk_word_ = unk_word
        self.ind_gen_ = IndexGenerator([unk_index])

    def add_from_text(self, text):
        assert self.ind_gen_

        words = text.split()
        for word in words:
            if word not in self.w2i_: # unknown word
                index = self.ind_gen_.next()
                self.w2i_[word] = index
                self.i2w_[index] = word
            else:
                pass # do not do anything for known words

    def w2i(self, word):
        return self.w2i_.get(word, self.unk_index_)

    def i2w(self, index):
        return self.i2w_[index]

    def size(self):
        return len(self.w2i_)

    def __len__(self):
        if self.ind_gen_:
            return self.ind_gen_.next_
        else:
            return self.size_

def vocab_from_kaldi_wordlist(f, unk_word='<unk>'):
    d = {}
    for i, line in enumerate(f):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError("Weird line {}: '{}'".format(i, line))
             
        w = fields[0]
        i = int(fields[1])
        assert i >= 0
        d[w] = i

    vocab = Vocabulary(unk_word, d[unk_word]) 
    vocab.ind_gen_ = None
    vocab.w2i_ = d
    for w in d:
        vocab.i2w_[d[w]] = w

    vocab.size_ = max(d.values()) + 1

    return vocab
