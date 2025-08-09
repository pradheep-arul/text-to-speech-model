
CHARACTER_TOKENS = "abcdefghijklmnopqrstuvwxyz .,'?!"



class CharTokenizer:
    def __init__(self):
        chars = list(CHARACTER_TOKENS)
        self.pad_token = "_"
        self.vocab = [self.pad_token] + chars
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

    def encode(self, text):
        text = text.lower()
        return [self.char2idx.get(ch, self.char2idx[self.pad_token]) for ch in text]

    def decode(self, indices):
        return "".join([self.idx2char.get(idx, self.pad_token) for idx in indices])
