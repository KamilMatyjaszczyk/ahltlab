
import torch
import torch.nn as nn
import torch.nn.functional as func


criterion = nn.CrossEntropyLoss()

class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        n_lc_words = codes.get_n_lc_words()
        n_words = codes.get_n_words()
        n_feat = codes.get_n_features()
        n_labels = codes.get_n_labels()
        n_pos = codes.get_n_pos()
        self.suf_lens = codes.get_suf_lens()
        self.pref_lens = codes.get_pref_lens()

        embLWsize = 100
        embWsize = 100
        embSsize = 50
        embPsize = 50
        embPOSsize = 20

        self.embLW = nn.Embedding(n_lc_words, embLWsize)
        self.embW = nn.Embedding(n_words, embWsize)

        self.suf_embs = nn.ModuleDict({
            str(L): nn.Embedding(codes.get_n_sufs(L), embSsize)
            for L in self.suf_lens
        })

        self.pref_embs = nn.ModuleDict({
            str(L): nn.Embedding(codes.get_n_prefs(L), embPsize)
            for L in self.pref_lens
        })

        self.embPOS = nn.Embedding(n_pos, embPOSsize)
        self.dropLW = nn.Dropout(0.1)
        self.dropW = nn.Dropout(0.1)
        self.dropS = nn.Dropout(0.1)
        self.dropP = nn.Dropout(0.1)
        self.dropPOS = nn.Dropout(0.1)

        lstm_in_size = embLWsize + embWsize
        lstm_in_size += len(self.suf_lens) * embSsize
        lstm_in_size += len(self.pref_lens) * embPsize
        lstm_in_size += embPOSsize
        lstm_in_size += n_feat

        lstm_out_size = 200
        self.lstm = nn.LSTM(lstm_in_size, lstm_out_size, bidirectional=True, batch_first=True)

        linear_out_size = 200
        self.linear = nn.Linear(2 * lstm_out_size, linear_out_size)
        self.out = nn.Linear(linear_out_size, n_labels)

    def forward(self, lw, w, *rest):
        x_lw = self.dropLW(self.embLW(lw))
        x_w = self.dropW(self.embW(w))

        pieces = [x_lw, x_w]
        idx = 0

        for L in self.suf_lens:
            s = rest[idx]
            pieces.append(self.dropS(self.suf_embs[str(L)](s)))
            idx += 1

        for L in self.pref_lens:
            p = rest[idx]
            pieces.append(self.dropP(self.pref_embs[str(L)](p)))
            idx += 1

        pos = rest[idx]
        pieces.append(self.dropPOS(self.embPOS(pos)))
        idx += 1

        f = rest[idx].float()
        pieces.append(f)

        x = torch.cat(pieces, dim=2)
        x = self.lstm(x)[0]
        x = func.relu(x)
        x = self.linear(x)
        x = self.out(x)
        return x


