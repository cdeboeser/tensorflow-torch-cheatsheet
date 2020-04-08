import torch
import numpy as np
import json
from collections import Counter


class TranslationDatasetTorch(torch.utils.data.Dataset):
    def __init__(self,
                 file_name,
                 source_field='src',
                 target_filed='tgt',
                 unk_idx=1,
                 sos_idx=2,
                 eos_idx=3,
                 first_index=4,
                 device='cpu'):
        super(TranslationDatasetTorch, self).__init__()

        self.source_field = source_field
        self.target_filed = target_filed
        self.unk_idx = unk_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

        src_vocab = Counter()
        tgt_vocab = Counter()

        data_tmp = []
        with open(file_name) as f:
            for l in f:
                obj = json.loads(l)
                src_vocab += Counter(obj['src'])
                tgt_vocab += Counter(obj['tgt'])
                data_tmp.append(obj)

        self.stoi_src = {w[0]: c for c, w in enumerate(
            src_vocab.most_common(len(src_vocab)), first_index)}
        self.stoi_tgt = {w[0]: c for c, w in enumerate(
            tgt_vocab.most_common(len(tgt_vocab)), first_index)}

        self.data = []
        for obj in data_tmp:
            src = obj[source_field]
            tgt = obj[target_filed]

            src = [sos_idx] + \
                [self.stoi_src.get(w, unk_idx) for w in src] + [eos_idx]
            tgt = [sos_idx] + \
                [self.stoi_tgt.get(w, unk_idx) for w in tgt] + [eos_idx]

            self.data.append([src, tgt])

    def __getitem__(self, index):
        src, tgt = self.data[index]
        src = torch.from_numpy(np.array(src)).to(self.device)
        tgt = torch.from_numpy(np.array(tgt)).to(self.device)
        src_len = torch.from_numpy(
            np.array(src.size()[0])).unsqueeze(-1).to(self.device)
        return src, src_len, tgt

    def __len__(self):
        return len(self.data)


def collate_fn(inputs):
    src_seqs, src_lens, tgt_seqs = zip(*inputs)
    src_lens = torch.stack(src_lens)
    order = torch.argsort(src_lens, dim=0, descending=True).squeeze(-1)

    max_len_src = max([x.size(0) for x in src_seqs])
    max_len_tgt = max([x.size(0) for x in tgt_seqs])

    src_seqs = [torch.cat([e, torch.zeros((max_len_src - e.size(0),), dtype=torch.long, device=e.device)])
                for e in src_seqs]

    tgt_seqs = [torch.cat([e, torch.zeros((max_len_tgt - e.size(0),), dtype=torch.long, device=e.device)])
                for e in tgt_seqs]

    src_lens = src_lens[order]

    src_seqs = torch.stack(src_seqs)[order]
    tgt_seqs = torch.stack(tgt_seqs)[order]

    return src_seqs, src_lens, tgt_seqs
