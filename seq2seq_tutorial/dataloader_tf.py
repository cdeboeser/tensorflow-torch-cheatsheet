import tensorflow as tf
import json
from collections import Counter


class TranslationDatasetTF(object):
    def __init__(self,
                 file_name,
                 source_field='src',
                 target_filed='tgt',
                 unk_idx=1,
                 sos_idx=2,
                 eos_idx=3,
                 first_index=4):
        self.source_field = source_field
        self.target_filed = target_filed
        self.unk_idx = unk_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

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

            src = ','.join([str(x) for x in src])
            tgt = ','.join([str(x) for x in tgt])

            self.data.append([src, tgt])

    def generate_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(self.data)

        def split_to_int(inp):
            src = tf.strings.to_number(tf.strings.split(
                inp[0], sep=','), out_type=tf.int32)
            src_lens = tf.expand_dims(tf.shape(src)[0], axis=0)
            tgt = tf.strings.to_number(tf.strings.split(
                inp[1], sep=','), out_type=tf.int32)
            return src, src_lens, tgt

        def sort_batch(src, src_lens, tgt):
            sorting_order = tf.argsort(tf.squeeze(
                src_lens, axis=-1), direction='DESCENDING')
            src = tf.gather(src, sorting_order, axis=0)
            tgt = tf.gather(tgt, sorting_order, axis=0)
            src_lens = tf.gather(src_lens, sorting_order, axis=0)
            return src, src_lens, tgt

        def resort_results(src, src_lens, tgt):
            return (src, tgt), tgt[:, 1:]

        dataset = dataset\
            .map(split_to_int)\
            .padded_batch(batch_size, padded_shapes=([None], [1], [None]))\
            .map(sort_batch)\
            .map(resort_results)\
            .cache()

        return dataset
