import re
import sys
from collections import Counter

sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages')
import argparse
import json
import os
import random
import itertools
import math
from collections import defaultdict
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
_linear = core_rnn_cell._linear
#from tensorflow.python.ops.rnn_cell import _linear
#from ../basic.graph_handler import GraphHandler
#from my.tensorflow import grouper
#from my.utils import index
from tqdm import tqdm
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

d = 100
VERY_NEGATIVE = -1e30

batch_size  = 60


def get_2d_spans(text, tokenss):
    spanss = []
    cur_idx = 0
    for tokens in tokenss:
        spans = []
        for token in tokens:
            if text.find(token, cur_idx) < 0:
                print(tokens)
                print("{} {} {}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss


def get_word_span(context, wordss, start, stop):
    spanss = get_2d_spans(context, wordss)
    idxs = []
    for sent_idx, spans in enumerate(spanss):
        for word_idx, span in enumerate(spans):
            if not (stop <= span[0] or start >= span[1]):
                idxs.append((sent_idx, word_idx))

    assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
    return idxs[0], (idxs[-1][0], idxs[-1][1] + 1)

def get_word_idx(context, wordss, idx):
    spanss = get_2d_spans(context, wordss)
    return spanss[idx[0]][idx[1]][0]




def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


class Data(object):
    def get_size(self):
        raise NotImplementedError()

    def get_by_idxs(self, idxs):
        """
        Efficient way to obtain a batch of items from filesystem
        :param idxs:
        :return dict: {'X': [,], 'Y', }
        """
        data = defaultdict(list)
        for idx in idxs:
            each_data = self.get_one(idx)
            for key, val in each_data.items():
                data[key].append(val)
        return data

    def get_one(self, idx):
        raise NotImplementedError()

    def get_empty(self):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()


class DataSet(object):
    def __init__(self, data, data_type, shared=None, valid_idxs=None):
        self.data = data  # e.g. {'X': [0, 1, 2], 'Y': [2, 3, 4]}
        self.data_type = data_type
        self.shared = shared
        total_num_examples = self.get_data_size()
        self.valid_idxs = range(total_num_examples) if valid_idxs is None else valid_idxs
        self.num_examples = len(self.valid_idxs)

    def _sort_key(self, idx):
        rx = self.data['*x'][idx]
        x = self.shared['x'][rx[0]][rx[1]]
        return max(map(len, x))

    def get_data_size(self):
        if isinstance(self.data, dict):
            return len(next(iter(self.data.values())))
        elif isinstance(self.data, Data):
            return self.data.get_size()
        raise Exception()

    def get_by_idxs(self, idxs):
        if isinstance(self.data, dict):
            out = defaultdict(list)
            for key, val in self.data.items():
                out[key].extend(val[idx] for idx in idxs)
            return out
        elif isinstance(self.data, Data):
            return self.data.get_by_idxs(idxs)
        raise Exception()

def load_metadata(config, data_type):
    metadata_path = os.path.join(config.data_dir, "metadata_{}.json".format(data_type))
    with open(metadata_path, 'r') as fh:
        metadata = json.load(fh)
        for key, val in metadata.items():
            config.__setattr__(key, val)
        return metadata



def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(args.source_dir, "{}-v1.1.json".format(data_type))
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    inds = []

    qind = -1
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        xp, cxp = [], []
        pp = []
        x.append(xp)
        cx.append(cxp)
        p.append(pp)
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            # given xi, add chars
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                #qind += 1
                cqi = [list(qij) for qij in qi]
                yi = []
                cyi = []
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    # TODO : put some function that gives word_start, word_stop here
                    yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                    # yi0 = answer['answer_word_start'] or [0, 0]
                    # yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    w0 = xi[yi0[0]][yi0[1]]
                    w1 = xi[yi1[0]][yi1[1]-1]
                    i0 = get_word_idx(context, xi, yi0)
                    i1 = get_word_idx(context, xi, (yi1[0], yi1[1]-1))
                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1
                    # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                    assert answer_text[-1] == w1[cyi1]
                    assert cyi0 < 32, (answer_text, w0)
                    assert cyi1 < 32, (answer_text, w1)

                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])
                    #qind += 1
                    #inds.append([ai, pi, qind])

               
            for qij in qi:
                word_counter[qij] += 1
                lower_word_counter[qij.lower()] += 1
                for qijk in qij:
                    char_counter[qijk] += 1

            q.append(qi)
            qind += 1
            cq.append(cqi)
            y.append(yi)
            cy.append(cyi)
            rx.append(rxi)
            rcx.append(rxi)
            inds.append([ai, pi, qind])
            ids.append(qa['id'])
            idxs.append(len(idxs))
            answerss.append(answers)

        if args.debug:
            break

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict, 'inds': inds}

    print("saving ...")
    save(args, data, shared, out_name)



def read_data(config, data_type, ref, data_filter=None):
    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(config.data_dir, "shared_{}.json".format(data_type))
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)

    num_examples = len(next(iter(data.values())))
    if data_filter is None:
        valid_idxs = range(num_examples)
    else:
        mask = []
        keys = data.keys()
        values = data.values()
        for vals in zip(*values):
            each = {key: val for key, val in zip(keys, vals)}
            mask.append(data_filter(each, shared))
        valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

    print("Loaded {}/{} examples from {}".format(len(valid_idxs), num_examples, data_type))

    shared_path = config.shared_path or os.path.join(out_dir, "shared.json")
    if not ref:
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        word_counter = shared['lower_word_counter'] if config.lower_word else shared['word_counter']
        char_counter = shared['char_counter']
        if config.finetune:
            shared['word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th or (config.known_if_glove and word in word2vec_dict))}
        else:
            assert config.known_if_glove
            assert config.use_glove_for_unk
            shared['word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th and word not in word2vec_dict)}
        shared['char2idx'] = {char: idx + 2 for idx, char in
                              enumerate(char for char, count in char_counter.items()
                                        if count > config.char_count_th)}
        NULL = "-NULL-"
        UNK = "-UNK-"
        shared['word2idx'][NULL] = 0
        shared['word2idx'][UNK] = 1
        shared['char2idx'][NULL] = 0
        shared['char2idx'][UNK] = 1
        json.dump({'word2idx': shared['word2idx'], 'char2idx': shared['char2idx']}, open(shared_path, 'w'))
    else:
        new_shared = json.load(open(shared_path, 'r'))
        for key, val in new_shared.items():
            shared[key] = val

    if config.use_glove_for_unk:
        # create new word2idx and word2vec
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
        shared['new_word2idx'] = new_word2idx_dict
        offset = len(shared['word2idx'])
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        shared['new_emb_mat'] = new_emb_mat

    data_set = DataSet(data, data_type, shared=shared, valid_idxs=valid_idxs)
    return data_set


def get_squad_data_filter(config):
    def data_filter(data_point, shared):
        assert shared is not None
        rx, rcx, q, cq, y = (data_point[key] for key in ('*x', '*cx', 'q', 'cq', 'y'))
        x, cx = shared['x'], shared['cx']
        if len(q) > config.ques_size_th:
            return False

        # x filter
        xi = x[rx[0]][rx[1]]
        if config.squash:
            for start, stop in y:
                stop_offset = sum(map(len, xi[:stop[0]]))
                if stop_offset + stop[1] > config.para_size_th:
                    return False
            return True

        if config.single:
            for start, stop in y:
                if start[0] != stop[0]:
                    return False

        if config.data_filter == 'max':
            for start, stop in y:
                    if stop[0] >= config.num_sents_th:
                        return False
                    if start[0] != stop[0]:
                        return False
                    if stop[1] >= config.sent_size_th:
                        return False
        elif config.data_filter == 'valid':
            if len(xi) > config.num_sents_th:
                return False
            if any(len(xij) > config.sent_size_th for xij in xi):
                return False
        elif config.data_filter == 'semi':
            """
            Only answer sentence needs to be valid.
            """
            for start, stop in y:
                if stop[0] >= config.num_sents_th:
                    return False
                if start[0] != start[0]:
                    return False
                if len(xi[start[0]]) > config.sent_size_th:
                    return False
        else:
            raise Exception()

        return True
    return data_filter


def update_config(config, data_sets):
    config.max_num_sents = 0
    config.max_sent_size = 0
    config.max_ques_size = 0
    config.max_word_size = 0
    config.max_para_size = 0
    for data_set in data_sets:
        data = data_set.data
        shared = data_set.shared
        for idx in data_set.valid_idxs:
            rx = data['*x'][idx]
            q = data['q'][idx]
            sents = shared['x'][rx[0]][rx[1]]
            config.max_para_size = max(config.max_para_size, sum(map(len, sents)))
            config.max_num_sents = max(config.max_num_sents, len(sents))
            config.max_sent_size = max(config.max_sent_size, max(map(len, sents)))
            config.max_word_size = max(config.max_word_size, max(len(word) for sent in sents for word in sent))
            if len(q) > 0:
                config.max_ques_size = max(config.max_ques_size, len(q))
                config.max_word_size = max(config.max_word_size, max(len(word) for word in q))

    if config.mode == 'train':
        config.max_num_sents = min(config.max_num_sents, config.num_sents_th)
        config.max_sent_size = min(config.max_sent_size, config.sent_size_th)
        config.max_para_size = min(config.max_para_size, config.para_size_th)

    config.max_word_size = min(config.max_word_size, config.word_size_th)

    config.char_vocab_size = len(data_sets[0].shared['char2idx'])
    config.word_emb_size = len(next(iter(data_sets[0].shared['word2vec'].values())))
    config.word_vocab_size = len(data_sets[0].shared['word2idx'])

    #if config.single:
     #   config.max_num_sents = 1
    #if config.squash:
     #   config.max_sent_size = config.max_para_size
      #  config.max_num_sents = 1





def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/squad"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    # TODO : put more args here
    return parser.parse_args()



args = get_args()


flags = tf.app.flags

# Names and directories
flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_string("data_dir", "data/squad", "Data dir [data/squad]")
flags.DEFINE_string("run_id", "0", "Run ID [0]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")
flags.DEFINE_string("forward_name", "single", "Forward name [single]")
flags.DEFINE_string("answer_path", "", "Answer path []")
flags.DEFINE_string("eval_path", "", "Eval path []")
flags.DEFINE_string("load_path", "", "Load path []")
flags.DEFINE_string("shared_path", "", "Shared path []")

# Device placement
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")
flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")

# Essential training and test options
flags.DEFINE_string("mode", "test", "trains | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")
flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
flags.DEFINE_bool('load_ema', True, "load exponential average of variables when testing?  [True]")
flags.DEFINE_bool("eval", True, "eval? [True]")

# Training / test parameters
flags.DEFINE_integer("batch_size", 60, "Batch size [60]")
flags.DEFINE_integer("max_num_sents", 0, "max num sents [0]")
flags.DEFINE_integer("max_sent_size", 0, "max sent size [0]")
flags.DEFINE_integer("max_ques_size", 0, "max quest size [0]")
flags.DEFINE_integer("max_word_size", 0, "max word size [0]")
flags.DEFINE_integer("max_para_size", 0, "max para size [0]")
#flags.DEFINE_integer("word_vocab_size", 0, "Word vocab size [0]")

flags.DEFINE_integer("char_vocab_size", 0, "char vocab size [0]")
flags.DEFINE_integer("word_emb_size", 0, "word emb size [0]")
flags.DEFINE_integer("word_vocab_size", 0, "word emb size [0]")

flags.DEFINE_integer("val_num_batches", 100, "validation num batches [100]")
flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
flags.DEFINE_integer("num_epochs", 12, "Total number of epochs for training [12]")
flags.DEFINE_integer("num_steps", 20000, "Number of steps [20000]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]")
flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
flags.DEFINE_integer("hidden_size", 100, "Hidden size [100]")
flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_string("out_channel_dims", "100", "Out channel dims of Char-CNN, separated by commas [100]")
flags.DEFINE_string("filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]")
flags.DEFINE_bool("finetune", False, "Finetune word embeddings? [False]")
flags.DEFINE_bool("highway", True, "Use highway? [True]")
flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
flags.DEFINE_float("var_decay", 0.999, "Exponential moving average decay for variables [0.999]")

# Optimizations
flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")
flags.DEFINE_bool("len_opt", False, "Length optimization? [False]")
flags.DEFINE_bool("cpu_opt", False, "CPU optimization? GPU computation can be slower [False]")

# Logging and saving options
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
flags.DEFINE_integer("eval_period", 1000, "Eval period [1000]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

# Thresholds for speed and less memory usage
flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 400, "sent size th [64]")
flags.DEFINE_integer("num_sents_th", 8, "num sents th [8]")
flags.DEFINE_integer("ques_size_th", 30, "ques size th [32]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("para_size_th", 256, "para size th [256]")

# Advanced training options
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
flags.DEFINE_bool("known_if_glove", True, "consider as known if present in glove [False]")
flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")

# Ablation options
flags.DEFINE_bool("use_char_emb", False, "use char emb? [True]")
flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")
flags.DEFINE_bool("q2c_att", True, "question-to-context attention? [True]")
flags.DEFINE_bool("c2q_att", True, "context-to-question attention? [True]")
flags.DEFINE_bool("dynamic_att", False, "Dynamic attention [False]")
#flags.DEFINE_string("out_dir", os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2)), "out  dir [out_dir]")
config = flags.FLAGS

out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))


#Please uncomment the following two lines when runninng for the first time
#prepro_each(args, 'train', out_name='train')
#prepro_each(args, 'dev', out_name='dev')



data_filter = get_squad_data_filter(config)
train_data = read_data(config, 'train', False, data_filter=data_filter)
dev_data = read_data(config, 'dev', False, data_filter=data_filter)
update_config(config, [train_data, dev_data])

word2vec_dict = train_data.shared['lower_word2vec'] if config.lower_word else train_data.shared['word2vec']
word2idx_dict = train_data.shared['word2idx']
idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}


print("idx keys ", idx2vec_dict.keys())
emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
                        else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for idx in range(config.word_vocab_size)])
print(emb_mat)

new_emb_mat = train_data.shared['new_emb_mat'] 
print("new embat mat ", new_emb_mat)
emb_mat = tf.concat( [emb_mat.astype('float32'), new_emb_mat], 0)


#print(emb_mat)



#print(train_data.shared['inds'])
#print(len(train_data.data['q']))

def _get_word(word):
    d = train_data.shared['word2idx']
    #d = train_data.shared['new_word2idx']
    for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in d:
            return d[each]
    if config.use_glove_for_unk:
        #d2 = batch.shared['new_word2idx']
        d2 = train_data.shared['new_word2idx']
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in d2:
                return d2[each] + len(d)
    return 1










M = 1
JX = 20*4
JQ = 30



xval = tf.placeholder(tf.int32, [batch_size, M*JX])
qval = tf.placeholder(tf.int32, [batch_size, M*JQ])
x_mask = tf.placeholder( 'bool', [ batch_size, M*JX])
q_mask = tf.placeholder('bool', [batch_size, M*JQ])
Ax = tf.nn.embedding_lookup(emb_mat, xval)
Aq = tf.nn.embedding_lookup(emb_mat, qval)

Axx = tf.reshape(Ax, [ batch_size*M*JX, d])

#with tf.variable_scope("hway2"):
trans = _linear([Axx], d, True)
trans = tf.reshape(trans, [ batch_size, M*JX, d])
trans = tf.nn.relu(trans)
tf.get_variable_scope().reuse_variables()
gate = _linear(Axx, d, True)
gate = tf.reshape( gate, [ batch_size, M*JX, d])
gate = tf.nn.sigmoid(gate)
Axx = tf.reshape( Axx, [batch_size, M*JX, d])
Axx = gate*trans + (1- gate)*Axx
Aq= tf.reshape(Aq, [ batch_size*M*JQ, d])
#tf.get_variable_scope().reuse_variables()
trans = _linear( [Aq], d, True)
trans = tf.reshape( trans, [ batch_size, M*JQ, d])
trans = tf.nn.relu(trans)
gate = _linear( Aq, d, True)
gate = tf.reshape( gate, [batch_size, M*JQ, d])
gate = tf.nn.sigmoid(gate)
Aqq = tf.reshape(Aq, [batch_size, M*JQ, d])
Aqq = gate*trans + (1 - gate)*Aqq





d = 100
WC = M*JX
WQ = M*JQ


xx = Axx
qq = Aqq
x_len = tf.reduce_sum(tf.cast(x_mask, 'int32'), 1)
q_len = tf.reduce_sum(tf.cast(q_mask, 'int32'), 1)


print("the shape os xx is ", xx)
with tf.variable_scope('backward', reuse=tf.AUTO_REUSE):
    lstmcbw = BasicLSTMCell(100, state_is_tuple = True)
    lstmcfw = BasicLSTMCell(100, state_is_tuple = True)
    outputsx, output_statesx = bidirectional_dynamic_rnn(lstmcfw, lstmcbw, xx,  dtype = tf.float32)
    print("the shape of outputsx is ", outputsx)
    print(" the shape of outputs states x ", output_statesx)
    outx_fw, outx_bw = outputsx
    outxx = tf.concat([outx_fw, outx_bw], axis = 2)
    print('out xx ', outxx)
with tf.variable_scope("qcell",  reuse=tf.AUTO_REUSE):
    lstmcbw = BasicLSTMCell(100, state_is_tuple = True)
    lstmcfw = BasicLSTMCell(100, state_is_tuple = True)
    outputsq, output_statesq = bidirectional_dynamic_rnn(lstmcfw, lstmcbw,  qq,  dtype = tf.float32)
    outqfw, outqbw = outputsq
    outqq = tf.concat([outqfw, outqbw], axis = 2)
    print('out qq is ', outqq)

outx = outxx #outputsx #tf.transpose(outx, [1,0,2])
outq = outqq #outputsq #tf.transpose(outq, [1,0,2])

with tf.variable_scope("sec_layer", reuse=tf.AUTO_REUSE):
    w = tf.get_variable( "w", shape = [batch_size, 1, 6*d])

w_t = tf.tile(w, [ 1, WC*WQ, 1])

w_r = w_t

h = outx
u = outq
h_stacked = h#tf.stack(h)
u_stacked = u #tf.stack(u)

print("shape of h_stacked ", h_stacked)
h_aug = tf.tile( h_stacked, [ 1, WQ, 1])
u_aug = tf.tile( u_stacked, [ 1, 1, WC])

h_mask = tf.tile(tf.expand_dims(x_mask, 2), [1, 1, WQ])
u_mask = tf.transpose( tf.tile(tf.expand_dims(q_mask, 2), [1, 1, WC]), [0,2,1])
print("hmask ", h_mask)
print(" umask " , u_mask)
s_mask = h_mask & u_mask

print("smask shape is ", s_mask)


u_aug = tf.reshape( u_aug, [ batch_size, WC*WQ, 200])

print('shape of u_aug ', u_aug)

hu_conc = tf.concat( [ h_aug, u_aug, tf.multiply( h_aug, u_aug)], axis = 2)

alpha = tf.multiply( w_t, hu_conc)
alpha = tf.reduce_sum( alpha, 2)

S = tf.reshape( alpha, [ batch_size, WC, WQ])
S = tf.add(S, (1 - tf.cast(s_mask, 'float'))*VERY_NEGATIVE )




S_soft =tf.map_fn ( lambda x : tf.nn.softmax(x), S)

flat_s = tf.reshape(S_soft, [batch_size, WC*WQ, 1])
flat_s = tf.tile(flat_s, [ 1, 1, 200])
u_rep  = tf.tile( u_stacked , [ 1, WC, 1])

produ = tf.multiply( flat_s, u_rep)


split_p = tf.split(produ, WC, axis = 1 )
u_tild = tf.reduce_sum(split_p, 2)
u_tild = tf.transpose(u_tild, [1, 0, 2])


#aa = tf.reshape( tf.map_fn( lambda x : tf.nn.softmax(x), S) , [ batch_size, M*M*JX*JQ] )
#brd = tf.reshape( tf.expand_dims( aa, -1)*u_rep, [ batch_size, M*JX, M*JQ, 2*d])
#u_tild = tf.reduce_sum( brd, 2)





bt = tf.nn.softmax( tf.reduce_max(S, reduction_indices = [2]))
btex = tf.expand_dims( bt, 2)
b_til = tf.tile( btex, [ 1, 1, 2*d])

elmul = tf.multiply( h_stacked, b_til)

htild = tf.reduce_sum( elmul, 1)
htild = tf.expand_dims( htild, 1)
htild = tf.tile( htild, [ 1, WC, 1])


with tf.variable_scope("finallayers", reuse=tf.AUTO_REUSE):
    G = tf.concat( [ h_stacked, u_tild, tf.multiply( h_stacked, u_tild), tf.multiply(h_stacked, htild)], axis = 2)
    print('shape of G is ', G)
    lstmcbw = BasicLSTMCell(100, state_is_tuple = True)
    lstmcfw = BasicLSTMCell(100, state_is_tuple = True)
    outg, g_states = bidirectional_dynamic_rnn( lstmcfw, lstmcbw, G, dtype = tf.float32)
    
    outgfw , outgbw = outg

    print('outg fw, ', outgfw)
    #outg = tf.transpose(outg, [1,0,2])
    outg = tf.concat([outgfw, outgbw], axis = 2)
    print('outg ', outg)
    print('gfw ', outgfw)
    print('gbw ', outgbw)
with tf.variable_scope("double", reuse=tf.AUTO_REUSE):
    lstmcbw = BasicLSTMCell(100, state_is_tuple = True)
    lstmcfw = BasicLSTMCell(100, state_is_tuple = True)
    outgg, gstates = bidirectional_dynamic_rnn ( lstmcfw, lstmcbw, outg, dtype = tf.float32)
    #outgg = tf.pack(outgg)
    #outgg = tf.transpose(outgg, [1,0,2])
    outggfw, outggbw = outgg
    outgg = tf.concat([outggfw, outggbw], axis = 2)
    #outggfw, outggbw = outgg
    print('shape of outggfw is ', outggfw)


with tf.variable_scope( "fin", reuse=tf.AUTO_REUSE):
    wf1 = tf.get_variable( "wf1", shape = [ batch_size, 1, 10*d])
    MN = outgg
    print("shape og G", G)
    print('shape of MN ', MN)
    lstmcbw = BasicLSTMCell(100, state_is_tuple = True)
    lstmcfw = BasicLSTMCell(100, state_is_tuple = True)
    p0 = tf.matmul( wf1, tf.transpose( tf.concat (  [ G, MN], axis = 2), [0, 2, 1]))
    p0 = tf.reshape( p0, [ batch_size, M*JX])
    #p0 = tf.add(p0 , ( 1 - tf.cast(x_mask, 'float'))*VERY_NEGATIVE)
    outm2, m2states = bidirectional_dynamic_rnn( lstmcfw, lstmcbw, MN, dtype = tf.float32)
    outm2fw, outm2bw = outm2
   
    #M2 = tf.pack(outm2)
    #M2 = tf.transpose(M2, [1,0,2])
    M2 = tf.concat([outm2fw, outm2bw], axis = 2)
    outggr = tf.reshape(outgg, [batch_size,M*JX, 2*d])
    a = tf.nn.softmax(p0)
    print("a hspae ", a)
   
    out = tf.reduce_sum(tf.expand_dims(a, -1)*outggr, 1)
    print("outshpa ie ", out)
    p0t = tf.tile(tf.expand_dims( out , 1), [1, M*JX, 1])
    print("p0 t is ", p0t)
    #p0t = tf.tile( tf.expand_dims(tf.reshape(outgg, [batch_size, M*JX, 2*d], -1), [1, 1, 200])
    wf2 = tf.get_variable("wf2",  shape = [ batch_size, 1, 10*100])
    p1 = tf.matmul(wf2, tf.transpose( tf.concat( [ G, M2], axis = 2), [0, 2, 1]))
    p1 = tf.reshape( p1, [ batch_size, M*JX])
    #p1 = tf.add( p1, ( 1 - tf.cast( x_mask, 'float'))*VERY_NEGATIVE)

    lhs_inds = tf.placeholder(tf.int32, [batch_size])
    rhs_inds = tf.placeholder(tf.int32, [batch_size])

    lhs_acts = tf.placeholder( tf.int32, [ batch_size])
    rhs_acts = tf.placeholder( tf.int32, [ batch_size])



#p0 = tf.nn.softmax(p0, -1)
#p1 = tf.nn.softmax(p1, -1)
    p0g = tf.gather(tf.reshape(p0, [batch_size*M*JX]), lhs_inds)
    p1g = tf.gather(tf.reshape(p1, [batch_size*M*JX]), rhs_inds)
    ambeg = tf.argmax(p0, 1)
    amend  = tf.argmax(p1, 1)

    fl = tf.log(p0g)
    sl = tf.log(p1g)


    logl = tf.one_hot(indices = lhs_acts, depth = M*JX )
    logr = tf.one_hot( indices = rhs_acts, depth = M*JX)


    loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits = p0 , labels = lhs_acts) + tf.nn.sparse_softmax_cross_entropy_with_logits(logits = p1, labels = rhs_acts) )#tf.reduce_mean(tf.mul ( fl + sl, tf.constant(-1.0)))
                         

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    optimizer = tf.train.AdamOptimizer(0.0005)


    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    var_ema = tf.train.ExponentialMovingAverage(config.var_decay)

    ema_op = var_ema.apply(tf.trainable_variables())

def get_batch_data():
    shufinds = train_data.shared['inds']
    random.shuffle(shufinds)
    shufinds = itertools.cycle(shufinds)
    while(True): 
        lhs = []
        rhs = []
        rhs_acts = []
        lhs_acts = []
        bind = 0
        x = np.zeros([ batch_size, M*JX], dtype = 'int32')
        q = np.zeros( [ batch_size, M*JQ], dtype = 'int32')
        x_mask = np.zeros([batch_size, M*JX], dtype = 'bool')
        q_mask = np.zeros([batch_size, M*JQ], dtype = 'bool')
    
        while len(lhs) < batch_size:
            smp = next(shufinds)
            too_large = False
            index = 0
            artind = smp[0]
            parind = smp[1]
            qind = smp[2]
            for i, xi in enumerate(train_data.shared['x'][artind][parind]):
                for j, xj in enumerate(xi):
                    each = _get_word(xj)
                    if index < M*JX:
                        x[bind, index] = each
                        x_mask[bind, index]= True
                    else:
                        too_large = True
                    index += 1
            ques = train_data.data['q'][qind]
            for i, word in enumerate(ques):
                if i < M*JQ:
                    q[bind, i] = _get_word(word)
                    q_mask[bind, i] = True
                else:
                    too_large = True

            ans = train_data.data['y'][qind]
            
            lh = ans[0][0][1]
            rh = ans[0][1][1]
 

            if (not lh < JX) or not (rh < JX):
                too_large = True
            if too_large:
                q[bind, :] =0
                x[bind, :] = 0
                x_mask[bind, :] = 0  
                q_mask[bind, :] = 0  
            if not too_large:
                lhs.append([M*JX*bind + lh])
                rhs.append([M*JX*bind + rh])
                lhs_acts.append([lh])
                rhs_acts.append([rh])
                bind += 1
        yield x, x_mask,q_mask,   q, lhs, rhs, lhs_acts, rhs_acts


sess.run(tf.initialize_all_variables())
#saver = tf.train.Saver()
gentrain  = get_batch_data()
for i in range(0, 100000000):
    xb, x_mb, q_mb,  qb, lhsb, rhsb, lact, ract = next(gentrain)
    fd = {xval: xb, x_mask: x_mb, q_mask: q_mb, qval: qb, lhs_inds: np.reshape(lhsb, (batch_size, )), rhs_inds: np.reshape( rhsb, (batch_size,)), lhs_acts : np.reshape( lact, (batch_size,)), rhs_acts : np.reshape( ract, (batch_size,))  }          
    print("iter", i)
    #if i % 1000 == 0:
        #saver.save(sess, "flat_trans_working") 
    trop, lo = sess.run([train_op, loss], feed_dict = fd)
    sess.run([ema_op], feed_dict = fd )
    print(sess.run([ambeg], feed_dict = fd))
    print(sess.run([amend], feed_dict = fd))
    print(x_mb)
    print(xb)
    print("p0 is ", sess.run([p0], feed_dict = fd))
    print( "p1 is ", sess.run( [p1], feed_dict= fd))
    print(lact)
    print(ract)
    print("loss ", lo) 
