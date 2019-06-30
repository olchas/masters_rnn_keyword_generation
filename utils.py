# -*- coding: utf-8 -*-
from numpy.random import seed
seed(1)

import collections
import itertools
import os
import re
import string
from six.moves import cPickle

import h5py
import numpy as np
import pandas as pd
import sentencepiece as spm
from nltk.corpus import stopwords

# punctuation_to_leave = "-,!?'/."
punctuation_to_leave = "'"
punctuation_to_remove = string.punctuation + '—'
for sign in punctuation_to_leave:
    punctuation_to_remove = punctuation_to_remove.replace(sign, '')
translator = str.maketrans(punctuation_to_remove, ' ' * len(punctuation_to_remove))

stop_words = set(stopwords.words('english') + ["'s", "'m", "'ve", "n't", "'re", "'d", "'ll"])


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
    """
    if not isinstance(string, str):
        string = str(string)
    # lowercase and remove most of punctuation
    string = string.lower().translate(translator)
    # remove duplicates of any punctuation left, separate certain punctuation from words
    string = re.sub(r"\?+", " ? ", string)
    string = re.sub(r",+", " , ", string)
    string = re.sub(r"!+", " ! ", string)
    string = re.sub(r"-+", "-", string)
    string = re.sub(r"'+", "'", string)
    string = re.sub(r"/+", " / ", string)
    string = re.sub(r"\.+", " . ", string)
    # remove any dashes at the beginning of string or not preceded by a letter
    string = re.sub(r"([^\w]|^)-", r"\1", string)
    # remove any dashes at the end of string or not preceding a letter
    string = re.sub(r"-([^\w]|$)", r"\1", string)
    # separate short forms from modal verbs
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"n \'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    # remove space duplicates
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class TextLoader():
    def __init__(self, load_preprocessed, corpus_type, data_dir, batch_size, seq_length, vocab_size, unk_max_number, unk_max_count, vocabulary, use_bpe, bpe_size, bpe_model_path, pretrained_embeddings=None, use_attention=False, pos_tags=[], encoding=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.corpus_type = corpus_type

        self.attention_model = use_attention

        self.input_file = os.path.join(data_dir, "input.txt")
        self.words_vocab_file = os.path.join(data_dir, "words_vocab.pkl")
        self.key_words_file = os.path.join(data_dir, "input_tagged.txt") if self.attention_model else None
        self.database_file = os.path.join(self.data_dir, self.corpus_type)
        self.embedding_dir = os.path.dirname(pretrained_embeddings) if pretrained_embeddings is not None else None

        # Let's not read voca and data from file. We many change them.
        if not load_preprocessed:
            print("reading text file")
            self.preprocess(vocab_size, unk_max_number, unk_max_count, vocabulary, use_bpe, bpe_size, bpe_model_path, pos_tags, encoding)
            if pretrained_embeddings is not None and self.corpus_type == 'training':
                self.prepare_embedding_matrix(pretrained_embeddings)

            self.prepare_data_matrices()

        else:
            print("Loading preprocessed files")
            if bpe_model_path is not None:
                self.bpe_model_path = bpe_model_path
            elif use_bpe:
                self.bpe_model_path = "bpe.model"
            else:
                self.bpe_model_path = None

            if self.corpus_type == 'training':
                with open(self.words_vocab_file, 'rb') as f:
                    self.words, self.vocab = cPickle.load(f)
                self.vocab_size = len(self.words)
                self.unk_token = self.vocab.get('<unk>')
                self.sos_token = self.vocab.get('<s>')
                self.eos_token = self.vocab.get('</s>')

        print("{} data loaded!".format(self.corpus_type))

        self.database = h5py.File(self.database_file, 'r')
        self.num_batches = len(self.database.keys())

        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.reset_batch_pointer()

    def close(self):
        self.database.close()

    def count_unks(self, sentence):
        return sentence.count(self.unk_token)

    def build_vocab(self, sentences, vocab_size, bpe_model):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        if bpe_model is not None:
            vocabulary_inv = [bpe_model.IdToPiece(x) for x in range(bpe_model.GetPieceSize())]
        else:
            # Build vocabulary

            word_counts = collections.Counter([word for sentence in sentences for word in sentence])

            # Mapping from index to word
            vocabulary_inv = ['<unk>', '<s>', '</s>']

            counts = sorted(list(set(word_counts.values())), reverse=True)
            for count in counts:
                words = sorted([x[0] for x in word_counts.items() if x[1] == count])
                vocabulary_inv += words

            if vocab_size is not None:
                vocabulary_inv = vocabulary_inv[:vocab_size]

        # Mapping from word to index
        vocab = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocab, vocabulary_inv]

    def prepare_embedding_matrix(self, pretrained_embeddings):

        embedding_dict = {}
        with open(pretrained_embeddings) as f:
            if self.bpe_model_path is not None:
                for line in f:
                    items = line.split()
                    word = items[0]
                    if word in self.vocab or word + '▁' in self.vocab or '▁' + word in self.vocab:
                        embedding_dict[word] = [float(item) for item in items[1:]]
            else:
                for line in f:
                    items = line.split()
                    word = items[0]
                    if word in self.vocab:
                        embedding_dict[word] = [float(item) for item in items[1:]]

        print('Embedding dict loaded')

        embedding_size = len(next(iter(embedding_dict.values())))

        # with open(os.path.join(self.embedding_dir, 'embedding_dict.pkl'), 'wb') as f:
        #     cPickle.dump(embedding_dict, f)

        vocabulary_embedding = []
        for word in self.words:
            word_embedding = embedding_dict.get(word.strip('▁').lower())
            if word_embedding is None:
                word_embedding = np.random.rand(embedding_size).tolist()
                print('word missing embedding: ', word)
            vocabulary_embedding.append(word_embedding)
        vocabulary_embedding = np.array(vocabulary_embedding)

        with open(os.path.join(self.embedding_dir, 'embedding_matrix.pkl'), 'wb') as f:
            cPickle.dump(vocabulary_embedding, f)

        print('Embedding matrix saved')

    def prepare_data_matrices(self):

        database = h5py.File(self.database_file, 'w')
        data_file = open(os.path.join(self.data_dir, 'input_cleaned.txt'))
        target_weights_batch = []
        data_batch = []
        target_batch = []
        batch_counter = 0
        if self.attention_model:

            key_words_batch = []
            key_words_count_batch = []

            for line in data_file:
                line_items = line.strip().split('\t')
                sentence = eval(line_items[0])
                key_words = eval(line_items[1])

                key_words_count = len(key_words)

                target_weights_batch.append([1] * len(sentence) + [0] * (self.seq_length - len(sentence)))

                target_sentence = sentence[1:] + [sentence[0]]

                # zero pad to seq_length
                key_words = key_words + [0] * (self.seq_length - key_words_count)
                sentence = sentence + [0] * (self.seq_length - len(sentence))
                target_sentence = target_sentence + [0] * (self.seq_length - len(target_sentence))

                key_words_count_batch.append(key_words_count)
                key_words_batch.append(key_words)
                data_batch.append(sentence)
                target_batch.append(target_sentence)

                if len(data_batch) == self.batch_size:
                    database.create_group(str(batch_counter))
                    database[str(batch_counter)].create_dataset('data', data=np.array(data_batch).transpose())
                    database[str(batch_counter)].create_dataset('target', data=np.array(target_batch).transpose())
                    database[str(batch_counter)].create_dataset('target_weights', data=np.array(target_weights_batch).transpose())

                    # For the attention mechanism, we need to make sure the "memory" passed in is batch major, so we need to transpose attention_states.
                    database[str(batch_counter)].create_dataset('key_words', data=np.array(key_words_batch))
                    database[str(batch_counter)].create_dataset('key_words_count', data=np.array(key_words_count_batch))

                    key_words_batch = []
                    key_words_count_batch = []
                    target_weights_batch = []
                    data_batch = []
                    target_batch = []

                    batch_counter += 1

        else:
            for line in data_file:
                sentence = eval(line.strip().split('\t')[0])

                target_weights_batch.append([1] * len(sentence) + [0] * (self.seq_length - len(sentence)))

                target_sentence = sentence[1:] + [sentence[0]]

                # zero pad to seq_length
                sentence = sentence + [0] * (self.seq_length - len(sentence))
                target_sentence = target_sentence + [0] * (self.seq_length - len(target_sentence))

                data_batch.append(sentence)
                target_batch.append(target_sentence)

                if len(data_batch) == self.batch_size:
                    database.create_group(str(batch_counter))
                    database[str(batch_counter)].create_dataset('data', data=np.array(data_batch).transpose())
                    database[str(batch_counter)].create_dataset('target', data=np.array(target_batch).transpose())
                    database[str(batch_counter)].create_dataset('target_weights', data=np.array(target_weights_batch).transpose())

                    target_weights_batch = []
                    data_batch = []
                    target_batch = []

                    batch_counter += 1

    def preprocess(self, vocab_size, unk_max_number, unk_max_count, vocabulary, use_bpe, bpe_size, bpe_model_path, pos_tags, encoding):

        data = pd.read_csv(self.input_file, sep='\t', header=None, encoding=encoding)

        print("Initial data size: {}".format(len(data)))

        if self.key_words_file is not None:
            data['key_words'] = list(pd.read_csv(self.key_words_file, sep='\t', header=None, encoding=encoding)[0])

            assert len(data['key_words']) == len(data[0]), "Size of input data and tagged data does not match"

            # replace tagged sentence with list of key words from allowed pos tags, not taking stop words into account
            data['key_words'] = data['key_words'].apply(lambda sentence: [word.split('_')[0]
                                                                          for word in sentence.strip().split()
                                                                          if word.endswith(tuple(pos_tags))
                                                                          and word.split('_')[0] not in stop_words])

            # remove lines without any keywords of specified pos tags
            data = data[(data['key_words'].map(len) > 0)]

            print("Size after removing utterances without specified pos tags: {}".format(len(data)))

        # save the file with reduced number of sentences in order to train bpe model from it
        with open(os.path.join(self.data_dir, 'input_cleaned.txt'), 'w') as f:
            f.write('\n'.join(list(data[0])) + '\n')

        if use_bpe or bpe_model_path is not None:

            bpe_model = spm.SentencePieceProcessor()

            if bpe_model_path is None:

                bpe_model_path = "bpe.model"

                spm.SentencePieceTrainer.Train(
                    '--input=' + os.path.join(self.data_dir, 'input_cleaned.txt') + ' --model_prefix=bpe --vocab_size=' + str(bpe_size) + ' --model_type=bpe')

            bpe_model.Load(bpe_model_path)

            # list of encoded sentences with added start and end os sequence tokens
            data[0] = data[0].apply(lambda sentence: [1] + bpe_model.EncodeAsIds(sentence) + [2])

            data = data[data[0].map(len) <= self.seq_length]

            # encode each key word separately
            if self.attention_model:
                data['key_words'] = data['key_words'].apply(lambda sentence: [bpe_model.EncodeAsIds(word) for word in sentence])

                data['key_words'] = data['key_words'].apply(lambda sentence: list(itertools.chain.from_iterable(sentence)))

        else:
            bpe_model = None

            data[0] = data[0].apply(lambda sentence: sentence.split())

            # remove sentences longer than sequence length minus 2 (place for eos tokens)
            data = data[data[0].map(len) < self.seq_length - 1]

        print("Size after removing utterances longer than seq_length: {}".format(len(data)))

        self.bpe_model_path = bpe_model_path

        # KAMIL words to lista slow posortowana od najczestszych
        if vocabulary is None:
            self.vocab, self.words = self.build_vocab(list(data[0]), vocab_size, bpe_model)
            self.vocab_size = len(self.words)

        else:
            self.vocab = vocabulary

        self.unk_token = self.vocab.get('<unk>')
        self.sos_token = self.vocab.get('<s>')
        self.eos_token = self.vocab.get('</s>')

        if self.bpe_model_path is None:
            data[0] = data[0].apply(
                lambda sentence: [self.sos_token] + [self.vocab.get(word, self.unk_token) for word in sentence] + [self.eos_token])
            if self.attention_model:
                data['key_words'] = data['key_words'].apply(
                    lambda sentence: [self.vocab[word] for word in sentence if word in self.vocab])

        # remove training sentences without any key_words or with more key_words than sequence length
        if self.attention_model:
            data = data[(data['key_words'].map(len) > 0) & (data['key_words'].map(len) <= self.seq_length)]

            print("Size after removing utterances with wrong number of key words: {}".format(len(data)))

        if unk_max_number >= 0 and self.unk_token is not None:
            data = data[data[0].map(self.count_unks) <= unk_max_number]

            print("Size after removing utterances not meeting unk_max_number criteria: {}".format(len(data)))

        if unk_max_count is not None and self.unk_token is not None:
            unk_sentences_indexes = data[data[0].map(self.count_unks) > 0].index[unk_max_count:]
            data = data[~data.index.isin(unk_sentences_indexes)]

            print("Size after removing utterances not meeting unk_max_count criteria: {}".format(len(data)))

        # sort data by length of training sentences so that batches contain sentences within similar range
        data = data.reindex(data[0].map(len).sort_values().index)

        with open(os.path.join(self.data_dir, 'input_cleaned.txt'), 'w') as f:
            data.to_csv(f, sep='\t', header=False, index=False)

    def next_batch(self):
        batch_index = str(self.batch_order[self.pointer])
        x = self.database[batch_index]['data']
        y = self.database[batch_index]['target']
        target_weights = self.database[batch_index]['target_weights']
        key_words = self.database[batch_index]['key_words'] if self.attention_model else None
        key_words_count = self.database[batch_index]['key_words_count'] if self.attention_model else None
        self.pointer += 1
        return x, y, target_weights, key_words, key_words_count

    def reset_batch_pointer(self):
        self.pointer = 0
        self.batch_order = np.random.permutation(self.num_batches)
