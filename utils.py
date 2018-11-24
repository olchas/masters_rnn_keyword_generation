# -*- coding: utf-8 -*-

import codecs
import collections
import itertools
import os
import re
from functools import reduce

import numpy as np
import sentencepiece as spm
from six.moves import cPickle


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
    """
    # KAMIL usuwanie znakow poza literami lacinskimi, koreanskimi, japonskimi (?), znakami ,?!'`()
    # KAMIL dodanie spacji przed apostrofy, przecinki, etc
    # KAMIL usuniecie wielokrotnych spacji
    string = re.sub(r"[^A-Za-z0-9(),!?\'\\<>`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\'t", " \'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, use_bpe, bpe_size, bpe_model_path, pretrained_embeddings=None, encoding=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        # Let's not read voca and data from file. We many change them.
        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file, use_bpe, bpe_size, bpe_model_path, pretrained_embeddings, encoding)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences, bpe_model, pretrained_embeddings):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        if bpe_model is not None:
            vocabulary_inv = [bpe_model.IdToPiece(x) for x in range(bpe_model.GetPieceSize())]
        else:
            # Build vocabulary
            word_counts = collections.Counter(sentences)
            # Mapping from index to word
            vocabulary_inv = [x[0] for x in word_counts.most_common()]
            #vocabulary_inv = [x for x in vocabulary_inv if vocabulary_inv]
            # KAMIL jaki sens ma linia ponizej: po co to sortowac, jak powyzej posortowano wg czestosci wystepowania
            #vocabulary_inv = list(sorted(vocabulary_inv))
        if pretrained_embeddings is not None:
            embedding_dict = {}
            with open(pretrained_embeddings) as f:
                for line in f:
                    items = line.split()
                    word = items[0]
                    # skip word containing non ascii characters
                    if not any(ord(letter) > 127 for letter in word):
                        embedding_dict[word] = items[1:]
            embedding_size = len(next(iter(embedding_dict.values())))

            vocabulary_embedding = []
            for word in vocabulary_inv:
                word_embedding = embedding_dict.get(word.lower().strip('▁'))
                if word_embedding is None:
                    word_embedding = np.random.rand(embedding_size).tolist()
                    print('word missing embedding: ' + word)
                vocabulary_embedding.append(word_embedding)

            vocabulary_embedding = np.array(vocabulary_embedding)
            embedding_dir = os.path.dirname(pretrained_embeddings)
            embedding_file = os.path.splitext(os.path.basename(pretrained_embeddings))[0]
            embedding_array_file = embedding_file + '_processed.pkl'
            with open(os.path.join(embedding_dir, embedding_array_file), 'wb') as f:
                cPickle.dump(vocabulary_embedding, f)

        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file, tensor_file, use_bpe, bpe_size, bpe_model_path, pretrained_embeddings, encoding):
        with codecs.open(input_file, "r", encoding=encoding) as f:
            data = f.read()

        # Optional text cleaning or make them lower case, etc.
        # data = clean_str(data)
        # KAMIL tu usuwaja podzial na zdania

        if use_bpe:
            bpe_model = spm.SentencePieceProcessor()
            if bpe_model_path is None:
                spm.SentencePieceTrainer.Train(
                    '--input=' + input_file + ' --model_prefix=bpe --vocab_size=' + str(bpe_size) + ' --model_type=bpe')

                bpe_model.Load("bpe.model")
            else:
                bpe_model.Load(bpe_model_path)
            x_text = None
            # list of encoded sentences with added start and end os sequence tokens
            sentences = list(map(lambda x: [1] + bpe_model.EncodeAsIds(x) + [2], data.split('\n')))
        else:
            bpe_model = None
            x_text = ' '.join(list(map(lambda x: '<s> ' + x + ' <\s>', data.split('\n')))).split()
        # KAMIL words to lista slow posortowana od najczestszych
        self.vocab, self.words = self.build_vocab(x_text, bpe_model, pretrained_embeddings)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)
        #
        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        # KAMIL zamiana danych w postaci listy slow na liste indexow
        self.tensor = np.array(reduce(lambda x, y: x + y, sentences)) if use_bpe else np.array(list(map(self.vocab.get, x_text)))
        # Save the data to data.npy
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        # KAMIL jeden batch to pewna liczba (batch_size) sekwencji o dlugosci seq_length
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        # KAMIL ograniczenie danych do wielokrotnosci batcha
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        # KAMIL target data to po prostu to samo, co input data -> zamienic na x: keywordy, y: zdanie?
        ydata = np.copy(self.tensor)
        # KAMIL przesuniecie pierwszego slowa na koniec w targecie: po co? -> zeby targetem bylo slow nastepne po input
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0] # mozna zmienic na pierwsze slowo po self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        # KAMIL podzielenie danych na num_batches kawalkow
        # KAMIL reshape zmienia array na macierz batch_size wierszy
        # KAMIL split powoduje, ze otrzymujesz liste num_batches arrayow o wymiarach batch_size x seq_length
        # KAMIL czyli kazdy batch to batch_size sekwencji seq_length wyrazow (nie kolejnych sekwencji)
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
