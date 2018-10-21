# -*- coding: utf-8 -*-
import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        # Let's not read voca and data from file. We many change them.
        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file, encoding)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        # KAMIL usuwanie znakow poza literami lacinskimi, koreanskimi, japonskimi (?), znakami ,?!'`()
        # KAMIL dodanie spacji przed apostrofy, przecinki, etc
        # KAMIL usuniecie wielokrotnych spacji
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = collections.Counter(sentences)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file, tensor_file, encoding):
        with codecs.open(input_file, "r", encoding=encoding) as f:
            data = f.read()

        # Optional text cleaning or make them lower case, etc.
        #data = self.clean_str(data)
        # KAMIL tu chyba usuwaja podzial na zdania
        x_text = data.split()
        # KAMIL words to lista slow posortowana od najczestszych
        self.vocab, self.words = self.build_vocab(x_text)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        # KAMIL zamiana danych w postaci listy slow na liste indexow
        self.tensor = np.array(list(map(self.vocab.get, x_text)))
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
        # KAMIL przesuniecie pierwszego slowa na koniec w targecie: po co?
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
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
