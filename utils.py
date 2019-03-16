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
    string = re.sub(r"[^A-Za-z0-9:,!?\'/]", " ", string)
    string = re.sub(r"\?{2,}", " \? ", string)
    string = re.sub(r"!{2,}", " ! ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\'t", " \'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, vocab_size, use_bpe, bpe_size, bpe_model_path, pretrained_embeddings=None, encoding=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")
        target_tensor_file = os.path.join(data_dir, "target_data.npy")
        weights_file = os.path.join(data_dir, "weights.npy")

        # Let's not read voca and data from file. We many change them.
        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file, target_tensor_file, weights_file, vocab_size, use_bpe, bpe_size, bpe_model_path, pretrained_embeddings, encoding)
        else:
            print("loading preprocessed files")
        self.create_batches()
        self.reset_batch_pointer()

    def prepare_vocabulary(self, word_counts, vocab_size):
        vocabulary_inv = ['<unk>', '<s>', '<\s>'] if vocab_size is not None and vocab_size < len(word_counts.items()) + 2 else ['<s>', '<\s>']
        counts = sorted(list(set(word_counts.values())), reverse=True)
        for count in counts:
            words = sorted([x[0] for x in word_counts.items() if x[1] == count])
            vocabulary_inv += words
        return vocabulary_inv[:vocab_size] if vocab_size is not None else vocabulary_inv

    def build_vocab(self, sentences, vocab_size, bpe_model, pretrained_embeddings):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        if bpe_model is not None:
            vocabulary_inv = [bpe_model.IdToPiece(x) for x in range(bpe_model.GetPieceSize())]
        else:
            # Build vocabulary

            # list of words
            x_text = [word for sentence in sentences for word in sentence]

            word_counts = collections.Counter(x_text)

            # Mapping from index to word
            vocabulary_inv = self.prepare_vocabulary(word_counts, vocab_size)
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

    def preprocess(self, input_file, vocab_file, tensor_file, target_tensor_file, weights_file, vocab_size, use_bpe, bpe_size, bpe_model_path, pretrained_embeddings, encoding):
        with codecs.open(input_file, "r", encoding=encoding) as f:
            data = f.read()

        sentences = data.strip().split('\n')

        # Optional text cleaning or make them lower case, etc.
        sentences = [clean_str(sentence) for sentence in sentences]
        # Remove sentences with less than three words
        sentences = [sentence for sentence in sentences if len(sentence.split()) > 3]

        if use_bpe:

            bpe_model = spm.SentencePieceProcessor()
            if bpe_model_path is None:

                # save the file with reduced number of sentences in order to train bpe model from it
                input_dir = os.path.dirname(input_file)
                input_file_processed = os.path.splitext(os.path.basename(input_file))[0] + '_processed.txt'
                with open(os.path.join(input_dir, input_file_processed), 'w') as f:
                    f.write('\n'.join(sentences) + '\n')

                spm.SentencePieceTrainer.Train(
                    '--input=' + os.path.join(input_dir, input_file_processed) + ' --model_prefix=bpe --vocab_size=' + str(bpe_size) + ' --model_type=bpe')

                bpe_model.Load("bpe.model")
            else:
                bpe_model.Load(bpe_model_path)
            # list of encoded sentences with added start and end os sequence tokens
            sentences = [[1] + bpe_model.EncodeAsIds(sentence) + [2] for sentence in sentences]
            # remove sentences longer than sequence length
            sentences = [sentence for sentence in sentences if len(sentence) <= self.seq_length]

        else:
            bpe_model = None

            # remove sentences longer than sequence length minus 2 (place for eos tokens)
            sentences = [sentence.split()
                         for sentence in sentences if len(sentence.split()) < self.seq_length - 1]

        # KAMIL words to lista slow posortowana od najczestszych
        self.vocab, self.words = self.build_vocab(sentences, vocab_size, bpe_model, pretrained_embeddings)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        if not use_bpe:
            unk_token_id = self.vocab.get('<unk>')
            sentences = [[self.vocab.get(word, unk_token_id) for word in sentence] for sentence in sentences]

        self.target_weights = np.array([[1] * len(sentence) + [0] * (self.seq_length - len(sentence))
                                        for sentence in sentences])

        self.target_tensor = [sentence[1:] + [sentence[0]] for sentence in sentences]

        # zero pad sentences to sequence length
        self.tensor = np.array([sentence + [0] * (self.seq_length - len(sentence)) for sentence in sentences])

        self.target_tensor = np.array([sentence + [0] * (self.seq_length - len(sentence)) for sentence in self.target_tensor])

        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        # KAMIL zamiana danych w postaci listy slow na liste indexow
        # self.tensor = np.array(reduce(lambda x, y: x + y, sentences))
        # self.target_weights = np.array(reduce(lambda x, y: x + y, self.target_weights))
        # Save the data to data.npy
        np.save(tensor_file, self.tensor)
        np.save(weights_file, self.target_weights)
        np.save(target_tensor_file, self.target_tensor)

    def create_batches(self):
        # KAMIL jeden batch to pewna liczba (batch_size) sekwencji o dlugosci seq_length
        self.num_batches = int(len(self.tensor) / self.batch_size)
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        # KAMIL ograniczenie danych do wielokrotnosci batcha
        self.tensor = self.tensor[:self.num_batches * self.batch_size]
        self.target_weights = self.target_weights[:self.num_batches * self.batch_size]
        self.target_tensor = self.target_tensor[:self.num_batches * self.batch_size]

        # KAMIL podzielenie danych na num_batches kawalkow
        # KAMIL reshape zmienia array na macierz batch_size wierszy
        # KAMIL split powoduje, ze otrzymujesz liste num_batches arrayow o wymiarach batch_size x seq_length
        # KAMIL czyli kazdy batch to batch_size sekwencji seq_length wyrazow (nie kolejnych sekwencji)
        self.x_batches = np.split(self.tensor.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(self.target_tensor.reshape(self.batch_size, -1), self.num_batches, 1)
        self.target_weights_batches = np.split(self.target_weights.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y, target_weights = self.x_batches[self.pointer], self.y_batches[self.pointer].transpose(), self.target_weights_batches[self.pointer].transpose()
        self.pointer += 1
        return x, y, target_weights

    def reset_batch_pointer(self):
        self.pointer = 0
