# -*- coding: utf-8 -*-

import codecs
import collections
import itertools
import os
import re
from functools import reduce

import numpy as np
import pandas as pd
import sentencepiece as spm
from six.moves import cPickle


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
    """
    string = string.lower()
    string = re.sub(r"[^a-z0-9\-,!?\'/]", " ", string)
    # remove duplicates of any punctuation, separate certain punctuation from words
    string = re.sub(r"\?+", " ? ", string)
    string = re.sub(r",+", " , ", string)
    string = re.sub(r"!+", " ! ", string)
    string = re.sub(r"-+", "-", string)
    string = re.sub(r"'+", "'", string)
    string = re.sub(r"/+", " / ", string)
    # remove any dashes at the beginning of string or not preceded by a letter
    string = re.sub(r"([^\w]|^)-", r"\1", string)
    # remove any dashes at the end of string or not preceding a letter
    string = re.sub(r"-([^\w]|$)", r"\1", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\'t", " \'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    # remove space duplicates
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, vocab_size, use_bpe, bpe_size, bpe_model_path, pretrained_embeddings=None, key_words_file=None, pos_tags=[], encoding=None):
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
            sentences = self.preprocess(input_file, vocab_file, vocab_size, use_bpe, bpe_size, bpe_model_path, pretrained_embeddings, key_words_file, pos_tags, encoding)
            self.prepare_tensors(sentences, tensor_file, target_tensor_file, weights_file, use_bpe)

        else:
            print("loading preprocessed files")
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences, vocab_size, bpe_model, add_eos_tokens=True):
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
            if add_eos_tokens:
                vocabulary_inv = ['<unk>', '<s>', '</s>'] \
                    if vocab_size is not None and vocab_size < len(word_counts.items()) + 2 else ['<s>', '</s>']
            else:
                vocabulary_inv = ['<unk>']

            counts = sorted(list(set(word_counts.values())), reverse=True)
            for count in counts:
                words = sorted([x[0] for x in word_counts.items() if x[1] == count])
                vocabulary_inv += words

            if vocab_size is not None:
                vocabulary_inv = vocabulary_inv[:vocab_size]

        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def prepare_embedding_matrix(self, pretrained_embeddings):

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
        for word in self.words:
            word_embedding = embedding_dict.get(word.lower().strip('▁'))
            if word_embedding is None:
                word_embedding = np.random.rand(embedding_size).tolist()
                print('word missing embedding: ' + word)
            vocabulary_embedding.append(word_embedding)

        print('Embedding dict loaded')

        vocabulary_embedding = np.array(vocabulary_embedding)
        embedding_dir = os.path.dirname(pretrained_embeddings)
        embedding_file = os.path.splitext(os.path.basename(pretrained_embeddings))[0]
        embedding_array_file = embedding_file + '_processed.pkl'
        with open(os.path.join(embedding_dir, embedding_array_file), 'wb') as f:
            cPickle.dump(vocabulary_embedding, f)

        print('Embedding matrix saved')

        # with open(os.path.join(embedding_dir, 'embedding_dict.pkl'), 'wb') as f:
        #     cPickle.dump(embedding_dict, f)

    def prepare_keywords_embedding_matrix(self, data, key_words_file, pos_tags):

        # with open(os.path.join(embedding_dir, 'embedding_dict.pkl'), 'rb') as f:
        #     embedding_dict = cPickle.load(f)

        key_words_dir = os.path.dirname(key_words_file)
        key_words_filename = os.path.splitext(os.path.basename(key_words_file))[0] + '_matrix.npy'

        # replace tagged sentences with list of key words from allowed pos tags
        data['key_words'] = data['key_words'].apply(lambda sentence: [word.split('_')[0]
                                                                      for word in sentence.strip().split()
                                                                      if word.endswith(tuple(pos_tags))])

        print('Key words extracted')

        # data['key_words'] = data['key_words'].apply(
        #     lambda key_words_group: [embedding_dict.get(key_word) for key_word in key_words_group if key_word in embedding_dict])

        # leave only key words for which there are embeddings
        # data['key_words'] = data['key_words'].apply(
        #     lambda key_words_group: [key_word for key_word in key_words_group if key_word in embedding_dict])

        # use the same dictionary as for training data
        data['key_words'] = data['key_words'].apply(
            lambda key_words_group: [self.vocab[key_word] for key_word in key_words_group if key_word in self.vocab])

        print('Key words without embeddings removed')

        # remove training sentences without any key_words
        data = data[data['key_words'].map(len) > 0]

        # prepare vocabulary of key words up to the same size as the size of data vocabulary
        # self.key_words_vocab, self.key_words_vocab_inv = self.build_vocab(list(data['key_words']), self.vocab_size,
        #                                                                   None, False)
        #
        # print('Key words vocabulary prepared')
        #
        # self.key_words_vocab_size = len(self.key_words_vocab_inv)
        #
        # # leave only the key words from the vocabulary
        # data['key_words'] = data['key_words'].apply(lambda key_words_group: [self.key_words_vocab[key_word]
        #                                                                      for key_word in key_words_group
        #                                                                      if key_word in self.key_words_vocab])
        #
        # print('Key words outside vocabulary removed')
        #
        # # remove training sentences without any key_words
        # data = data[data['key_words'].map(len) > 0]
        #
        # print('Sentences without key words removed')

        self.key_words_count = np.array([len(key_words_group) for key_words_group in data['key_words']])

        print('Key words counted')

        # self.key_words = np.array([key_words_group + [[0] * embedding_size] * (self.seq_length - len(key_words_group))
        #                  for key_words_group in list(data['key_words'])])

        # fill the arrays to sequence length
        self.key_words = np.array([key_words_group + [0] * (self.seq_length - len(key_words_group))
                                   for key_words_group in list(data['key_words'])])

        print('Key words matrix prepared')

        # prepare embedding matrix for key words
        # key_words_embedding = [[0] * embedding_size]
        # for key_word in self.key_words_vocab_inv[1:]:
        #     key_words_embedding.append(embedding_dict.get(key_word))
        #
        # key_words_embedding = np.array(key_words_embedding)
        # with open(os.path.join(embedding_dir, 'key_words_embedding.pkl'), 'wb') as f:
        #     cPickle.dump(key_words_embedding, f)
        #
        # print('Key words embedding matrix saved')
        #
        # with open(os.path.join(self.data_dir, 'key_words_vocab.pkl'), 'wb') as f:
        #     cPickle.dump(self.key_words_vocab_inv, f)

        np.save(os.path.join(key_words_dir, key_words_filename), self.key_words)
        np.save(os.path.join(key_words_dir, 'key_words_count.npy'), self.key_words_count)

        # sort data by length of training sentences so that batches contain sentences within similar range
        data = data.reindex(data[0].map(len).sort_values().index)

        return list(data[0])

    def preprocess(self, input_file, vocab_file, vocab_size, use_bpe, bpe_size, bpe_model_path, pretrained_embeddings, key_words_file, pos_tags, encoding):

        data = pd.read_csv(input_file, sep='\t', header=None, encoding=encoding)

        # Optional text cleaning or make them lower case, etc.
        data[0] = data[0].apply(clean_str)
        # Remove sentences with less than three words
        data = data[data[0].apply(lambda sentence: sentence.split()).map(len) > 3]

        # save the file with reduced number of sentences in order to train bpe model from it and extract pos tags
        input_dir = os.path.dirname(input_file)
        with open(os.path.join(input_dir, 'input_cleaned.txt'), 'w') as f:
            f.write('\n'.join(list(data[0])) + '\n')

        # TODO: add a function to extract pos tags first
        if key_words_file is not None and pretrained_embeddings is not None:
            data['key_words'] = list(pd.read_csv(key_words_file, sep='\t', header=None, encoding=encoding)[0])

        if use_bpe:

            bpe_model = spm.SentencePieceProcessor()
            if bpe_model_path is None:

                spm.SentencePieceTrainer.Train(
                    '--input=' + os.path.join(input_dir, 'input_cleaned.txt') + ' --model_prefix=bpe --vocab_size=' + str(bpe_size) + ' --model_type=bpe')

                bpe_model.Load("bpe.model")
            else:
                bpe_model.Load(bpe_model_path)

            # list of encoded sentences with added start and end os sequence tokens
            data[0] = data[0].apply(lambda sentence: [1] + bpe_model.EncodeAsIds(sentence) + [2])

            data = data[data[0].map(len) <= self.seq_length]

        else:
            bpe_model = None

            data[0] = data[0].apply(lambda sentence: sentence.split())

            # remove sentences longer than sequence length minus 2 (place for eos tokens)
            data = data[data[0].map(len) < self.seq_length - 1]

        # KAMIL words to lista slow posortowana od najczestszych
        self.vocab, self.words = self.build_vocab(list(data[0]), vocab_size, bpe_model)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        self.key_words = None
        self.key_words_count = None
        self.key_words_batches = None
        self.key_words_count_batches = None
        self.key_words_vocab_size = None

        if pretrained_embeddings is not None:
            self.prepare_embedding_matrix(pretrained_embeddings)

            if 'key_words' in data:
                return self.prepare_keywords_embedding_matrix(data, key_words_file, pos_tags)

            else:
                # sort data by length of training sentences so that batches contain sentences within similar range
                data = data.reindex(data[0].map(len).sort_values().index)

        else:

            data = data.reindex(data[0].map(len).sort_values().index)

        return list(data[0])

    def prepare_tensors(self, sentences, tensor_file, target_tensor_file, weights_file, use_bpe):

        if not use_bpe:
            unk_token_id = self.vocab.get('<unk>')
            sentences = [[1] + [self.vocab.get(word, unk_token_id) for word in sentence] + [2] for sentence in
                         sentences]

        self.target_weights = np.array([[1] * len(sentence) + [0] * (self.seq_length - len(sentence))
                                        for sentence in sentences])

        self.target_tensor = [sentence[1:] + [sentence[0]] for sentence in sentences]

        # zero pad sentences to sequence length
        self.tensor = np.array([sentence + [0] * (self.seq_length - len(sentence)) for sentence in sentences])

        self.target_tensor = np.array(
            [sentence + [0] * (self.seq_length - len(sentence)) for sentence in self.target_tensor])

        # The same operation like this [self.vocab[word] for word in x_text]
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
        # self.x_batches = np.split(self.tensor, self.num_batches)
        self.x_batches = [x_batch.transpose() for x_batch in np.split(self.tensor, self.num_batches)]
        self.y_batches = [y_batch.transpose() for y_batch in np.split(self.target_tensor, self.num_batches)]
        self.target_weights_batches = [target_weights_batch.transpose() for target_weights_batch in np.split(self.target_weights, self.num_batches)]

        if self.key_words is not None:
            self.key_words = self.key_words[:self.num_batches * self.batch_size]
            self.key_words_count = self.key_words_count[:self.num_batches * self.batch_size]
            self.key_words_batches = np.split(self.key_words, self.num_batches)
            self.key_words_count_batches = np.split(self.key_words_count, self.num_batches)

    def next_batch(self):
        batch_index = self.batch_order[self.pointer]
        x, y, target_weights = self.x_batches[batch_index], self.y_batches[batch_index], self.target_weights_batches[batch_index]
        key_words = self.key_words_batches[batch_index] if self.key_words_batches is not None else None
        key_words_count = self.key_words_count_batches[batch_index] if self.key_words_count_batches is not None else None
        self.pointer += 1
        return x, y, target_weights, key_words, key_words_count

    def reset_batch_pointer(self):
        self.pointer = 0
        self.batch_order = np.random.permutation(self.num_batches)
