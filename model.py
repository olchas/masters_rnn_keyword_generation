import os
import random
from six.moves import cPickle

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

from bahdanau_coverage_attention import BahdanauCoverageAttention, CustomAttentionWrapper
from beam import BeamSearch


class Model():
    def __init__(self, args, helper_type=None):
        self.args = args

        self.attention_seq_length = args.seq_length

        if helper_type is not None:
            args.batch_size = 1
            args.seq_length = 1

        cells = []
        for _ in range(args.num_layers):
            cell = rnn.BasicLSTMCell(args.rnn_size)
            if args.dropout_prob > 0:
               cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=max(0, 1 - args.dropout_prob))
            cells.append(cell)

        self.cell = rnn.MultiRNNCell(cells)
        self.rnn_size = args.rnn_size
        self.num_layers = args.num_layers

        self.input_data = tf.placeholder(tf.int32, [args.seq_length, args.batch_size])
        self.targets = tf.placeholder(tf.int32, [args.seq_length, args.batch_size])
        self.target_weights = tf.placeholder(tf.float32, [args.seq_length, args.batch_size])
        self.target_sequence_length = tf.placeholder(tf.int32, args.batch_size)

        self.attention_key_words = tf.placeholder(tf.int32, [args.batch_size, self.attention_seq_length])
        self.attention_states_count = tf.placeholder(tf.int32, args.batch_size)

        self.best_val_epoch = tf.Variable(0, name="best_val_epoch", trainable=False, dtype=tf.int32)
        self.best_val_error = tf.Variable(0.0, name="best_val_error", trainable=False, dtype=tf.float32)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False, dtype=tf.int32)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)

                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))

        with tf.variable_scope('rnnlm'):

            with tf.device("/cpu:0"):

                # initialize embedding matrix with preprocessed GloVe vectors
                if args.processed_embeddings is not None:

                    with open(args.processed_embeddings, 'rb') as f:
                        pretrained_embedding = cPickle.load(f)

                    self.embedding_size = pretrained_embedding.shape[1]

                    train_embedding = not args.dont_train_embeddings

                    self.embedding = tf.get_variable(name="embedding",
                                                     shape=[pretrained_embedding.shape[0], self.embedding_size],
                                                     initializer=tf.constant_initializer(pretrained_embedding),
                                                     trainable=train_embedding)

                else:
                    # if embedding_size was not given then use the size of LSTM cell
                    self.embedding_size = args.embedding_size if args.embedding_size is not None else self.rnn_size

                    self.embedding = tf.get_variable(name="embedding",
                                                     shape=[args.vocab_size, self.embedding_size])

                inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

            # if embedding_size is different from layer size then prepare a dense layer for rescaling
            if self.embedding_size != self.rnn_size:

                self.embedding_w = tf.get_variable("embedding_w", [self.embedding_size, self.rnn_size])
                variable_summaries(self.embedding_w)
                self.embedding_b = tf.get_variable("embedding_b", [self.rnn_size])
                variable_summaries(self.embedding_b)

                inputs = tf.add(tf.tensordot(inputs, self.embedding_w, axes=[[2], [0]]), self.embedding_b)

        self.attention_model = False

        if args.use_attention:

            attention_states = tf.nn.embedding_lookup(self.embedding, self.attention_key_words)

            if args.attention_type == 'bahdanau':
                attention_type = tf.contrib.seq2seq.BahdanauAttention
                attention_wrapper = tf.contrib.seq2seq.AttentionWrapper
            elif args.attention_type == 'bahdanau_coverage':
                attention_type = BahdanauCoverageAttention
                attention_wrapper = CustomAttentionWrapper
            else:
                attention_type = tf.contrib.seq2seq.LuongAttention
                attention_wrapper = tf.contrib.seq2seq.AttentionWrapper

            self.attention_model = True

            print('Using attention model')
            # rnn_size here is the size of dense layers used to calculate attention weights
            self.attention_mechanism = attention_type(
                self.rnn_size, attention_states,
                memory_sequence_length=self.attention_states_count)

            # attention_layer_size is the size of final layer that concatenates cell output and context vector
            self.cell = attention_wrapper(
                self.cell, self.attention_mechanism,
                attention_layer_size=self.rnn_size)

        self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)
        current_state = self.initial_state

        helper = tf.contrib.seq2seq.TrainingHelper(inputs, self.target_sequence_length, time_major=True)

        maximum_iterations = None

        # last layer to calculate scores for every word in a dictionary
        projection_layer = tf.layers.Dense(args.vocab_size, use_bias=True, name="output_projection")

        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.cell,
            helper,
            current_state,
            output_layer=projection_layer
        )

        # Dynamic decoding
        outputs, current_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            output_time_major=True,
            swap_memory=True,
            maximum_iterations=maximum_iterations,
            scope='rnnlm')

        self.final_state = current_state

        self.logits = outputs.rnn_output
        self.sample_id = outputs.sample_id

        # probabilities for every word to come next in the sequence
        self.probs = tf.nn.softmax(self.logits)

        # cut targets matrix according to longest target sequence
        labels = tf.slice(self.targets, [0, 0], [tf.keras.backend.max(self.target_sequence_length), args.batch_size])
        target_weigths_cut = tf.slice(self.target_weights, [0, 0], [tf.keras.backend.max(self.target_sequence_length), args.batch_size])

        # calculate cross entropy as the loss
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.logits)
        self.cost = tf.reduce_sum(crossent * target_weigths_cut) / args.batch_size

        tf.summary.scalar("cost", self.cost)

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        # calculate gradients and clip them
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)

        # Adam optimizer to apply gradients to trainable variables
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def embedding_lookup(self, words):
        if self.embedding_size == self.rnn_size:
            return tf.nn.embedding_lookup(self.embedding, words)
        else:
            return tf.matmul(tf.nn.embedding_lookup(self.embedding, words), self.embedding_w) + self.embedding_b

    def sample(self, sess, words, vocab, num=50, prime='', sampling_type=1, pick=0, width=4, quiet=False, tokens=False, keywords=[], state_initialization='average'):

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(t, np.random.rand(1)*s))

        def beam_search_predict(sample, state, keywords_ids=None, keywords_count=None):
            """Returns the updated probability distribution (`probs`) and
            `state` for a given `sample`. `sample` should be a sequence of
            vocabulary labels, with the last word to be tested against the RNN.
            """

            x = np.zeros((1, 1))
            x[0, 0] = sample[-1]
            if keywords_count is not None:
                feed = {self.input_data: x, self.initial_state: state, self.target_sequence_length: [1], self.attention_key_words: keywords_ids, self.attention_states_count: keywords_count}
            else:
                feed = {self.input_data: x, self.initial_state: state, self.target_sequence_length: [1]}
            [probs, final_state] = sess.run([self.probs, self.final_state],
                                            feed)
            return probs, final_state

        def beam_search_pick(prime_labels, width, initial_state, tokens=False, keywords_ids=None, keywords_count=None):
            """Returns the beam search pick."""

            bs = BeamSearch(beam_search_predict,
                            initial_state,
                            prime_labels,
                            keywords_ids,
                            keywords_count)
            eos = vocab.get('</s>', 0) if tokens else None
            oov = vocab.get('<unk>', None)
            samples, scores = bs.search(oov, eos, k=width, maxsample=num)
            # returning the best sequence
            return samples[np.argmin(scores)]

        ret = []
        keywords_ids = None
        keywords_count = None

        # take only up to seq_length keywords
        keywords = keywords[:self.attention_seq_length]

        unk_token = vocab.get('<unk>', 0)

        if keywords:

            # key words outside of vocabulary are not used
            keywords_ids = [vocab[keyword] for keyword in keywords if keyword in vocab]

            # if there are no keywords in dictionary, then assign the zero state
            if not keywords_ids or state_initialization == 'zero':
                initial_state = sess.run(self.cell.zero_state(1, tf.float32))

            # prepare initial state as average of keyword embeddings
            elif state_initialization == 'average':

                keywords_embedded = sess.run(self.embedding_lookup(keywords_ids))
                mean_embedding = np.mean(keywords_embedded, axis=0)
                mean_embedding = mean_embedding.reshape(1, mean_embedding.shape[0])

                if self.attention_model:
                    mean_embedding = tf.convert_to_tensor(mean_embedding, dtype=tf.float32)
                    lstm_cells_state = tuple(
                        [tf.nn.rnn_cell.LSTMStateTuple(mean_embedding, mean_embedding) for _ in range(self.num_layers)])

                    # TODO: decide whether state of attention cells should be initiated as well
                    initial_state = sess.run(self.cell.zero_state(1, tf.float32).clone(cell_state=lstm_cells_state))
                else:

                    initial_state = tuple(
                        [tf.nn.rnn_cell.LSTMStateTuple(mean_embedding, mean_embedding) for _ in range(self.num_layers)])

            if self.attention_model:

                keywords_ids = [keywords_ids + [unk_token] * (self.attention_seq_length - len(keywords_ids))]
                keywords_count = [len(keywords_ids)] if keywords_ids else [1]

        else:
            initial_state = sess.run(self.cell.zero_state(1, tf.float32))
            if self.attention_model:
                keywords_count = [1]
                keywords_ids = [[unk_token] * self.attention_seq_length]

        if tokens and not prime.startswith('<s>'):
            prime = '<s> ' + prime
        if not prime:
            prime = random.choice(list(vocab.keys()))
        if not quiet:
            print(prime)
        if pick == 1:

            state = initial_state
            for word in prime.split()[:-1]:
                if not quiet:
                    print(word)
                x = np.zeros((1, 1))
                # only the state matters while we are feeding the prime words
                x[0, 0] = vocab.get(word, unk_token)
                if self.attention_model:
                    feed = {self.input_data: x, self.initial_state: state, self.target_sequence_length: [1], self.attention_key_words: keywords_ids, self.attention_states_count: keywords_count}
                else:
                    feed = {self.input_data: x, self.initial_state: state, self.target_sequence_length: [1]}
                [state] = sess.run([self.final_state], feed)

            ret = prime.split()
            word = prime.split()[-1]
            # feed the word picked in the last iteration
            for n in range(num - len(prime)):
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, unk_token)
                if self.attention_model:
                    feed = {self.input_data: x, self.initial_state: state, self.target_sequence_length: [1], self.attention_key_words: keywords_ids, self.attention_states_count: keywords_count}
                else:
                    feed = {self.input_data: x, self.initial_state: state, self.target_sequence_length: [1]}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
                p = probs[0]

                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    if word == '\n':
                        sample = weighted_pick(p)
                    else:
                        sample = np.argmax(p)
                else: # sampling_type == 1 default:
                    sample = weighted_pick(p)

                word = words[sample]
                if tokens and word == '</s>':
                    break
                ret += [word]
        elif pick == 2:
            pred = beam_search_pick([vocab.get(word, unk_token) for word in prime.split()], width, initial_state, tokens, keywords_ids, keywords_count)
            for i, label in enumerate(pred):
                ret += [words[label] if i > 0 else words[label]]
        return ret
