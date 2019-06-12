import os
import random
from six.moves import cPickle

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

from beam import BeamSearch
from utils import clean_str


class Model():
    def __init__(self, args, helper_type=None):
        self.args = args

        self.attention_seq_length = args.seq_length

        if helper_type is not None:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            # KAMIL rozmiar hidden layer
            cell = cell_fn(args.rnn_size)
            if args.dropout_prob > 0:
               cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=max(0, 1 - args.dropout_prob))
            cells.append(cell)

        self.cell = rnn.MultiRNNCell(cells)
        self.rnn_size = args.rnn_size
        self.num_layers = args.num_layers

        # KAMIL placeholdery sluza do ladowania danych treningowych
        # KAMIL potrzebny placeholder na slowa kluczowe do atencji
        # self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.input_data = tf.placeholder(tf.int32, [args.seq_length, args.batch_size])
        # self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.seq_length, args.batch_size])
        # self.target_weights = tf.placeholder(tf.float32, [args.batch_size, args.seq_length])
        self.target_weights = tf.placeholder(tf.float32, [args.seq_length, args.batch_size])
        self.target_sequence_length = tf.placeholder(tf.int32, args.batch_size)
        self.attention_key_words = tf.placeholder(tf.int32, [args.batch_size, self.attention_seq_length])
        self.attention_states_count = tf.placeholder(tf.int32, args.batch_size)
        # KAMIL zmienne ktore sie zmieniaja ale nie sa parametrami sieci, wiec nie sa trenowalne
        self.best_val_epoch = tf.Variable(0, name="best_val_epoch", trainable=False, dtype=tf.int32)
        self.best_val_error = tf.Variable(0.0, name="best_val_error", trainable=False, dtype=tf.float32)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False, dtype=tf.int32)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                #with tf.name_scope('stddev'):
                #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                #tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                #tf.summary.histogram('histogram', var)

        # KAMIL tutaj okreslenie zmiennych sieci - macierzy W i wektora b
        # KAMIL liczba wierszy - rozmiar warstwy, liczba kolumn - wagi dla kazdego slowa ze slownika
        # KAMIl tu należy zdefiniowac atencje w trakcie treningu
        with tf.variable_scope('rnnlm'):
            # KAMIL te zmienne chyba trzeba zainicjowac wartosciami nauczonymi w celu uzyskania poprawnego syntaxu na 'zwyklych danych' (bez atencji)
            # KAMIL to chyba moze byc tylko ostatnia warstwa - output layer
            # softmax_w = tf.get_variable("softmax_w", [self.rnn_size, args.vocab_size])
            # variable_summaries(softmax_w)
            # softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            # variable_summaries(softmax_b)

            # KAMIL tutaj okreslenie embeddingu jako zmiennej sieci? rozmiar embeddingu taki sam jak rozmiar hidden layer sieci
            with tf.device("/cpu:0"):
                # KAMIL tu by trzeba zamienic zmienne na gotowy embedding z GloVe
                if args.processed_embeddings is not None:

                    with open(args.processed_embeddings, 'rb') as f:
                        pretrained_embedding = cPickle.load(f)

                    self.embedding_size = pretrained_embedding.shape[1]

                    train_embedding = not args.dont_train_embeddings
                    self.embedding = tf.get_variable(name="embedding",
                                                     shape=[pretrained_embedding.shape[0], self.embedding_size],
                                                     initializer=tf.constant_initializer(pretrained_embedding),
                                                     trainable=train_embedding)

                    if args.use_attention:

                        attention_states = tf.nn.embedding_lookup(self.embedding, self.attention_key_words)

                else:
                    self.embedding_size = args.embedding_size if args.embedding_size is not None else self.rnn_size

                    self.embedding = tf.get_variable(name="embedding",
                                                     shape=[args.vocab_size, self.embedding_size])

                # KAMIL powiazanie wejscia z embeddingiem
                # KAMIL funkcja zwraca wiersze z macierzy embedding odpowiadajace indeksom z input data
                # KAMIL chyba powinno wystarczy zastapienie embedding odpowiednia macierza, mozna tez zostawic zmienna, ale ja zainicjowac gotowymi embedingami
                # inputs = tf.split(tf.nn.embedding_lookup(self.embedding, self.input_data), args.seq_length, 1)
                inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
                # KAMIL to przeraba na liste kolejnych wartosci z sekwencji w batchu (pierwsze wartosci, potem drugie itd)
                # KAMIL to ponizej usuwa niepotrzebny wymiar 1, zostawiajac liste tensorow o wymiarach: rozmiar batcha, rozmiar embeddingu
                # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

            # if embedding_size is different from layer size then prepare a dense layer for recalculation
            if self.embedding_size != self.rnn_size:

                self.embedding_w = tf.get_variable("embedding_w", [self.embedding_size, self.rnn_size])
                variable_summaries(self.embedding_w)
                self.embedding_b = tf.get_variable("embedding_b", [self.rnn_size])
                variable_summaries(self.embedding_b)

                inputs = tf.add(tf.tensordot(inputs, self.embedding_w, axes=[[2], [0]]), self.embedding_b)

        # self.attention_states = tf.placeholder(tf.float32, [args.batch_size, self.attention_seq_length, self.embedding_size])

        # KAMIL tutaj obliczenie wyjscia z sieci dla wejscia
        # KAMIL tu chyba trzeba dodac attention - w trakcie testow
        # KAMIL czy tu sie dzieje wybranie slowa wyjsciowego, aby je podac na wejscie w nastepnym kroku? (zamiast slowa z batch?) -> tak na potrzeby testu
        # KAMIL do treningu z atencja chyba potrzebna by byla nowa petla, ktora by miala dostep do stanu 
        # KAMIL do atencji jest nam potrzebny zero padding na sekwencjach w batchu
        # def test_loop(prev, _):
        #     prev = tf.matmul(prev, softmax_w) + softmax_b
        #     prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
        #     return tf.nn.embedding_lookup(self.embedding, prev_symbol)

        # def train_loop(prev, i):
        #     state =
        #     prev = tf.matmul(prev, softmax_w) + softmax_b
        #     prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
        #     return inputs[i + 1]
        # KAMIL tu podstawowa operacja - dekodowanie, obliczenie wyjsc dla wejsc
        # KAMIL last_state - koncowy stan sieci po przejsciu aktualnego batcha danych
        # KAMIL przejscie tylko przez komorki LSTM


        # outputs = []
        # for input in inputs:
        #     current_output, current_state = legacy_seq2seq.rnn_decoder([input], current_state, cell, loop_function=test_loop if infer else None, scope='rnnlm')
        #     outputs.append(current_output)

        # helper for the decoder
        #helper = tf.contrib.seq2seq.TrainingHelper(inputs, self.target_sequence_length, time_major=True)

        # if helper_type is not None:
        #     # testing
        #     if helper_type == 'greedy':
        #         helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding, start_tokens=tf.fill([args.batch_size], 1), end_token=2)
        #     maximum_iterations = args.seq_length
        # else:
        #     # training
        #     helper = tf.contrib.seq2seq.TrainingHelper(inputs, [args.seq_length] * args.batch_size, time_major=True)
        #
        #     maximum_iterations = None

        self.attention_model = False

        if args.use_attention and args.processed_embeddings is not None:

            self.attention_model = True

            print('Using attention model')
            # rnn_size here is the size of dense layers used to calculate attention weights
            self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.rnn_size, attention_states,
                memory_sequence_length=self.attention_states_count)

            # attention_layer_size is the size of final layer that concatenates cell output and context vector
            self.cell = tf.contrib.seq2seq.AttentionWrapper(
                self.cell, self.attention_mechanism,
                attention_layer_size=self.rnn_size)

        self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)
        current_state = self.initial_state

        # training
        # helper = tf.contrib.seq2seq.TrainingHelper(inputs, [args.seq_length] * args.batch_size, time_major=True)

        helper = tf.contrib.seq2seq.TrainingHelper(inputs, self.target_sequence_length, time_major=True)

        maximum_iterations = None

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

        #output = tf.reshape(outputs.rnn_output, [-1, self.rnn_size])

        #print(output)

        self.final_state = current_state

        # KAMIL tutaj wyliczenie prawdopodobienstw z ostanitej warstwy pelnych polaczen - softmax
        # self.logits = tf.matmul(outputs.rnn_output, softmax_w) + softmax_b
        self.logits = outputs.rnn_output
        self.sample_id = outputs.sample_id
        self.probs = tf.nn.softmax(self.logits)
        # KAMIL loss ktory jest liczony na podstawie prawdopodobienstw slow, ktore maja wypadac nastepne w sekwencji
        # loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
        #         [tf.reshape(self.targets, [-1])],
        #         #[tf.ones([args.batch_size * args.seq_length])],
        #         [tf.reshape(self.target_weights, [-1])],
        #         args.vocab_size)

        # cut targets matrix according to longest target sequence

        labels = tf.slice(self.targets, [0, 0], [tf.keras.backend.max(self.target_sequence_length), args.batch_size])

        target_weigths_cut = tf.slice(self.target_weights, [0, 0], [tf.keras.backend.max(self.target_sequence_length), args.batch_size])

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.logits)
        self.cost = tf.reduce_sum(crossent * target_weigths_cut) / args.batch_size

        # KAMIL sredni blad na jedno slowo
        # self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=self.targets, logits=self.logits)
        # self.cost = tf.reduce_sum(crossent * self.target_weights) / args.batch_size

        tf.summary.scalar("cost", self.cost)

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        # KAMIL okreslenie gradientow na podstawie funkcji kosztow i modyfikowalnych zmiennych
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        # KAMIL wybranie sposobu modyfikacji zmiennych treningowych przy uwzglednieniu gradientow lrate i strategii (Adam)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def embedding_lookup(self, words):
        if self.embedding_size == self.rnn_size:
            return tf.nn.embedding_lookup(self.embedding, words)
        else:
            return tf.matmul(tf.nn.embedding_lookup(self.embedding, words), self.embedding_w) + self.embedding_b

    def sample(self, sess, words, vocab, num=50, prime='', sampling_type=1, pick=0, width=4, quiet=False, bpe_model_path=None, tokens=False, keywords=[]):
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

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
            # KAMIL inicjalizacja beam search stanem zerowym, funkcja do predykcji i labelami prime wordow

            bs = BeamSearch(beam_search_predict,
                            initial_state,
                            prime_labels,
                            keywords_ids,
                            keywords_count)
            eos = vocab.get('</s>', 0) if tokens else None
            oov = vocab.get('<unk>', None)
            samples, scores = bs.search(oov, eos, k=width, maxsample=num)
            # zwrocenie najlepszej sekwencji
            return samples[np.argmin(scores)]

        ret = []
        prime = clean_str(prime)
        keywords_ids = None
        keywords_count = None

        if bpe_model_path is not None:
            bpe_model = spm.SentencePieceProcessor()
            bpe_model.Load(bpe_model_path)
            prime = " ".join(bpe_model.EncodeAsPieces(prime))
            keywords_bpe = []
            for keyword in keywords:
                keywords_bpe += bpe_model.EncodeAsPieces(keyword)
            keywords = keywords_bpe

        # take only up to seq_length keywords
        keywords = keywords[:self.attention_seq_length]

        unk_token = vocab.get('<unk>', 0)

        # prepare initial state as mean of keyword embeddings
        if keywords:
            # TODO: what to do with key words outside of dictionary
            keywords_ids = [vocab.get(keyword, unk_token) for keyword in keywords]
            keywords_embedded = sess.run(self.embedding_lookup(keywords_ids))
            mean_embedding = np.mean(keywords_embedded, axis=0)
            mean_embedding = mean_embedding.reshape(1, mean_embedding.shape[0])

            if self.attention_model:
                mean_embedding = tf.convert_to_tensor(mean_embedding, dtype=tf.float32)
                lstm_cells_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(mean_embedding, mean_embedding) for _ in range(self.num_layers)])
                keywords_ids = [keywords_ids + [0] * (self.attention_seq_length - len(keywords))]
                keywords_count = [len(keywords)]
                # keywords_embedded = np.expand_dims(np.concatenate((keywords_embedded, [[0] * self.embedding_size] * (self.attention_seq_length - len(keywords)))), axis=0)
                # initial_state = sess.run(self.cell.zero_state(1, tf.float32))

                # TODO: decide whether state of attention cells should be initiated as well
                initial_state = sess.run(self.cell.zero_state(1, tf.float32).clone(cell_state=lstm_cells_state))
            else:

                initial_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(mean_embedding, mean_embedding) for _ in range(self.num_layers)])
        else:
            initial_state = sess.run(self.cell.zero_state(1, tf.float32))
            if self.attention_model:
                keywords_count = [1]
                keywords_ids = [[0] * self.attention_seq_length]
        if tokens and not prime.startswith('<s>'):
            prime = '<s> ' + prime
        if not prime:
            prime = random.choice(list(vocab.keys()))
        if not quiet:
            print(prime)
        if pick == 1:
            # KAMIL tutaj nalezy ustawic inital state na srednia keywordow
            state = initial_state
            for word in prime.split()[:-1]:
                if not quiet:
                    print(word)
                x = np.zeros((1, 1))
                # KAMIL tu sa feedowane primewordy i zmieniany jest w ten sposob tylko stan
                x[0, 0] = vocab.get(word, unk_token)
                if self.attention_model:
                    feed = {self.input_data: x, self.initial_state: state, self.target_sequence_length: [1], self.attention_key_words: keywords_ids, self.attention_states_count: keywords_count}
                else:
                    feed = {self.input_data: x, self.initial_state: state, self.target_sequence_length: [1]}
                [state] = sess.run([self.final_state], feed)

            ret = prime.split()
            word = prime.split()[-1]
            # KAMIL tutaj sa feedowane kolejne slowa zwracane przez siec
            # KAMIL jakies zakonczenie po wygenerowaniu kropki konczacej zdanie
            for n in range(num):
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, unk_token)
                if self.attention_model:
                    feed = {self.input_data: x, self.initial_state: state, self.target_sequence_length: [1], self.attention_key_words: keywords_ids, self.attention_states_count: keywords_count}
                else:
                    feed = {self.input_data: x, self.initial_state: state, self.target_sequence_length: [1]}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
                p = probs[0]
                # KAMIL rozne sposoby wyboru nastepnego slowa
                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    if word == '\n':
                        sample = weighted_pick(p)
                    else:
                        sample = np.argmax(p)
                else: # sampling_type == 1 default:
                    sample = weighted_pick(p)
                # KAMIL sample to pewnie indeks
                word = words[sample]
                if tokens and word == '</s>':
                    break
                ret += [word]
        elif pick == 2:
            pred = beam_search_pick([vocab.get(word, unk_token) for word in prime.split()], width, initial_state, tokens, keywords_ids, keywords_count)
            for i, label in enumerate(pred):
                ret += [words[label] if i > 0 else words[label]]
        ret = bpe_model.DecodePieces(ret) if bpe_model_path is not None else " ".join(ret).replace('<s> ', '').replace(' </s>', '')
        print(ret)
        return ret
