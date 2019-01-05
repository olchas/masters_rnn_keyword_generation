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
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
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
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells)

        # KAMIL placeholdery sluza do ladowania danych treningowych
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        # KAMIL to initial_state na poczatku treningu, a nie ten, co jest srednia slow kluczowych (chyba)
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        # KAMIL zmienne ktore sie zmieniaja ale nie sa parametrami sieci, wiec nie sa trenowalne
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        tf.summary.scalar("time_batch", self.batch_time)

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
        with tf.variable_scope('rnnlm'):
            # KAMIL te zmienne chyba trzeba zainicjowac wartosciami nauczonymi w celu uzyskania poprawnego syntaxu na 'zwyklych danych' (bez atencji)
            # KAMIL to chyba moze byc tylko ostatnia warstwa - output layer
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            variable_summaries(softmax_w)
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            variable_summaries(softmax_b)
            # KAMIL tutaj okreslenie embeddingu jako zmiennej sieci? rozmiar embeddingu taki sam jak rozmiar hidden layer sieci
            with tf.device("/cpu:0"):
                # KAMIL tu by trzeba zamienic zmienne na gotowy embedding z GloVe
                if args.processed_embeddings is not None:

                    with open(args.processed_embeddings, 'rb') as f:
                        pretrained_embedding = cPickle.load(f)

                    train_embedding = not args.dont_train_embeddings
                    embedding = tf.get_variable(name="embedding",
                                                shape=[args.vocab_size, args.rnn_size],
                                                initializer=tf.constant_initializer(pretrained_embedding),
                                                trainable=train_embedding)
                else:
                    embedding = tf.get_variable(name="embedding",
                                                shape=[args.vocab_size, args.rnn_size])
                # KAMIL powiazanie wejscia z embeddingiem
                # KAMIL funkcja zwraca wiersze z macierzy embedding odpowiadajace indeksom z input data
                # KAMIL chyba powinno wystarczy zastapienie embedding odpowiednia macierza, mozna tez zostawic zmienna, ale ja zainicjowac gotowymi embedingami
                inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), args.seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # KAMIL tutaj obliczenie wyjscia z sieci dla wejscia
        # KAMIL tu chyba trzeba dodac attention
        # KAMIL czy tu sie dzieje wybranie slowa wyjsciowego, aby je podac na wejscie w nastepnym kroku? (zamiast slowa z batch?)l
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)
        # KAMIL tu podstawowa operacja - dekodowanie, obliczenie wyjsc dla wejsc
        # KAMIL last_state - koncowy stan sieci po przejsciu aktualnego batcha danych
        # KAMIL tutaj definicja wlasciwej sieci
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        # KAMIL tutaj nalezy dodac atencje
        # KAMIL wykorzystanie wyjscia do propagacji wstecznej - obliczenie lossu (costu) przez porowananie z targetem
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        # KAMIL loss ktory jest liczony na podstawie prawdopodobienstw slow, ktore maja wypadac nastepne w sekwencji
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        # KAMIL sredni blad na jedno slowo
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        tf.summary.scalar("cost", self.cost)
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        # KAMIL okreslenie gradientow na podstawie funkcji kosztow i modyfikowalnych zmiennych
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        # KAMIL wybranie sposobu modyfikacji zmiennych treningowych przy uwzglednieniu gradientow lrate i strategii (Adam)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, vocab, num=50, prime='first all', sampling_type=1, pick=0, width=4, quiet=False, bpe_model_path=None, tokens=False):
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        def beam_search_predict(sample, state):
            """Returns the updated probability distribution (`probs`) and
            `state` for a given `sample`. `sample` should be a sequence of
            vocabulary labels, with the last word to be tested against the RNN.
            """

            x = np.zeros((1, 1))
            x[0, 0] = sample[-1]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, final_state] = sess.run([self.probs, self.final_state],
                                            feed)
            return probs, final_state

        def beam_search_pick(prime, width, tokens=False, bpe_model_path=None):
            """Returns the beam search pick."""
            if not len(prime) or prime == ' ':
                prime = random.choice(list(vocab.keys()))
            prime_labels = [vocab.get(word, 0) for word in prime.split()]
            # KAMIL inicjalizacja beam search stanem zerowym, funkcja do predykcji i labelami prime wordow

            bs = BeamSearch(beam_search_predict,
                            sess.run(self.cell.zero_state(1, tf.float32)),
                            prime_labels)
            eos = vocab.get('</s>', 0) if bpe_model_path is not None else vocab.get('<\s>', 0) if tokens else None
            samples, scores = bs.search(None, eos, k=width, maxsample=num)
            # zwrocenie najlepszej sekwencji
            return samples[np.argmin(scores)]

        ret = []
        # prime = clean_str(prime)

        if bpe_model_path is not None:
            bpe_model = spm.SentencePieceProcessor()
            bpe_model.Load(bpe_model_path)
            prime = " ".join(bpe_model.EncodeAsPieces(prime))
        if tokens and not prime.startswith('<s>'):
            prime = '<s> ' + prime
        if not quiet:
            print(prime)
        if pick == 1:
            # KAMIL tutaj nalezy ustawic inital state na srednia keywordow
            state = sess.run(self.cell.zero_state(1, tf.float32))
            if not len(prime) or prime == ' ':
                prime = random.choice(list(vocab.keys()))
            for word in prime.split()[:-1]:
                if not quiet:
                    print(word)
                x = np.zeros((1, 1))
                # KAMIL tu sa feedowane primewordy i zmieniany jest w ten sposob tylko stan
                x[0, 0] = vocab.get(word, 0)
                feed = {self.input_data: x, self.initial_state:state}
                [state] = sess.run([self.final_state], feed)

            ret = prime.split()
            word = prime.split()[-1]
            # KAMIL tutaj sa feedowane kolejne slowa zwracane przez siec
            # KAMIL jakies zakonczenie po wygenerowaniu kropki konczacej zdanie
            for n in range(num):
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, 0)
                feed = {self.input_data: x, self.initial_state:state}
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
                if tokens and (word == '<\s>' or word == '</s>'):
                    break
                ret += [word]
        elif pick == 2:
            pred = beam_search_pick(prime, width, tokens, bpe_model_path)
            for i, label in enumerate(pred):
                ret += [words[label] if i > 0 else words[label]]
        print (ret)
        ret = bpe_model.DecodePieces(ret) if bpe_model_path is not None else " ".join(ret).strip('<s> ').strip(' <\s')
        return ret
