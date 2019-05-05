from __future__ import print_function
import argparse
import os
import time
from six.moves import cPickle

import numpy as np
import pandas as pd
import tensorflow as tf

from model import Model
from utils import TextLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    parser.add_argument('--validation_data_dir', type=str, default=None,
                       help='directory with validation data')
    parser.add_argument('--input_encoding', type=str, default=None,
                       help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--use_attention', default=False, action='store_true',
                       help='whether to use an attention model'
                            'if other than None, input_tagged.txt file must be present in data directories')
    parser.add_argument('--pos_tags', type=str, nargs='+',
                        default=['_JJ', '_NN', '_NNP', '_NNPS', '_NNS', '_RB', '_VB', '_VBD', '_VBG', '_VBN', '_VBP', '_VBZ'],
                        help='list of pos tags that should be used as keywords')
    parser.add_argument('--rnn_size', type=int, default=200,
                       help='size of RNN hidden state')
    parser.add_argument('--embedding_size', type=int, default=None,
                       help='optional size of embedding; if None, rnn_size will be used;'
                            'if you specify pretrained embeddings this argument is ignored')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=25,
                       help='RNN sequence length')
    parser.add_argument('--vocab_size', type=int, default=None,
                       help='max size of vocabulary of most common words (non bpe models only); words outside of limit will be turned to <unk> tokens; None for all words')
    parser.add_argument('--unk_max_number', type=int, default=1,
                       help='max number of <unk> tokens that can appear in a single traning sequence')
    parser.add_argument('--unk_max_count', type=int, default=100,
                       help='max total number of sentences containing <unk> tokens that can appear in training data')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1,
                       help='save frequency in epochs')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='initial learning rate')
    parser.add_argument('--adaptive_learning_rate', type=int, default=0,
                       help='maximal number of epochs which did not improve the best validation cost after which learning_rate should be decreased by decay_rate;'
                            '0 if learning rate should be decreased after each epoch;'
                            'requires validation data to work')
    parser.add_argument('--dropout_prob', type=float, default=0,
                       help='dropout probability that will be added to output of each recurrent layer; 0 for not dropout')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--gpu_mem', type=float, default=0.666,
                       help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    parser.add_argument('--use_bpe', default=False, action='store_true',
                       help='true if you want to use bpe on train data')
    parser.add_argument('--bpe_size',type=int, default=32000,
                       help='size of bpe vocabulary, valid only when use_bpe is true')
    parser.add_argument('--bpe_model_path', type=str, default=None,
                       help='path to pretrained bpe model, valid only when use_bpe is true')
    parser.add_argument('--pretrained_embeddings', type=str, default=None,
                       help='path to txt file with pretrained embeddings that should be matched with vocabulary and pickled;'
                            'this will overwrite embedding provided by processed_embeddings argument')
    parser.add_argument('--processed_embeddings', type=str, default=None,
                       help='path to pickled numpy array with processed embeddings that should be used as initialization;'
                            'this embedding must match vocabulary of input data')
    parser.add_argument('--dont_train_embeddings', action='store_true', default=False,
                       help='if you do not want to train embeddings')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.vocab_size, args.unk_max_number, args.unk_max_count, None, args.use_bpe, args.bpe_size, args.bpe_model_path,
                             args.pretrained_embeddings, args.use_attention, args.pos_tags, args.input_encoding)
    args.vocab_size = data_loader.vocab_size

    if args.pretrained_embeddings is not None:
        embedding_dir = os.path.dirname(args.pretrained_embeddings)
        embedding_file = os.path.splitext(os.path.basename(args.pretrained_embeddings))[0]
        embedding_array_file = embedding_file + '_processed.pkl'
        args.processed_embeddings = os.path.join(embedding_dir, embedding_array_file)

    if args.validation_data_dir is not None:
        val_data_loader = TextLoader(args.validation_data_dir, args.batch_size, args.seq_length, args.vocab_size, args.unk_max_number, args.unk_max_count, data_loader.vocab, args.use_bpe, args.bpe_size, data_loader.bpe_model_path,
                                     None, args.use_attention, args.pos_tags, args.input_encoding)
        validation_log = open(os.path.join(args.log_dir, 'validation_log.txt'), 'a')

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from), " %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, "config.pkl")), "config.pkl file does not exist in path %s" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, "words_vocab.pkl")), "words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme] == vars(args)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = cPickle.load(f)

        assert saved_words == data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab == data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)

    # KAMIL model sieci
    model = Model(args)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.log_dir)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')

    training_log = open(os.path.join(args.log_dir, 'training_log.txt'), 'a')

    best_val_error = None
    start_epoch = 0
    learning_rate = args.learning_rate

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        # KAMIL cos od zapisywania modeli
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

            learning_rate = model.lr.eval()

            best_val_error = model.best_val_error.eval() or None

            best_val_epoch = model.best_val_epoch.eval()

            start_epoch = model.epoch_pointer.eval() + 1

        else:
            sess.run(tf.assign(model.lr, learning_rate))

        for e in range(start_epoch, args.num_epochs):
            epoch_start = time.time()
            # KAMIL decay rate powoduje zmniejszanie sie learning rate w miare postepow
            if (args.validation_data_dir is None or args.adaptive_learning_rate <= 0) and e > 0:
                learning_rate *= args.decay_rate
                sess.run(tf.assign(model.lr, learning_rate))
            # KAMIL data loader laduje dane
            # KAMIL w kazdej epoce caly zbior danych jest analizowany?
            data_loader.reset_batch_pointer()
            # KAMIL czy to sprawia, ze po kazdej epoce jest zerowy stan?
            zero_state = sess.run(model.initial_state)

            state = zero_state

            epoch_error = 0

            # KAMIL po kazdej epoce zapisanie jej numeru w modelu
            sess.run(tf.assign(model.epoch_pointer, e))

            # KAMIL dla kazdego batcha, czym sa x i y, czyli dane wejsciowe i target? czy u mnie to beda keywordy i zdania?
            for b in range(data_loader.pointer, data_loader.num_batches):
                batch_start = time.time()
                x, y, target_weights, key_words, key_words_count = data_loader.next_batch()

                target_sequence_length = np.sum(target_weights, axis=0)

                if key_words is not None:
                    feed = {model.input_data: x, model.targets: y, model.target_weights: target_weights,
                            model.target_sequence_length: target_sequence_length, model.initial_state: state,
                            model.attention_key_words: key_words, model.attention_states_count: key_words_count}
                else:
                    feed = {model.input_data: x, model.targets: y, model.target_weights: target_weights,
                            model.target_sequence_length: target_sequence_length, model.initial_state: state}
                # KAMIL sess.run chyba sie podawalo parametry do zmiany i dane do treningu - feed
                # KAMIL model.train_op - optimizer do minimalizacji model.cost, model.final_state - stan sieci po tym batchu danych, jest poczatkowym stanem w nastepnym kroku?
                # KAMIL po co w sumie zachowywac stan do nastepnego batcha? -> czemu nie zerowy stan?
                # KAMIL merged - zbior zmiennych sieci modyfikowalnych? takie wartosci jak max, min, mean macierzy W i wektora b
                # KAMIL sess.run zwraca koncowe postaci tych wartosci po zakonczeniu obliczen

                # KAMIL: czemu sluzy inc_batch_pointer_op? -> to jest po to, zeby w modelu przechowywac informacje o batchu przy wznawianiu
                # KAMIL: po co w feedzie jest ten batch_time? -> chyba tylko po to, zeby dodac to do merged/summary

                # x_transposed = x.transpose()
                # for i, row in enumerate(key_words):
                #     for keyword in row:
                #         for word in x_transposed[i]:
                #             if word == keyword:
                #                 break
                #         else:
                #             sentence = [data_loader.words[index] for index in x_transposed[i]]
                #             print("keyword {} missing in sentence {}".format(data_loader.words[keyword], sentence))

                summary, train_loss, state, _ = sess.run([merged, model.cost, model.final_state,
                                                          model.train_op], feed)

                # KAMIL train loss mozna akumulowac, zeby sobie zachowac loss po kazdej epoce
                epoch_error += train_loss
                if (e * data_loader.num_batches + b) % args.batch_size == 0:
                    # KAMIL zapis logu?
                    train_writer.add_summary(summary, e * data_loader.num_batches + b)
                    batch_speed = time.time() - batch_start
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.num_batches,
                        e, train_loss, batch_speed))

            epoch_speed = time.time() - epoch_start
            print("epoch\t{}\tepoch_loss\t{:.3f}\tepoch_time\t{:.3f}\tlearning_rate\t{:.3f}\n".format(
                e, epoch_error / data_loader.num_batches, epoch_speed, learning_rate))
            training_log.write("epoch\t{}\tepoch_loss\t{:.3f}\tepoch_time\t{:.3f}\tlearning_rate\t{:.3f}\n".format(
                e, epoch_error / data_loader.num_batches, epoch_speed, learning_rate))

            # KAMIL czestotliwosc zapisu jest okreslana przez liczbe epok razy liczbe batchy
            if e % args.save_every == 0 or e == args.num_epochs - 1:  # save for the last result

                # validate every saved model
                if args.validation_data_dir is not None:
                    val_start = time.time()

                    val_data_loader.reset_batch_pointer()

                    val_error = 0

                    for val_b in range(val_data_loader.pointer, val_data_loader.num_batches):

                        val_x, val_y, val_target_weights, val_key_words, val_key_words_count = val_data_loader.next_batch()

                        val_target_sequence_length = np.sum(val_target_weights, axis=0)

                        if val_key_words is not None:
                            val_feed = {model.input_data: val_x, model.targets: val_y,
                                        model.target_weights: val_target_weights,
                                        model.target_sequence_length: val_target_sequence_length,
                                        model.initial_state: zero_state,
                                        model.attention_key_words: val_key_words,
                                        model.attention_states_count: val_key_words_count}
                        else:
                            val_feed = {model.input_data: val_x, model.targets: val_y,
                                        model.target_weights: val_target_weights,
                                        model.target_sequence_length: val_target_sequence_length,
                                        model.initial_state: zero_state}

                        val_train_loss = sess.run([model.cost], val_feed)

                        val_error += val_train_loss[0]

                    val_speed = time.time() - val_start

                    mean_val_error = val_error / val_data_loader.num_batches

                    print("epoch\t{}\tvalidation_loss\t{:.3f}\tvalidation_time\t{:.3f}\n".format(
                        e, mean_val_error, val_speed))
                    validation_log.write("epoch\t{}\tvalidation_loss\t{:.3f}\tvalidation_time\t{:.3f}\n".format(
                        e, mean_val_error, val_speed))

                    if args.adaptive_learning_rate > 0:

                        if best_val_error is None or best_val_error > mean_val_error:
                            # save information about best validation error and epoch in model
                            best_val_error = mean_val_error
                            best_val_epoch = e

                            sess.run(tf.assign(model.best_val_error, best_val_error))
                            sess.run(tf.assign(model.best_val_epoch, best_val_epoch))

                        elif e - best_val_epoch >= args.adaptive_learning_rate:
                            learning_rate *= args.decay_rate
                            sess.run(tf.assign(model.lr, learning_rate))

                saver.save(sess, checkpoint_path, global_step=e)
                print("model saved to {}".format(checkpoint_path))

        train_writer.close()

    training_log.close()

    if args.validation_data_dir is not None:
        validation_log.close()


if __name__ == '__main__':
    main()
