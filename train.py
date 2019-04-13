from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    parser.add_argument('--input_encoding', type=str, default=None,
                       help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--key_words_file', type=str, default=None,
                       help='file with key words extracted from input.txt to be used in attention mechanism;'
                            'if other than None, pretrained_embeddings must be provided as well')
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
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
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
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.vocab_size, args.use_bpe, args.bpe_size, args.bpe_model_path,
                             args.pretrained_embeddings, args.key_words_file, args.pos_tags, args.input_encoding)
    args.vocab_size = data_loader.vocab_size

    if args.pretrained_embeddings is not None:
        embedding_dir = os.path.dirname(args.pretrained_embeddings)
        embedding_file = os.path.splitext(os.path.basename(args.pretrained_embeddings))[0]
        embedding_array_file = embedding_file + '_processed.pkl'
        args.processed_embeddings = os.path.join(embedding_dir, embedding_array_file)

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = cPickle.load(f)

        assert saved_words==data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)

    # KAMIL model sieci
    model = Model(args)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.log_dir)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        # KAMIL cos od zapisywania modeli
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(model.epoch_pointer.eval(), args.num_epochs):
            # KAMIL decay rate powoduje zmniejszanie sie learning rate w miare postepow
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            # KAMIL data loader laduje dane
            # KAMIL w kazdej epoce caly zbior danych jest analizowany?
            data_loader.reset_batch_pointer()
            # KAMIL czy to sprawia, ze po kazdej epoce jest zerowy stan?
            state = sess.run(model.initial_state)
            speed = 0
            epoch_error = 0
            if args.init_from is None:
                assign_op = model.epoch_pointer.assign(e)
                sess.run(assign_op)
            else:
                # KAMIL czy mozna kontynuowac tylko na tych samych danych?
                data_loader.pointer = model.batch_pointer.eval()
                args.init_from = None
            # KAMIL dla kazdego batcha, czym sa x i y, czyli dane wejsciowe i target? czy u mnie to beda keywordy i zdania?
            for b in range(data_loader.pointer, data_loader.num_batches):
                start = time.time()
                x, y, target_weights, key_words, key_words_count = data_loader.next_batch()

                target_sequence_length = np.sum(target_weights, axis=0)

                if key_words is not None:
                    feed = {model.input_data: x, model.targets: y, model.target_weights: target_weights, model.target_sequence_length: target_sequence_length, model.initial_state: state,
                            model.batch_time: speed, model.attention_key_words: key_words, model.attention_states_count: key_words_count, model.target_sequence_length: target_sequence_length}
                else:
                    feed = {model.input_data: x, model.targets: y, model.target_weights: target_weights, model.target_sequence_length: target_sequence_length, model.initial_state: state,
                            model.batch_time: speed, model.target_sequence_length: target_sequence_length}
                # KAMIL sess.run chyba sie podawalo parametry do zmiany i dane do treningu - feed
                # KAMIL model.train_op - optimizer do minimalizacji model.cost, model.final_state - stan sieci po tym batchu danych, jest poczatkowym stanem w nastepnym kroku?
                # KAMIL po co w sumie zachowywac stan do nastepnego batcha?
                # KAMIL merged - zbior zmiennych sieci modyfikowalnych? takie wartosci jak max, min, mean macierzy W i wektora b
                # KAMIL sess.run zwraca koncowe postaci tych wartosci po zakonczeniu obliczen
                summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
                                                             model.train_op, model.inc_batch_pointer_op], feed)
                # KAMIL train loss mozna akumulowac, zeby sobie zachowac loss po kazdej epoce
                epoch_error += train_loss
                # KAMIL zapis logu?
                train_writer.add_summary(summary, e * data_loader.num_batches + b)
                speed = time.time() - start
                if (e * data_loader.num_batches + b) % args.batch_size == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.num_batches + b,
                                args.num_epochs * data_loader.num_batches,
                                e, train_loss, speed))
                # KAMIL czestotliwosc zapisu jest okreslana przez liczbe epok razy liczbe batchy

                if (e * data_loader.num_batches + b) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
            print("mean epoch error is {}".format(epoch_error / data_loader.num_batches))
        train_writer.close()


if __name__ == '__main__':
    main()
