from __future__ import print_function

import argparse
import os
import time
from six.moves import cPickle

import numpy as np
import tensorflow as tf

from model import Model
from utils import TextLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    parser.add_argument('--validation_data_dir', type=str, default=None,
                       help='directory with validation data')
    parser.add_argument('--load_preprocessed', action='store_true', default=False,
                       help='attempt to load ready data objects from provided directories')
    parser.add_argument('--input_encoding', type=str, default=None,
                       help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--use_attention', default=False, action='store_true',
                       help='whether to use an attention model'
                            'if other than None, input_tagged.txt file must be present in data directories')
    parser.add_argument('--without_coverage_loss', default=False, action='store_true',
                       help="whether to not include coverage loss in bahdanau_coverage model's loss calculation")
    parser.add_argument('--attention_type', default='luong', choices=['bahdanau', 'luong', 'bahdanau_coverage'], type=str,
                       help='type of attention to use')
    parser.add_argument('--pos_tags', type=str, nargs='+',
                        default=['_JJ', '_JJR', '_JJS', '_NN', '_NNP', '_NNPS', '_NNS', '_RB', '_RBR', '_RBS', '_VB', '_VBD', '_VBG', '_VBN', '_VBP', '_VBZ'],
                        help='list of pos tags that should be used as keywords')
    parser.add_argument('--key_word_count_multiplier', type=int, default=1,
                       help='if you want to multiply key words counts to make <unk> vectors appear in attention mechanism')
    parser.add_argument('--state_initialization', default='random', choices=['prev', 'random', 'zero', 'average'], type=str,
                       help='how the state of rnn should be initialized for each batch during training')
    parser.add_argument('--rnn_size', type=int, default=200,
                       help='size of RNN hidden state')
    parser.add_argument('--embedding_size', type=int, default=None,
                       help='optional size of embedding; if None, rnn_size will be used;'
                            'if you specify pretrained embeddings this argument is ignored')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
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
    parser.add_argument('--save_all', action='store_true', default=False,
                       help='True if you do want to save all models, not only the ones that are better than previous best')
    parser.add_argument('--max_worse_validations', type=int, default=5,
                       help='maximal number of validations that did not improve after which training should be finished early')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='initial learning rate')
    parser.add_argument('--adaptive_learning_rate', type=int, default=0,
                       help='maximal number of epochs which did not improve the best validation cost after which learning_rate should be decreased by decay_rate;'
                            '0 if learning rate should be decreased after each epoch;'
                            'requires validation data to work')
    parser.add_argument('--dropout_prob', type=float, default=0,
                       help='dropout probability that will be added to output of each recurrent layer; 0 for not dropout')
    parser.add_argument('--decay_rate', type=float, default=1.00,
                       help='decay rate for adam')
    parser.add_argument('--gpu_mem', type=float, default=0.666,
                       help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    parser.add_argument('--use_bpe', default=False, action='store_true',
                       help='true if you want to use bpe on train data')
    parser.add_argument('--bpe_size', type=int, default=32000,
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
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)


def train(args):

    provide_key_words = args.use_attention or args.state_initialization == 'average'

    if args.use_attention and args.state_initialization == 'prev' and args.attention_type == 'bahdanau_coverage':
        args.state_initialization = 'random'

    data_loader = TextLoader(args.load_preprocessed,
                             'training',
                             args.data_dir,
                             args.batch_size,
                             args.seq_length,
                             args.vocab_size,
                             args.unk_max_number,
                             args.unk_max_count,
                             None,
                             args.use_bpe,
                             args.bpe_size,
                             args.bpe_model_path,
                             args.pretrained_embeddings,
                             provide_key_words,
                             args.key_word_count_multiplier,
                             args.pos_tags,
                             args.input_encoding)

    args.vocab_size = data_loader.vocab_size
    args.words_vocab_file = data_loader.words_vocab_file
    args.bpe_model_path = data_loader.bpe_model_path

    if args.pretrained_embeddings is not None:
        args.processed_embeddings = os.path.join(data_loader.embedding_dir, 'embedding_matrix.pkl')

    if args.validation_data_dir is not None:
        val_data_loader = TextLoader(args.load_preprocessed,
                                     'validation',
                                     args.validation_data_dir,
                                     args.batch_size,
                                     args.seq_length,
                                     args.vocab_size,
                                     args.unk_max_number,
                                     args.unk_max_count,
                                     data_loader.vocab,
                                     args.use_bpe,
                                     args.bpe_size,
                                     data_loader.bpe_model_path,
                                     args.pretrained_embeddings,
                                     provide_key_words,
                                     args.key_word_count_multiplier,
                                     args.pos_tags,
                                     args.input_encoding)

        validation_log = open(os.path.join(args.log_dir, 'validation_log.txt'), 'a')

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from), " %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, "config.pkl")), "config.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["rnn_size", "embedding_size", "num_layers", "dropout_prob", "batch_size", "seq_length", "attention_type", "use_attention", "dont_train_embeddings"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme] == vars(args)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        assert os.path.isfile(saved_model_args.words_vocab_file), "words_vocab.pkl.pkl file does not exist in path %s" % saved_model_args.words_vocab_file

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(saved_model_args.words_vocab_file, 'rb') as f:
            saved_words, saved_vocab = cPickle.load(f)

        assert saved_words == data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab == data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)

    with open(data_loader.words_vocab_file, 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)

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

        saver = tf.train.Saver(tf.global_variables())

        zero_state = sess.run(model.initial_state)
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
            # decrease learning rate after every epoch if adaptive_learning_rate is not used
            if (args.validation_data_dir is None or args.adaptive_learning_rate <= 0) and e > 0:
                learning_rate *= args.decay_rate
                sess.run(tf.assign(model.lr, learning_rate))

            data_loader.reset_batch_pointer()

            state = zero_state

            epoch_error = 0
            epoch_coverage_loss = 0

            # as every epoch is started, save its number in the model
            sess.run(tf.assign(model.epoch_pointer, e))

            for b in range(data_loader.pointer, data_loader.num_batches):

                x, y, target_weights, target_sequence_length, key_words, key_words_count, key_words_weights = data_loader.next_batch()

                if args.state_initialization == 'zero':
                    state = zero_state

                if key_words is not None:
                    feed = {model.input_data: x, model.targets: y, model.target_weights: target_weights,
                            model.target_sequence_length: target_sequence_length, model.initial_state: state,
                            model.attention_key_words: key_words, model.attention_states_count: key_words_count, model.attention_states_weights: key_words_weights}
                else:
                    feed = {model.input_data: x, model.targets: y, model.target_weights: target_weights,
                            model.target_sequence_length: target_sequence_length, model.initial_state: state}

                summary, train_loss, state, _ = sess.run([merged, model.cost, model.final_state,
                                                          model.train_op], feed)

                # if model trained has bahdanau_coverage attention type, collect coverage_loss as well
                if args.use_attention and args.attention_type == 'bahdanau_coverage':
                    epoch_coverage_loss += np.sum(state.coverage_loss)

                # accumulate the train_loss
                epoch_error += train_loss

                if (e * data_loader.num_batches + b) % args.batch_size == 0:
                    train_writer.add_summary(summary, e * data_loader.num_batches + b)

            epoch_speed = time.time() - epoch_start
            if args.use_attention and args.attention_type == 'bahdanau_coverage':
                print("epoch\t{}\tepoch_loss\t{:.3f}\tepoch_coverage_loss\t{:.3f}\tepoch_time\t{:.3f}\tlearning_rate\t{:.3f}\n".format(
                    e, epoch_error / data_loader.num_batches, epoch_coverage_loss / args.batch_size / data_loader.num_batches, epoch_speed, learning_rate))
                training_log.write("epoch\t{}\tepoch_loss\t{:.3f}\tepoch_coverage_loss\t{:.3f}\tepoch_time\t{:.3f}\tlearning_rate\t{:.3f}\n".format(
                    e, epoch_error / data_loader.num_batches, epoch_coverage_loss / args.batch_size / data_loader.num_batches, epoch_speed, learning_rate))
            else:
                print("epoch\t{}\tepoch_loss\t{:.3f}\tepoch_time\t{:.3f}\tlearning_rate\t{:.3f}\n".format(
                    e, epoch_error / data_loader.num_batches, epoch_speed, learning_rate))
                training_log.write("epoch\t{}\tepoch_loss\t{:.3f}\tepoch_time\t{:.3f}\tlearning_rate\t{:.3f}\n".format(
                    e, epoch_error / data_loader.num_batches, epoch_speed, learning_rate))

            if e % args.save_every == 0 or e == args.num_epochs - 1:  # save for the last result

                # validate every saved model
                if args.validation_data_dir is not None:
                    val_start = time.time()

                    val_data_loader.reset_batch_pointer()

                    val_error = 0

                    val_coverage_loss = 0

                    val_state = zero_state

                    for val_b in range(val_data_loader.pointer, val_data_loader.num_batches):

                        if args.state_initialization == 'zero':
                            val_state = zero_state

                        val_x, val_y, val_target_weights, val_target_sequence_length, val_key_words, val_key_words_count, val_key_words_weights = val_data_loader.next_batch()

                        if val_key_words is not None:
                            val_feed = {model.input_data: val_x, model.targets: val_y,
                                        model.target_weights: val_target_weights,
                                        model.target_sequence_length: val_target_sequence_length,
                                        model.initial_state: val_state,
                                        model.attention_key_words: val_key_words,
                                        model.attention_states_count: val_key_words_count,
                                        model.attention_states_weights: val_key_words_weights}
                        else:
                            val_feed = {model.input_data: val_x, model.targets: val_y,
                                        model.target_weights: val_target_weights,
                                        model.target_sequence_length: val_target_sequence_length,
                                        model.initial_state: val_state}

                        val_train_loss, val_state = sess.run([model.cost, model.final_state], val_feed)

                        val_error += val_train_loss
                        # if model trained has bahdanau_coverage attention type, collect coverage_loss as well
                        if args.use_attention and args.attention_type == 'bahdanau_coverage':
                            val_coverage_loss += np.sum(val_state.coverage_loss)

                    mean_val_error = val_error / val_data_loader.num_batches

                    val_speed = time.time() - val_start

                    if args.use_attention and args.attention_type == 'bahdanau_coverage':
                        print("epoch\t{}\tvalidation_loss\t{:.3f}\tval_coverage_loss\t{:.3f}\tvalidation_time\t{:.3f}\n".format(
                            e, mean_val_error, val_coverage_loss / args.batch_size / val_data_loader.num_batches, val_speed))
                        validation_log.write("epoch\t{}\tvalidation_loss\t{:.3f}\tval_coverage_loss\t{:.3f}\tvalidation_time\t{:.3f}\n".format(
                            e, mean_val_error, val_coverage_loss / args.batch_size / val_data_loader.num_batches, val_speed))
                    else:
                        print("epoch\t{}\tvalidation_loss\t{:.3f}\tvalidation_time\t{:.3f}\n".format(
                            e, mean_val_error, val_speed))
                        validation_log.write("epoch\t{}\tvalidation_loss\t{:.3f}\tvalidation_time\t{:.3f}\n".format(
                            e, mean_val_error, val_speed))

                    # save information about best validation error and epoch in model
                    if best_val_error is None or best_val_error > mean_val_error:

                        print('======= NEW BEST EPOCH =======')
                        best_val_error = mean_val_error
                        best_val_epoch = e

                        sess.run(tf.assign(model.best_val_error, best_val_error))
                        sess.run(tf.assign(model.best_val_epoch, best_val_epoch))

                    # if adaptive learning rate is used and enough epochs have passed without improvement then decrease learning rate
                    elif e - best_val_epoch >= args.adaptive_learning_rate and args.adaptive_learning_rate > 0:
                        learning_rate *= args.decay_rate
                        sess.run(tf.assign(model.lr, learning_rate))

                    if args.save_all or best_val_epoch == e:
                        saver.save(sess, checkpoint_path, global_step=e)
                        print("model saved to {}".format(checkpoint_path))

                    if e - best_val_epoch >= args.max_worse_validations and args.max_worse_validations > 0:
                        print("finishing early as {} evaluated models did not lower the validation loss".format(args.max_worse_validations))
                        break

                else:
                    saver.save(sess, checkpoint_path, global_step=e)
                    print("model saved to {}".format(checkpoint_path))

    training_log.close()

    if args.validation_data_dir is not None:
        validation_log.close()
        val_data_loader.close()

    train_writer.close()
    data_loader.close()

if __name__ == '__main__':
    main()
