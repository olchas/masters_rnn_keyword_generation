from __future__ import print_function
import argparse
import os

import sentencepiece as spm
import tensorflow as tf
from six.moves import cPickle


from model import Model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to load stored checkpointed models from')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='optional path to checkpointed model')
    parser.add_argument('-n', type=int, default=50,
                        help='number of words to sample')
    parser.add_argument('--prime', type=str, default=' ',
                        help='prime text')
    parser.add_argument('--pick', type=int, default=1,
                        help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=4,
                        help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each time step, 1 to sample at each time step, 2 to sample on spaces')
    parser.add_argument('--count', '-c', type=int, default=1,
                        help='number of samples to print')
    parser.add_argument('--bpe_model_path', type=str, default=None,
                        help='path to bpe model used in training for processing of data')
    parser.add_argument('--processed_embeddings', type=str, default=None,
                       help='path to pickled numpy array with processed embeddings that was used for training (if it changed);'
                            'this embedding must match vocabulary of input data')
    parser.add_argument('--tokens', '-t', default=False, action='store_true',
                        help='use <s> and <\s> tokens to determine start and end of sequence')
    parser.add_argument('--state_initialization', default='average', choices=['average', 'zero'], type=str,
                       help='how the state of rnn should be initialized')
    parser.add_argument('--keywords', '-k', type=str, nargs='*', default=[],
                        help='list of space separated keywords to use for initial state')
    parser.add_argument('--quiet', '-q', default=False, action='store_true',
                        help='suppress printing the prime text (default false)')

    return parser.parse_args()


def main():
    args = parse_arguments()
    sample(args)


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    if args.processed_embeddings is not None:
        saved_args.processed_embeddings = args.processed_embeddings
    if args.bpe_model_path is not None:
        saved_args.bpe_model_path = args.bpe_model_path
    with open(saved_args.words_vocab_file, 'rb') as f:
        words, vocab = cPickle.load(f)

    model = Model(saved_args, True)

    if saved_args.bpe_model_path is not None:
        bpe_model = spm.SentencePieceProcessor()
        bpe_model.Load(saved_args.bpe_model_path)

        args.prime = " ".join(bpe_model.EncodeAsPieces(args.prime))
        keywords_bpe = []
        for keyword in args.keywords:
            keywords_bpe += bpe_model.EncodeAsPieces(keyword)
        args.keywords = keywords_bpe

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())

        if args.model_checkpoint is not None:
            model_checkpoint_path = args.model_checkpoint
        else:
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            model_checkpoint_path = ckpt.model_checkpoint_path

        saver.restore(sess, model_checkpoint_path)
        for _ in range(args.count):
            generated_utterance = model.sample(sess, words, vocab, args.n, args.prime, args.sample, args.pick,
                                               args.width, args.quiet, args.tokens, args.keywords,
                                               args.state_initialization)

            generated_utterance = bpe_model.DecodePieces(generated_utterance) if saved_args.bpe_model_path is not None \
                else " ".join(generated_utterance).replace('<s> ', '').replace(' </s>', '')

            print(generated_utterance)


if __name__ == '__main__':
    main()
