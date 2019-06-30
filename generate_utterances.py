from __future__ import print_function
import argparse
import os

import sentencepiece as spm
import tensorflow as tf
from six.moves import cPickle

from model import Model
from utils import stop_words


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, default='save',
                        help='model directory to load stored checkpointed models from')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='optional path to checkpointed model')
    parser.add_argument('--test_file', '-t', type=str,
                        help='path do file with test sentences')
    parser.add_argument('--output_dir', '-o', type=str, default='evaluation',
                        help='model directory to store evaluation results')
    parser.add_argument('--keywords_file', '-k', type=str, required=True,
                        help='path to file with pos tagged test file')
    parser.add_argument('--dont_provide_keywords', default=False, action='store_true',
                        help='set to true if you do not want to provide keywords to model')
    parser.add_argument('--prime', '-p', type=int, default=3,
                        help='number of prime words to use from test_file')
    parser.add_argument('--pick', type=int, default=1,
                        help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', '-w', type=int, default=4,
                        help='width of the beam search')
    parser.add_argument('--sample', '-s', type=int, default=1,
                        help='0 to use max at each time step, 1 to sample at each time step, 2 to sample on spaces')
    parser.add_argument('--bpe_model_path', '-b', type=str, default=None,
                        help='path to bpe model used in training for processing of data')
    parser.add_argument('--processed_embeddings', type=str, default=None,
                        help='path to pickled numpy array with processed embeddings that were used for training (if the path changed);'
                             'this embedding must match vocabulary of input data')
    parser.add_argument('--tokens', default=False, action='store_true',
                        help='use <s> and </s> tokens to determine start and end of sequence')
    parser.add_argument('--state_initialization', default='average', choices=['average', 'zero'], type=str,
                       help='how the state of rnn should be initialized')
    parser.add_argument('--quiet', '-q', default=False, action='store_true',
                        help='suppress printing the prime text (default false)')

    return parser.parse_args()


def main():
    args = parse_arguments()
    evaluate(args)


def evaluate(args):
    with open(os.path.join(args.input_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    if args.processed_embeddings is not None:
        saved_args.processed_embeddings = args.processed_embeddings
    if args.bpe_model_path is not None:
        saved_args.bpe_model_path = args.bpe_model_path
    with open(saved_args.words_vocab_file, 'rb') as f:
        words, vocab = cPickle.load(f)

    if saved_args.bpe_model_path is not None:
        bpe_model = spm.SentencePieceProcessor()
        bpe_model.Load(saved_args.bpe_model_path)

    with open(args.test_file) as f:
        test_utterances = f.read().strip().split('\n')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.keywords_file) as f:
        tagged_utterances = f.read().strip().split('\n')

    assert len(tagged_utterances) == len(test_utterances), "Size of input data and tagged data does not match"

    keywords = [[word.split('_')[0]
                 for word in tagged_utterance.split()[args.prime:]
                 if word.endswith(tuple(saved_args.pos_tags))
                 and word.split('_')[0] not in stop_words]
                for tagged_utterance in tagged_utterances]

    # for bpe model assume all keywords are known
    if saved_args.bpe_model_path is not None:
        known_keywords = keywords
    else:
        known_keywords = [[keyword for keyword in keyword_group if keyword in vocab]
                          for keyword_group in keywords]

    header = ['index', 'expected', 'generated', 'all_keywords', 'known_keywords', 'prime', 'with_unk']
    generated_data = [header]
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        if args.model_checkpoint is not None:
            model_checkpoint_path = args.model_checkpoint
        else:
            ckpt = tf.train.get_checkpoint_state(args.input_dir)
            model_checkpoint_path = ckpt.model_checkpoint_path

        saver.restore(sess, model_checkpoint_path)
        for i, test_utterance in enumerate(test_utterances):
            prime = ' '.join(test_utterance.split()[:args.prime])

            utterance_keywords = known_keywords[i] if not args.dont_provide_keywords else []

            # allow to produce up to 2 times more words than expected
            if saved_args.bpe_model_path is not None:
                prime = " ".join(bpe_model.EncodeAsPieces(prime))
                keywords_bpe = []
                for keyword in utterance_keywords:
                    keywords_bpe += bpe_model.EncodeAsPieces(keyword)
                utterance_keywords = keywords_bpe

                maximal_length = len(bpe_model.EncodeAsPieces(test_utterance)) * 2

                # for bpe model assume all words are known
                with_unk = False
            else:
                maximal_length = len(test_utterance.split()) * 2

                with_unk = any([word for word in test_utterance.split() if word not in vocab])

            generated_utterance = model.sample(sess, words, vocab, maximal_length, prime, args.sample,
                                               args.pick, args.width, args.quiet,
                                               args.tokens, utterance_keywords, args.state_initialization)

            generated_utterance = bpe_model.DecodePieces(generated_utterance) if saved_args.bpe_model_path is not None \
                else " ".join(generated_utterance).replace('<s> ', '').replace(' </s>', '')

            # index, expected utterance, generated utterance, all keywords, known keywords, number of prime words, if unk words present
            generated_data.append([str(i), test_utterance, generated_utterance, ' '.join(keywords[i]), ' '.join(known_keywords[i]), str(args.prime), str(with_unk)])

            print('\t'.join(generated_data[-1]))

    with open(os.path.join(args.output_dir, 'generated_utterances.txt'), 'w') as f:
        f.write('\n'.join(map(lambda x: '\t'.join(x), generated_data)) + '\n')


if __name__ == '__main__':
    main()
