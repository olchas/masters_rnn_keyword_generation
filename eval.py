from __future__ import print_function
import argparse
import os

import tensorflow as tf
from nlgeval import compute_metrics
from six.moves import cPickle

from model import Model


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
    parser.add_argument('--keywords_file', '-k', type=str, default=None,
                        help='path to file with pos tagged test file')
    parser.add_argument('--pos_tags', type=str, nargs='+',
                        default=['_JJ', '_NN', '_NNP', '_NNPS', '_NNS', '_RB', '_VB', '_VBD', '_VBG', '_VBN', '_VBP', '_VBZ'],
                        help='list of pos tags that should be used as keywords')
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
    with open(saved_args.words_vocab_file, 'rb') as f:
        words, vocab = cPickle.load(f)


    with open(args.test_file) as f:
        test_utterances = f.read().strip().split('\n')
    maximal_length = max(map(lambda x: len(x), test_utterances))

    if args.keywords_file is not None:
        with open(args.keywords_file) as f:
            tagged_utterances = f.read().strip().split('\n')
        keywords = [[word.split('_')[0] for word in utterance.split()[args.prime:] if word.endswith(tuple(args.pos_tags))]
                    for utterance in tagged_utterances]
        with open(os.path.join(args.output_dir, 'keywords.txt'), 'w') as f:
            f.write('\n'.join([' '.join(keyword_group) for keyword_group in keywords]) + '\n')
    else:
        keywords = [[]] * len(test_utterances)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    generated_utterances_file = open(os.path.join(args.output_dir, 'generated_utterances.txt'), 'w')

    generated_utterances = []
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        if args.model_checkpoint is not None:
            model_checkpoint_path = args.model_checkpoint
        else:
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            model_checkpoint_path = ckpt.model_checkpoint_path

        saver.restore(sess, model_checkpoint_path)
        for i, test_utterance in enumerate(test_utterances):
            prime = ' '.join(test_utterance.split()[:args.prime])
            utterance_keywords = keywords[i]
            print(test_utterance, prime, keywords[i])

            generated_utterance = model.sample(sess, words, vocab, maximal_length, prime, args.sample,
                                               args.pick, args.width, args.quiet, args.bpe_model_path,
                                               args.tokens, utterance_keywords)

            generated_utterances.append(generated_utterance)

            generated_utterances_file.write(generated_utterance + '\n')

    generated_utterances_file.close()

    with open(os.path.join(args.output_dir, 'generated_utterances_stripped.txt'), 'w') as f:
        f.write('\n'.join(map(lambda x: ' '.join(x.split()[args.prime:]), generated_utterances)) + '\n')

    with open(os.path.join(args.output_dir, 'test_utterances_stripped.txt'), 'w') as f:
        f.write('\n'.join(map(lambda x: ' '.join(x.split()[args.prime:]), test_utterances)) + '\n')

    metrics_dict_stripped = compute_metrics(hypothesis=os.path.join(args.output_dir, 'generated_utterances_stripped.txt'),
                                            references=[os.path.join(args.output_dir, 'test_utterances_stripped.txt')],
                                            no_skipthoughts=True,
                                            no_glove=True)

    metrics_dict = compute_metrics(hypothesis=generated_utterances_file,
                                   references=[args.test_file],
                                   no_skipthoughts=True)

    with open(os.path.join(args.output_dir, 'results_stripped.txt'), 'w') as f:
        for metric in sorted(metrics_dict_stripped.keys()):
            f.write('{metric}\t{value}\n'.format(metric=metric, value=metrics_dict_stripped[metric]))

    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        for metric in sorted(metrics_dict.keys()):
            f.write('{metric}\t{value}\n'.format(metric=metric, value=metrics_dict[metric]))


if __name__ == '__main__':
    main()
