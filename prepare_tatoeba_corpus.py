from numpy.random import seed
seed(1)

import argparse
import os
import xml.etree.ElementTree as ET

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import clean_str, punctuation_to_leave, stop_words, translator

min_seq_length = 4
min_test_data_seq_length = 8
max_test_data_seq_length = 15
max_test_prime = 3
test_data_percentage = 0.01
validation_data_percentage = 0.1

pos_tags = ['_JJ', '_JJR', '_JJS',
            '_NN', '_NNP', '_NNPS', '_NNS',
            '_RB', '_RBR', '_RBS',
            '_VB', '_VBD', '_VBG', '_VBN', '_VBP', '_VBZ']


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-s', '--stanford_pos_tagger_dir', type=str, default="stanford_pos_tagger")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    tree = ET.parse(args.input_file)
    root = tree.getroot()
    sentences = []
    for sentence in root:
        sentence_words = []
        for chunk in sentence:
            for word in chunk:
                sentence_words.append(word.text)
        sentences.append(' '.join(sentence_words))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    data = pd.DataFrame(sentences)

    with open(os.path.join(args.output_dir, 'input_raw.txt'), 'w') as f:
        data.to_csv(f, sep='\t', header=False, index=False)

    print("Initial corpora size: {}".format(len(data)))

    # remove lines containing non latin signs
    data = data[data[0].apply(lambda sentence: not any(ord(sign) > 127 for sign in sentence))]

    print("Size after removing utterances with non latin signs: {}".format(len(data)))

    # remove all sentences with digits
    data = data[data[0].apply(lambda sentence: not any(sign.isdigit() for sign in sentence))]

    print("Size after removing utterances with digits: {}".format(len(data)))

    # apply initial cleaning
    data[0] = data[0].apply(clean_str)

    # remove sentences with less than min_seq_length words, not counting the punctuation
    data = data[data[0].apply(lambda sentence: [word for word in sentence.split() if word not in punctuation_to_leave]).map(len) >= min_seq_length]

    print("Size after removing utterances shorter than {} words: {}".format(min_seq_length, len(data)))

    with open(os.path.join(args.output_dir, 'input.txt'), 'w') as f:
        data.to_csv(f, sep='\t', header=False, index=False)

    # prepare pos tagged version of data file
    bash_command = 'bash extract_pos_tags.sh {stanford_pos_tagger_dir} {input_file} {output_file}'.format(
        stanford_pos_tagger_dir=args.stanford_pos_tagger_dir,
        input_file=os.path.abspath(os.path.join(args.output_dir, 'input.txt')),
        output_file=os.path.abspath(os.path.join(args.output_dir, 'input_tagged.txt')))

    print('Executing bash command: {}'.format(bash_command))

    os.system(bash_command)

    data['key_words'] = list(pd.read_csv(os.path.join(args.output_dir, 'input_tagged.txt'), sep='\t', header=None)[0])

    # remove lines without any keywords of specified pos tags, not counting the stop words
    data = data[data['key_words'].apply(lambda sentence: [word.split('_')[0]
                                                          for word in sentence.strip().split()
                                                          if word.endswith(tuple(pos_tags))
                                                          and word.split('_')[0] not in stop_words]).map(len) > 0]

    print("Size after removing utterances without specified pos tags: {}".format(len(data)))

    test_data_indexes = data.copy()

    test_data_indexes[0] = test_data_indexes[0].apply(lambda sentence: sentence.split())

    # find sentences with desired test data length
    test_data_indexes = test_data_indexes[(test_data_indexes[0].map(len) >= min_test_data_seq_length)
                                          & (test_data_indexes[0].map(len) <= max_test_data_seq_length)]

    # take only the sentences which contain at least one desired pos tag starting from max_test_prime positions
    test_data_indexes = test_data_indexes[test_data_indexes['key_words'].apply(
        lambda sentence: [word.split('_')[0]
                          for word in sentence.strip().split()[max_test_prime:]
                          if word.endswith(tuple(pos_tags))
                          and word.split('_')[0] not in stop_words]).map(len) > 0]

    # sample up to test_data_percentage of main corpus size
    test_data_indexes = test_data_indexes.sample(min(int(len(data) * test_data_percentage), len(test_data_indexes)))

    test_data_indexes = test_data_indexes.index

    # from the main corpus take the test data by indexes
    test_data = data.ix[test_data_indexes]

    # remove the test indexes from the main corpus
    data.drop(test_data_indexes, inplace=True)

    # split remaining data into training and validation parts
    training_data, validation_data = train_test_split(data, test_size=validation_data_percentage)

    corporas = {'training_data': training_data,
                'test_data': test_data,
                'validation_data': validation_data,
                'non_test_data': data}

    for corpora_type in corporas:

        corpora_directory = os.path.join(args.output_dir, corpora_type)

        if not os.path.exists(corpora_directory):
            os.mkdir(corpora_directory)

        with open(os.path.join(corpora_directory, 'input.txt'), 'w') as f:
            corporas[corpora_type][0].to_csv(f, sep='\t', header=False, index=False)

        with open(os.path.join(corpora_directory, 'input_tagged.txt'), 'w') as f:
            corporas[corpora_type]['key_words'].to_csv(f, sep='\t', header=False, index=False)
