from __future__ import print_function
import argparse
import os

import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='directory with generated utterances')
    parser.add_argument('--accuracy_only', '-a', action='store_true', default=False,
                        help='calculate only accuracies')
    return parser.parse_args()


def main():
    args = parse_arguments()

    evaluate(args)


def calculate_key_words_accuracy(generated_sentences, keywords, data_raw=None, output_dir=None):

    keywords_count = 0
    keywords_generated = 0

    if data_raw is not None and output_dir is not None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        best_utterances_file = open(os.path.join(output_dir, 'best_utterances.tsv'), 'w')
        best_utterances_file.write('id\texpected\tgenerated\tknown_key_words\n')

    for i, generated_sentence in enumerate(generated_sentences):

        expected_keywords = set(keywords[i].split())
        generated_sentence_words = generated_sentence.split()
        sentence_keywords_generated = 0

        # for each unique key word
        for keyword in expected_keywords:
            keywords_count += 1
            if keyword in generated_sentence_words:
                sentence_keywords_generated += 1

        sentence_accuracy = float(sentence_keywords_generated) / len(expected_keywords) if expected_keywords else 0.0
        keywords_generated += sentence_keywords_generated

        if sentence_accuracy == 1.0 and data_raw is not None:
            print('\t'.join([str(i), data_raw['expected'][i], data_raw['generated'][i], str(expected_keywords)]))
            if output_dir is not None:
                best_utterances_file.write('\t'.join([str(i), data_raw['expected'][i], data_raw['generated'][i], str(expected_keywords)]) + '\n')

    key_word_accuracy = keywords_generated / keywords_count if keywords_count else 0.0

    return key_word_accuracy


def calculate_key_words_overloading(generated_sentences, keywords):

    keywords_overgenerated = 0
    keywords_generated = 0
    keywords_expected = 0

    for i, generated_sentence in enumerate(generated_sentences):

        expected_keywords = keywords[i].split()
        generated_sentence_words = generated_sentence.split()

        for keyword in set(expected_keywords):
            keywords_generated += generated_sentence_words.count(keyword)
            # add how many more of that key word appeared in generated sequence
            keywords_overgenerated += max(generated_sentence_words.count(keyword) - expected_keywords.count(keyword), 0)

            keywords_expected += expected_keywords.count(keyword)

    key_word_overloading = keywords_overgenerated / keywords_generated if keywords_generated else 0.0

    return key_word_overloading, keywords_generated, keywords_overgenerated, keywords_expected


def evaluate(args):

    # list to get the desired order of metric printed
    metrics = ['nr', 'all_key_words_expected', 'known_key_words_expected', 'known_key_words_generated',
               'known_key_words_overgenerated', 'all_key_word_accuracy', 'known_key_word_accuracy',
               'known_key_word_overloading']

    if not args.accuracy_only:
        from nlgeval import compute_metrics

        metrics += ['METEOR']

    data = pd.read_csv(os.path.join(args.input_dir, 'generated_utterances.txt'), sep='\t')

    data_raw = data.copy()

    data.replace(np.nan, '', regex=True, inplace=True)

    prime = data['prime'][0]

    # cut out prime words from expected and generated utterances
    data['expected'] = data['expected'].apply(lambda x: ' '.join(x.split()[prime:]))

    data['generated'] = data['generated'].apply(lambda x: ' '.join(x.split()[prime:]))

    categories = {
        '1_full': {'data': data, 'metrics': {}},
        '2_without_unk': {'data': data[data['with_unk'] == False], 'metrics': {}},
        '3_with_unk': {'data': data[data['with_unk'] == True], 'metrics': {}},
        '4_with_known_key_words': {'data': data[data['known_keywords'].map(len) > 0], 'metrics': {}},
        '5_without_known_key_words': {'data': data[data['known_keywords'].map(len) == 0], 'metrics': {}}
    }

    for category in sorted(categories):
        print('Category: {}'.format(category))

        categories[category]['metrics']['nr'] = len(categories[category]['data'])

        categories[category]['metrics']['all_key_word_accuracy'] = calculate_key_words_accuracy(list(categories[category]['data']['generated']),
                                                                                                list(categories[category]['data']['all_keywords']))

        _, _, _, categories[category]['metrics']['all_key_words_expected'] = calculate_key_words_overloading(list(categories[category]['data']['generated']), list(categories[category]['data']['all_keywords']))

        categories[category]['metrics']['known_key_word_accuracy'] = calculate_key_words_accuracy(
            list(categories[category]['data']['generated']),
            list(categories[category]['data']['known_keywords']),
            data_raw if category == '1_full' else None,
            args.input_dir)

        (categories[category]['metrics']['known_key_word_overloading'],
         categories[category]['metrics']['known_key_words_generated'],
         categories[category]['metrics']['known_key_words_overgenerated'],
         categories[category]['metrics']['known_key_words_expected']) = calculate_key_words_overloading(list(categories[category]['data']['generated']),
                                                                                                        list(categories[category]['data']['known_keywords']))

        if not args.accuracy_only:

            with open(os.path.join(args.input_dir, category + '_generated_utterances_stripped.txt'), 'w') as f:
                categories[category]['data']['generated'].to_csv(f, sep='\t', header=False, index=False)

            with open(os.path.join(args.input_dir, category + '_test_utterances_stripped.txt'), 'w') as f:
                categories[category]['data']['expected'].to_csv(f, sep='\t', header=False, index=False)

            categories[category]['metrics'].update(compute_metrics(hypothesis=os.path.join(args.input_dir, category + '_generated_utterances_stripped.txt'),
                                                                   references=[os.path.join(args.input_dir, category + '_test_utterances_stripped.txt')],
                                                                   no_skipthoughts=True,
                                                                   no_glove=True))

            print()

        for metric in metrics:
            print(metric, categories[category]['metrics'][metric])

        print()

    with open(os.path.join(args.input_dir, 'results_stripped.tsv'), 'w') as f:
        f.write('metric\t{categories}\n'.format(categories='\t'.join(sorted(categories.keys()))))
        for metric in metrics:
            metric_values = []
            for category in sorted(categories.keys()):
                metric_values.append(str(categories[category]['metrics'][metric]))
            f.write('{metric}\t{values}\n'.format(metric=metric, values='\t'.join(metric_values)))


if __name__ == '__main__':
    main()

