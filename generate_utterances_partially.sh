#!/bin/bash

# Usage:
# bash generate_utterances_partially.sh [generate_utterances_command] [input_file] [chunk_size]
# where:
# [generate_utterances_command] is a string containing the command of generate_utterances.py script execution (without start_index and nr_of_loops arguments provided)
# [input_file] is a path to an input file (the same as in [generate_utterances_command])
# [chunk_size] is the value for nr_of_loops argument of generate_utterances.py script
# Example:
# bash generate_utterances_partially.sh 'python3 generate_utterances.py -i save -t "data/test_data_100/test.txt" --pick=2 --tokens -q -o "generated_test" -k "data/test_data_100/test_tagged.txt"' data/test_data_100/test.txt 25
#
# You can extract generated utterances from log with this command:
# grep -P "^[0-9]+\t" <path_to_log_file>

NR_OF_LINES=$(wc -l < $2)

for start_index in `seq -s ' ' 0 $3 $NR_OF_LINES`; do

    command=$(echo "$1 --start_index $start_index --nr_of_loops $3")
    eval $command

done

