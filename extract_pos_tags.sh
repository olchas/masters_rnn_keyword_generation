#!/bin/bash

# Usage:
# bash extract_pos_tags.sh [stanfor_pos_tagger_dir] [input_file] [output_file]
# where:
# [stanfor_pos_tagger_dir] is directory with stanford-postagger.sh script
# [input_file] is a path to an input file containing one sentence per one line
# [output_file] is the path to file where tagged sentences should be stored

cd $1
bash stanford-postagger.sh models/wsj-0-18-bidirectional-distsim.tagger $2 > $3
cd - 
