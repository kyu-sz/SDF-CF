#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"/datasets/vot2017
mkdir -p $DIR
cd $DIR
wget -c http://data.votchallenge.net/vot2017/vot2017.zip && unzip vot2017.zip && rm vot2017.zip
