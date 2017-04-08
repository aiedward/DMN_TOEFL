Hierachical Episode Model for Question Answering
===============================================

An Modification of the Hierachical Attention Model described in the paper 
[Hierarchical Attention Model for Improved Comprehension of Spoken Content](https://128.84.21.199/abs/1608.07775)
by Wei Fang, Juei-Yang Hsu, Hung-Yi Lee, and Lin-Shan Lee.

We use **Episode memory module** of **Dynamic Memory Network plus** described in the paper [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417) to substitute the MemN2N in the HAM.

## Requirements

- [Torch7](https://github.com/torch/torch7)
- [penlight](https://github.com/stevedonovan/Penlight)
- [nn](https://github.com/torch/nn)
- [nngraph](https://github.com/torch/nngraph)
- [optim](https://github.com/torch/optim)
- [rnn](https://github.com/Element-Research/rnn)
- Java >= 8 (for Stanford CoreNLP utilities)
- Python2 >= 2.7
- Python3 >= 3.5

The Torch/Lua dependencies can be installed using [luarocks](http://luarocks.org). For example:

```
luarocks install nngraph
```

## Usage

First run the following script:

```
sh preprocess.sh
```

This downloads the following data:
  - [TOEFL Listening Comprehension Test Dataset](https://github.com/sunprinceS/Hierarchical-Attention-Model/releases/download/0.0.1/to_project.zip)
  - [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840B) -- **Warning:** this is a 2GB download!

and the following libraries:

  - [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml)
  - [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)

The preprocessing script generates dependency parses of the TOEFL dataset using the
[Stanford Neural Network Dependency Parser](http://nlp.stanford.edu/software/nndep.shtml).

Alternatively, the download and preprocessing scripts can be called individually.

## TOEFL Listening Comprehension Test

TOEFL is an English examination which tests knowledge and skills of academic English for non-native English learners. Each example consists of an audio story, a question, and four answer choices. Among these choices, one or two of them are correct. given the manual or ASR transcriptions of an audio story and a question, machine has to select the correct answer out of the four choices.

To train models for the TOEFL Listening Comprehension Test, 
run:

```
th toefl/main.lua --model <|hem|ham|lstm|bilstm|treelstm|memn2n> --task <manual|ASR> --level <phrase|sentence> --dim <sentence_representation_dim> --internal <dmn_dim> --hops <dmn_hops> --layers <num_layers> --epochs <num_epochs> --prune <pruning_rate>
```

where:

  - `model`: the model to train (default: hem, i.e. Hierachical Episode Model)
  - `task`: the transcription to be trained on  (default: manual)
  - `level`: the attention level of the HAM (default: sentence, ignored for other models)
  - `dim`: the dimension for sentence/phrase representations (default: 75)
  - `internal`: the dimension for memory module in DMN or for MemN2N (default: 75, ignored for other models)
  - `hops`: the number of hops for DMN or MemN2N (default: 1, ignored for other models)
  - `layers`: the number of layers for LSTM or BiLSTM (default: 1, ignored for other models)
  - `epochs`: the number of training epochs (default: 10)
  - `prune`: the preprocessing prune rate (default: 1, i.e. no pruning)

Trained model parameters are saved to the `trained_models` directory.

### The Limitation

This Hierachical Episode Model now has no ASR option, and the prune rate only for 0.1 & 1.0
