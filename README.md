<p align="center"><a href="https://wyydsb.xin" target="_blank" rel="noopener noreferrer"><img width="100" src="https://cdn.nlark.com/yuque/0/2018/jpeg/104214/1542104633961-66ba586b-4203-4442-b7b2-7de7bc693497.jpeg" alt="Spider logo"></a></p>
<h1 align="center">Model retrieval</h1>

[![GitHub](https://img.shields.io/github/license/iofu728/Model_retrieval.svg?style=popout-square)](https://github.com/iofu728/Model_retrieval/master/LICENSE)
[![GitHub tag](https://img.shields.io/github/tag/iofu728/Model_retrieval.svg?style=popout-square)](https://github.com/iofu728/Model_retrieval)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/iofu728/Model_retrieval.svg?style=popout-square)](https://github.com/iofu728/Model_retrieval)

<div align="center"><strong>Some ML Model retrieval</strong></div>

* [VSM(updated in 2018/11/13)](#1)
* [SMN(updated in 2018/12/04)](#2)

<h2 id="1">VSM</h2>

> VSM = Vector Space Model

This a `hand write VSM` retrieval

```vim
├── utils
│   └── utils.py         // public function
└── vsm
    ├── pre.sh           // data preprocessing shell
    └── vsm.py           // vsm py
```

VSM process:

1. word alignment
2. TF - IDF (smooth, similarity)
3. one by one calaulate

* `VSM.vsmCalaulate()`
  + Consider about bias by smooth
  + Choose one tuple(artile1, artile2) have specific (tf-idf1, tf-idf2)
  + In this way, we have low performance, even we have two class Threadings
* `VSM.vsmTest()`
  + Ignore bias by smooth
  + Calculate tf-idf in the pre processing which decided by artile instead of tuple(artile1, artile2)
  + In this way, we have fantastic performance
  + We calculate dataset of 3100✖️3100 in 215s

<h2 id="2">SMN</h2>

> SMN = Sequential Matching Network

some change from [MarkWuNLP/MultiTurnResponseSelection](https://github.com/MarkWuNLP/MultiTurnResponseSelection)

```vim
.
├── NN
│   ├── CNN.py              // CNN function
│   ├── Classifier.py       // classifier function
│   ├── Optimization.py     // NN optimization function
│   ├── RNN.py              // RNN function
│   └── logistic_sgd.py     // sgd function
├── SMN
│   ├── PreProcess.py       // pre deal function
│   ├── SMN_Last.py         // model function
│   ├── SimAsImage.py       // cnn pool & conv
│   └── sampleConduct.py    // got negative and true sample
└── utils
    ├── constant.py         // constant parameter
    └── utils.py            // public function
```

SMN process:

1. word embemdding
2. GRU
3. CNN
4. GRU
5. score

* `SMN.PreProcess.ParseMultiTurn(input_file)`
  + prepare deal sample to matrix
* `SMN.PreProcess..ParseMultiTurnTest(input_file)`
  + prepare deal test sample to matrix
* `SMN.sampleConduct.preWord2vec(input_file, out_file)`
  + embedding sample
* `SMN.sampleConduct.SampleConduct()`
  + got negative & true sample
* `SMN.SMN_Last.run_model()`
  + run SMN model
