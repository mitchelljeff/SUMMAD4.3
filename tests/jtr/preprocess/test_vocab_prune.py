# -*- coding: utf-8 -*-

from jtr.pipelines import pipeline
from pprint import pprint


def test_vocab():
    train_data = {
        'candidates': [['entailment', 'neutral', 'contradiction']],
        'answers': ['neutral'],
        'question': ['A person is training his horse for a competition.'],
        'support': ['A person on a horse jumps over a broken down airplane.']}

    print('build vocab based on train data')
    _, train_vocab, train_answer_vocab, train_candidate_vocab = \
        pipeline(train_data, normalize=True)

    pprint(train_vocab.sym2freqs)
    pprint(train_vocab.sym2id)

    MIN_VOCAB_FREQ, MAX_VOCAB_CNT = 2, 10
    train_vocab = train_vocab.prune(MIN_VOCAB_FREQ, MAX_VOCAB_CNT)

    pprint(train_vocab.sym2freqs)
    pprint(train_vocab.sym2id)

    print('encode train data')
    train_data, _, _, _ = pipeline(train_data, train_vocab, train_answer_vocab, train_candidate_vocab,
                                   normalize=True, freeze=True)
    print(train_data)
