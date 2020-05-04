from collections import defaultdict
import itertools as it
import tempfile

from gensim.models import KeyedVectors
import numpy as np


def matrix_to_w2v(matrix, vocab):
    vector_size = matrix.shape[1]
    tfile = tempfile.NamedTemporaryFile()
    with open(tfile.name, 'w') as f:
        meta_info = str(matrix.shape[0]) + ' ' + str(vector_size) + '\n'
        f.write(meta_info)
        for word, vec in zip(vocab, matrix):
            line = word + ' ' + ('{:.6f} ' * vector_size).format(*vec).strip(' ') + '\n'
            f.write(line)
    model = KeyedVectors.load_word2vec_format(tfile.name)
    tfile.close()
    return model


def most_similar_from_vocab(model, vocab):
    similarity_map = defaultdict(list)
    for w1, w2 in it.combinations(vocab, 2):
        if w1 in model.wv and w2 in model.wv:
            similarity = model.similarity(w1, w2)
        else:
            similarity = -1.
        similarity_map[w1].append((w2, similarity))
        similarity_map[w2].append((w1, similarity))
    return {w: sorted(similar, key=lambda pair: pair[1], reverse=True)
            for w, similar in similarity_map.items()}


def topn_overlap(model_1, model_2, vocab, n=5):
    '''Compares two models by how much their top n similarity results overlap.'''
    similarities_1 = most_similar_from_vocab(model_1, vocab)
    similarities_2 = most_similar_from_vocab(model_2, vocab)
    overlaps = []
    for word in vocab:
        topn_1 = [w for w, _ in similarities_1[word]][:n]
        topn_2 = [w for w, _ in similarities_2[word]][:n]
        overlaps.append(len([w for w in topn_1 if w in topn_2]))
    return np.mean(overlaps) / n


def preprocess_dataset(dataset):
    '''The simplest possible tokenisation'''
    allowed_characters = set('abcdefghijklmnopqrstuvwxyz0123456789-')
    corpus = []
    for paper in dataset:
        paper_sentences = paper['description'].lower().split('.') + [paper['title']]
        for sentence in paper_sentences:
            s = ''.join([c if c in allowed_characters else ' ' for c in sentence]).split(' ')
            s = [word for word in s if word != '']
            if s:
                corpus.append(s)
    return corpus

