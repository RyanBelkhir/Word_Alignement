import io
import numpy as np

def load_vec(emb_path, nmax=50000):
    """"Function to load the words and word embeddings """
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    """"Function to find the nearest neighbors of a word from the src_emb in the tgt_emb"""
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))


def build_dico(src_words, tgt_words):
    """Function ton build a python dictionnary to pair a src word (key) to his translation """

    dico = {}
    for (word1, word2) in zip(src_words, tgt_words):
        if word1 in dico.keys():
            dico[word1].append(word2)
        else:
            dico[word1] = [word2]

    return dico
