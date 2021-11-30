import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_nn_avg_dist(emb, query, knn=10):
    """Function to compute the average similarity of word's neighborhood """
    bs = 32
    emb = torch.Tensor(emb)
    query = torch.Tensor(query)
    all_distances = []
    emb = emb.transpose(0, 1).contiguous()
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb)
        best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances)
    return all_distances.numpy()


def CSLS(src_emb, tgt_emb):
    """Function ton compute the CSLS metric describe MUSE paper"""
    src_emb = torch.from_numpy(src_emb)
    tgt_emb = torch.from_numpy(tgt_emb)
    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
    src_emb = src_emb.numpy()
    tgt_emb = tgt_emb.numpy()
    cosine = cosine_similarity(src_emb, tgt_emb)
    src_avg = get_nn_avg_dist(tgt_emb, src_emb)
    
    tgt_avg = get_nn_avg_dist(src_emb, tgt_emb)
    cosine = np.subtract(2 * cosine, tgt_avg)
    csls_val = np.subtract(cosine.T, src_avg).T

    return csls_val


def get_similars(embedd1, embedd2):
    """Build a dictionnary with mutual nearest-neighbor"""
    print("Getting similars...")
    src_scores = CSLS(embedd1, embedd2)
    best_src_scores = src_scores.argmax(axis=1)
    src_similar = {(a, b) for a, b in enumerate(best_src_scores)}
    tgt_scores = CSLS(embedd2, embedd1)
    best_tgt_scores = tgt_scores.argmax(axis=1)
    tgt_similar = {(b, a) for a, b in enumerate(best_tgt_scores)}
    
    dico = list(src_similar.intersection(tgt_similar))
    test = list(src_similar)
    test2 =list(tgt_similar)
    test.sort()
    test2.sort()
    dico.sort()
    print("Dictionnary built.")
    return dico


def accuracy(src_words, preds,  dico):
    """Compute the the accuracy"""
    acc = 0
    src_words = np.array(list(src_words))
    _, idx = np.unique(np.array(list(src_words)), return_index=True)
    src_words = src_words[np.sort(idx)]

    preds = np.array(preds)
    preds = preds[np.sort(idx)]

    for (word1, pred) in zip(src_words, preds):
        if pred in dico[word1]:
            acc +=1
    
    return round(acc / len(preds) * 100, 2)
