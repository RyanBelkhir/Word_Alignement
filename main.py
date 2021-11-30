import time
import numpy as np
import pandas as pd
from data_initialization import load_vec, get_nn, build_dico
from supervised import linear_transform
from unsupervised import Discriminator, Generator, train, init_gan
from metrics import accuracy 


if __name__ == '__main__':
    # Hyperparameters
    NMAX = 200000

    # Data Initialization
    src_path = 'data/wiki.en.vec'
    targ_path = 'data/wiki.fr.vec'

    src_embeddings, src_id2word, src_word2id = load_vec(src_path, NMAX)
    targ_embeddings, targ_id2word, targ_word2id = load_vec(targ_path, NMAX)

    df_train = pd.read_csv("data/en-fr.0-5000.txt", sep=" ", header = None)
    df_test = pd.read_csv("data/en-fr.5000-6500.txt", sep=" ", header = None)

    df_train['SrcEmbed'] = df_train[0].apply(lambda k: src_embeddings[src_word2id[k]])
    df_train['TgtEmbed'] = df_train[1].apply(lambda k: targ_embeddings[targ_word2id[k]])
    df_test['SrcEmbed'] = df_test[0].apply(lambda k: src_embeddings[src_word2id[k]])
    df_test['TgtEmbed'] = df_test[1].apply(lambda k: targ_embeddings[targ_word2id[k]])

    # Check if there is NaN value
    print(df_test.isna().values.any())

    model = linear_transform(np.array(list(df_train['SrcEmbed'])), np.array(list(df_train['TgtEmbed'])), df_train[0], df_train[1])
    dico_src_tgt = build_dico(df_test[0], df_test[1])
    model = linear_transform(np.array(list(df_train['SrcEmbed'])), np.array(list(df_train['TgtEmbed'])), df_train[0], df_train[1])
    t = time.time()
    model.procrustes()
    t2 = time.time() - t
    print("Fitting done in {} sec.".format(np.round(t2, 2)))

    t = time.time()

    preds  = model.predict(np.array(list(df_test['SrcEmbed'])), targ_embeddings,  targ_id2word)
    print("Accuracy : {} %.".format(accuracy(df_test[0], preds, dico_src_tgt)))
    t2 = time.time() - t
    print("Fitting done in {} sec.".format(np.round(t2, 2)))
    

    dico_src_tgt_test = build_dico(df_test[0], df_test[1])
    dico_src_tgt_train = build_dico(df_train[0], df_train[1])

    netD, netG, D_optimizer, G_optimizer, loss, src_dataloader, tgt_dataloader = init_gan(src_embeddings, targ_embeddings)
    history = train( netD, netG, D_optimizer, G_optimizer, loss, src_dataloader, tgt_dataloader, src_embeddings, targ_embeddings,targ_id2word, df_test, dico_src_tgt)

