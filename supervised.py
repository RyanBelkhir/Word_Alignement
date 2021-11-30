from scipy import linalg
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_initialization import build_dico
from metrics import CSLS


class linear_transform:

    def __init__(self, X, Z, src_words, tgt_words):
        self.X = X
        self.Z = Z
        self.n = self.X.shape[1]
        self.W = np.random.random((self.n, self.n))
        self.dico = build_dico(src_words, tgt_words)
        self.src_words = src_words


    def score(self):
        preds = model.predict(self.X, targ_embeddings,  targ_id2word)
        acc = 0
        
        _, idx = np.unique(self.src_words, return_index=True)
        src_words = self.src_words[np.sort(idx)]
        preds = preds[np.sort(idx)]

        for (word1, pred) in zip(self.src_words, preds):
            if pred in self.dico[word1]:
                acc +=1
        
        return round(acc / len(preds) * 100, 2)


    def fit(self, learning_rate=0.000322, n_iter=100, ortho=True):
        """Gradient Descent with or without orthogonalization"""
        cpt = 0
        print(self.score())
        while cpt < n_iter:
            cpt += 1
            gradient = self.X.T @ (self.X @ self.W - self.Z)
            self.W = self.W - learning_rate * gradient

            if ortho:
                Q, E, P = linalg.svd(self.W, full_matrices=True)
                self.W = Q.dot(P)
                # print(np.round(self.W @ self.W.T))

            print(self.score())


    def fit_cosine(self, learning_rate=0.1, n_iter=100, ortho=True):
        """Gradient Descent with the second problem to maximize the cosine similarity"""
        cpt = 0
        print(self.score())
        gradient = np.zeros_like(self.W)
        for i in range(self.n):
            gradient += np.outer(self.X[i],self.Z[i])
        print(gradient)
        print(gradient.shape)
        while cpt < n_iter:
            print(cpt)
            cpt += 1

            self.W = self.W + learning_rate * gradient
            if ortho:
                Q, E, P = scipy.linalg.svd(self.W, full_matrices=True)
                E[E != 0] = 1
                self.W = Q.dot(P)
                # print(np.round(self.W @ self.W.T))

            print(self.score())


    def fit_sgd(self, learning_rate=0.1, n_iter=1000, ortho=True):
        """Stochastic Gradient Descent with or without orthogonalization"""
        print('X :', self.X.shape)
        print('W :', self.W.shape)
        print('Z :', self.Z.shape)
        cpt = 0
        S = list(range(self.W.shape[0]))
        print(self.score())
        while cpt < n_iter:
            np.random.shuffle(S)
            for i in S:
                error = self.X[i][:] @ self.W - self.Z[i][:]
                gradient =  np.dot(self.X[i][:][:,None], error[None, :])
                self.W = self.W - learning_rate * gradient
            cpt += 1

            if ortho:
                Q, E, P = linalg.svd(self.W, full_matrices=True)
                E[E != 0] = 1
                self.W = Q.dot(np.diag(E)).dot(P)
                # print(np.round(self.W @ self.W.T))

            print(self.score())

    def procrustes(self):
        U, _, V = linalg.svd(self.X.T.dot(self.Z), full_matrices=True)
        self.W = U.dot(V)

            



    def ols(self, ortho=True):
        least_square = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Z
        self.W = least_square
        if ortho:
            Q, E, P = linalg.svd(self.W, full_matrices=True)
            E[E != 0] = 1
            self.W = Q.dot(np.diag(E)).dot(P)

    def fit_ortho(self, learning_rate = 0.1, n_iter = 100):
        
        for i in range(n_iter):
            gradW=np.zeros_like(self.W)
            for j in range(self.X.shape[0]):
                gradW+=np.outer(self.X[j],self.Z[j])
                
            self.W = self.W + learning_rate*gradW
            u, s, vT = np.linalg.svd(self.W)
            self.W = np.dot(u,vT)
        print('Model FITTED')

    def predict(self, test_set, tgt_emb, tgt_id2word, method='CSLS'):
        """Predict according to the closest neighbor"""
        print("Start prediction...")
        preds = []
        word_emb = np.dot(test_set,self.W)

        # Here we can change CSLS with cosine_similarity() to have a NN prediction
        if method=='CSLS':
            scores = CSLS(word_emb, tgt_emb)
        else:
            scores = cosine_similarity(word_emb, tgt_emb)
        best_cos_sim = scores.argmax(axis=1)
        for i in best_cos_sim:
            preds.append(tgt_id2word[i])
        print("Prediction Done.")
        return(np.array(preds))
    

