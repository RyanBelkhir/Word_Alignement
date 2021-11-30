import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import device
from metrics import CSLS, get_similars, accuracy
from torch.utils.data import TensorDataset, DataLoader
from data_initialization import build_dico
from torch.optim.lr_scheduler import StepLR

# Hyperparameters
D = 300
learning_rate = 0.1
decay = 0.98
batch_size = 32
epochs = 40
smooth = 0.2
n_refinement = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):

    def __init__(self, id_init = True):
        super(Generator, self).__init__()
        self.W = nn.Linear(D, D, bias=False)
        if id_init :
            self.W.weight.data.copy_(torch.diag(torch.ones(D)))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.W(x)
        return x

    def get_w(self):
        return self.W.weight

    def orthogonalize(self):
        W = self.W.weight.data
        beta = 0.01
        self.W.weight.data = W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0,1).mm(W)))

    def procrustes(self, src_embedd, tgt_embedd, dico):
        print(dico)
        A = src_embedd[dico[:,0]]
        B = tgt_embedd[dico[:,1]]
        print("A :", A.shape)
        print("B :", B.shape)
        #W = self.W.weight.data
        U, S, V_t = scipy.linalg.svd(B.T.dot(A), full_matrices=True)
        self.W.weight.data.copy_(torch.from_numpy(U.dot(V_t)).type_as(self.W.weight.data))

    def predict(self, test_emb, tgt_emb, tgt_id2word):
        """Predict function for the GAN"""
        preds = []
        word_emb = self(torch.Tensor(test_emb).to(device))
        word_emb = word_emb.detach().cpu().numpy()

        scores = CSLS(word_emb, tgt_emb)
        k_best = scores.argmax(axis=1)
        for i in k_best:
            preds.append(tgt_id2word[i])

        return preds


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_fc = nn.Linear(D, 2048)
        self.hidden_fc1 = nn.Linear(2048, 2048)
        self.hidden_fc2 = nn.Linear(2048, 2048)
        self.out_fc = nn.Linear(2048, 1)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout(self.input_fc(x))
        x = self.activation(x)
        x = self.dropout(self.hidden_fc1(x))
        x = self.activation(x)
        x = self.dropout(self.hidden_fc2(x))
        x = self.activation(x)
        x = self.out_fc(x)
        return nn.Sigmoid()(x)


def train(netD, netG, D_optimizer, G_optimizer, loss, src_dataloader, tgt_dataloader, src_embed, tgt_embed, targ_id2word, df_test, dico_src_tgt_test, epochs=epochs, batch_size=batch_size):

    history = {'accuracy': [],
                'D_loss': [],
                'G_loss': []}

    scheduler_D = StepLR(D_optimizer, step_size=1, gamma=decay)
    scheduler_G = StepLR(G_optimizer, step_size=1, gamma=decay)

    print("Starting training...")
    for epoch in range(epochs):
        for idx, words in enumerate(tgt_dataloader):
            
    
            idx += 1
            # We train the discriminator
            # with real inputs from the dataloader and fake ones from the generator
            real_inputs = words[0].float().to(device)
            real_outputs = netD(real_inputs)
            real_label = torch.add(torch.ones(real_inputs.shape[0], 1), - smooth).to(device)

            noise = next(iter(src_dataloader))
            noise = noise[0].float().to(device)

            # We generate fake data using the generator
            # And train the discriminator on it
            fake_inputs = netG(noise)
            fake_outputs = netD(fake_inputs)
            fake_label = torch.add(torch.zeros(fake_inputs.shape[0], 1), smooth).to(device)


            outputs = torch.cat((real_outputs, fake_outputs), dim=0)
            targets = torch.cat((real_label, fake_label), dim=0)

            D_loss = loss(outputs, targets)
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # We train the generator
            noise = next(iter(src_dataloader))
            noise = noise[0].float().to(device)

            fake_inputs = netG(noise)
            fake_outputs = netD(fake_inputs)
            fake_targets = torch.add(torch.ones(fake_inputs.shape[0], 1), - smooth).to(device)

            G_loss = loss(fake_outputs, fake_targets)
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            netG.orthogonalize()

            if idx % 50 == 0 or idx == len(tgt_dataloader):
                print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f} lr {}'.format(epoch, idx,
                                                                                                    D_loss.item(),
                                                                                                    G_loss.item(),
                                                                                                    scheduler_D.get_last_lr()))
                
                if idx == len(tgt_dataloader):
                    preds = netG.predict(np.array(list(df_test['SrcEmbed'])), tgt_embed,  targ_id2word)
                    history["accuracy"].append(accuracy(df_test[0], preds, dico_src_tgt_test))
                    history["D_loss"].append(D_loss.item())
                    history["G_loss"].append(G_loss.item())
        scheduler_D.step()
        scheduler_G.step() 


    print("Starting refinement...")
    for n_iter in range(n_refinement):
        pred_embbed = netG(torch.Tensor(src_embed[:50000]).to(device)).detach().cpu().numpy()
        #src_similars = get_similars(pred_embbed, targ_embeddings[:50000])
        #tgt_similars = get_similars(targ_embeddings[:50000], pred_embbed)
        dico = get_similars(pred_embbed, tgt_embed[:10000])
        dico = np.array(dico)
        netG.procrustes(src_embed[:50000], tgt_embed[:50000], dico)

    return history


def init_gan(src_embed, targ_embed):
    netD = Discriminator()
    netD.to(device)
    netG = Generator()
    netG.to(device)

    D_optimizer = optim.SGD(netD.parameters(), lr=learning_rate)
    G_optimizer = optim.SGD(netG.parameters(), lr=learning_rate)

    scheduler_D = StepLR(D_optimizer, step_size=1, gamma=decay)
    scheduler_G = StepLR(G_optimizer, step_size=1, gamma=decay)

    loss = nn.BCELoss()

    targ_dataset = np.array(list(targ_embed[:50000]))
    src_dataset = np.array(list(src_embed[:50000]))

    french_dataset = TensorDataset(torch.from_numpy(targ_dataset))
    english_dataset = TensorDataset(torch.from_numpy(src_dataset))

    tgt_dataloader = DataLoader(french_dataset, batch_size=batch_size, shuffle=True)
    src_dataloader = DataLoader(english_dataset, batch_size=batch_size, shuffle=True)

    return netD, netG, D_optimizer, G_optimizer, loss, src_dataloader, tgt_dataloader

    