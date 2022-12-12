STUDENT = {'name': "Osnat Ackerman_Shira Yair",
    'ID': '315747204_315389759'}
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 300
PROJECT_DIM = 200


class Input_representation(nn.Module):

    def __init__(self, output_dim, text_field, device, if_cud):
        super(Input_representation, self).__init__()
        self.if_cud = if_cud
        self.device = device
        self.vocab_size = len(text_field.vocab.vectors)
        self.text_field = text_field
        weight = torch.tensor(self.text_field.vocab.vectors.clone().detach(), dtype=torch.float32, device=device)
        oov_vecs = torch.empty(100, EMBEDDING_DIM, device=self.device).normal_(mean=0, std=1)
        if if_cud:
            weight = weight.cuda()
            oov_vecs = oov_vecs.cuda()
        weight = torch.cat((weight, oov_vecs), 0)
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=1, max_norm=1)
        self.input_linear = nn.Linear(EMBEDDING_DIM, PROJECT_DIM)

        # atten
        self.project_dim = PROJECT_DIM
        self.F_ = self.build_layers(PROJECT_DIM, PROJECT_DIM, 0.2)
        self.G = self.build_layers(PROJECT_DIM * 2, PROJECT_DIM, 0.2)
        self.H = self.build_layers(PROJECT_DIM * 2, PROJECT_DIM, 0.2)
        self.linear_output = nn.Linear(PROJECT_DIM, output_dim)


    def build_layers(self, input_dim, output_dim, dropout_):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(p=dropout_),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(p=dropout_),
            nn.ReLU()
        )

    def Attend(self, sentence_a, sentence_b, length_a, length_b):
        a = self.F_(sentence_a.view(-1, PROJECT_DIM))  # a: ((batch_size * l_a) x hidden_dim)
        a = a.view(-1, sentence_a.shape[1], PROJECT_DIM)  # a: (batch_size x l_a x hidden_dim)
        b = self.F_(sentence_b.view(-1, PROJECT_DIM))  # b: ((batch_size * l_b) x hidden_dim)
        b = b.view(-1, sentence_b.shape[1], PROJECT_DIM)  # b: (batch_size x l_b x hidden_dim)
        e = torch.bmm(a, torch.transpose(b, 1, 2))  # e: (batch_size x l_a x l_b)
        beta = torch.bmm(F.softmax(e, dim=2), b)  # beta: (batch_size x l_a x hidden_dim)
        alpha = torch.bmm(F.softmax(torch.transpose(e, 1, 2), dim=2), a)  # alpha: (batch_size x l_b x hidden_dim)
        return beta, alpha, a, b

    def Compare(self, beta, alpha, a, b):
        a_cat_beta = torch.cat((a, beta), dim=2)
        b_cat_alpha = torch.cat((b, alpha), dim=2)
        v1 = self.G(a_cat_beta.view(-1, 2 * PROJECT_DIM))  # v1: ((batch_size * l_a) x hidden_dim)
        v2 = self.G(b_cat_alpha.view(-1, 2 * PROJECT_DIM))
        return a_cat_beta, b_cat_alpha, v1, v2

    def Agregate(self, v1, v2, length_a, length_b):
        v1 = torch.sum(v1.view(-1, length_a, PROJECT_DIM), dim=1)  # v1: (batch_size x 1 x hidden_dim)
        v2 = torch.sum(v2.view(-1, length_b, PROJECT_DIM), dim=1)  # v2: (batch_size x 1 x hidden_dim)

        v1 = torch.squeeze(v1, dim=1)
        v2 = torch.squeeze(v2, dim=1)

        v1_cat_v2 = torch.cat((v1, v2), dim=1)  # v1_cat_v2: (batch_size x (hidden_dim * 2))
        h = self.H(v1_cat_v2)
        return h

    def encode(self, sentence_a, sentence_b):
        if self.if_cud:
            sentence_a = sentence_a.cuda()
            sentence_b = sentence_b.cuda()
        torch.where(sentence_a == 0, torch.tensor(random.randint(self.vocab_size,
                                                                 self.vocab_size + 101), device=self.device),
                    sentence_a)
        torch.where(sentence_b == 0, torch.tensor(random.randint(self.vocab_size,
                                                                 self.vocab_size + 101), device=self.device),
                    sentence_b)
        batch_size = sentence_a.size(0)
        sentence_a = self.embedding(sentence_a)
        sentence_b = self.embedding(sentence_b)

        sentence_a = sentence_a.view(-1, EMBEDDING_DIM)
        sentence_b = sentence_b.view(-1, EMBEDDING_DIM)

        sentence_a_shrink = self.input_linear(sentence_a).view(
            batch_size, -1, PROJECT_DIM)
        sentence_b_shrink = self.input_linear(sentence_b).view(
            batch_size, -1, PROJECT_DIM)

        return sentence_a_shrink, sentence_b_shrink

    def forward(self, sentence_a, sentence_b, length_a, length_b):
        sentence_a, sentence_b = self.encode(sentence_a, sentence_b)
        beta, alpha, a, b = self.Attend(sentence_a, sentence_b, length_a, length_b)
        a_cat_beta, b_cat_alpha, v1, v2 = self.Compare(beta, alpha, a, b)
        h = self.Agregate(v1, v2, sentence_a.shape[1], sentence_b.shape[1])
        output = self.linear_output(h)

        return output
