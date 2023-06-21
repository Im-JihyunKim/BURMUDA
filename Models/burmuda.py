import rich
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.loss import weighted_cross_entropy, discrepancy
from Utils.optimizer import get_scheduler


class BURMUDA_Net(nn.Module):
    """
    Multi-Source Domain Adaptation with Discrepancy: adapts from multi-source with the hypothesis discrepancy 
    Learns both a feature representation and weight alpha
    """

    def __init__(self, num_sources, input_dim, feature_extractor, c1, c2, num_dropout, unc_scale, unc_type, unc_calculation_type):
        super(BURMUDA_Net, self).__init__()
        self.n_sources = num_sources
        self.input_dim = input_dim

        self.loss = nn.CrossEntropyLoss()
        self.weighted_loss = weighted_cross_entropy

        self.min_pred = -np.inf
        self.max_pred = np.inf

        self.feature_extractor = feature_extractor
        self.c1 = c1
        self.c2 = c2
        self.num_dropout = num_dropout
        self.unc_scale = unc_scale
        self.unc_type = unc_type
        self.unc_calculation_type = unc_calculation_type
        # set alpha as parameter
        self.register_parameter(name='alpha',
                                param=torch.nn.Parameter(torch.Tensor(np.ones(self.n_sources) / self.n_sources)))

    def set_optimzers(self, optimizer, lr, weight_decay, momentum, lr_alpha):
        if optimizer == 'adam':
            self.opt_f = torch.optim.Adam([{'params': self.feature_extractor.parameters()}], lr=lr,
                                          weight_decay=weight_decay)
            self.opt_c1 = torch.optim.Adam([{'params': self.c1.parameters()}], lr=lr, weight_decay=weight_decay)
            self.opt_c2 = torch.optim.Adam([{'params': self.c2.parameters()}], lr=lr, weight_decay=weight_decay)
            self.opt_alpha = torch.optim.Adam([{'params': self.alpha}], lr=lr_alpha, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            self.opt_f = torch.optim.SGD([{'params': self.feature_extractor.parameters()}], lr=lr, momentum=momentum,
                                         weight_decay=weight_decay)
            self.opt_c1 = torch.optim.SGD([{'params': self.c1.parameters()}], lr=lr, momentum=momentum,
                                          weight_decay=weight_decay)
            self.opt_c2 = torch.optim.SGD([{'params': self.c2.parameters()}], lr=lr, momentum=momentum,
                                          weight_decay=weight_decay)
            self.opt_alpha = torch.optim.SGD([{'params': self.alpha}], lr=lr_alpha, momentum=momentum,
                                             weight_decay=weight_decay)
        else:
            raise NotImplementedError

    def set_lr_scheduler(self, lr_scheduler):
        if lr_scheduler is None:
            rich.print("No learning rate scheduler")
            return lr_scheduler
        elif (lr_scheduler is not None) & (lr_scheduler == "exponential"):
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt_alpha, gamma=0.95)
            return scheduler
        elif (lr_scheduler is not None) & (lr_scheduler == "cosine"):
            scheduler = get_scheduler(self.opt_alpha, "cosine", 100)
            return scheduler
        else:
            raise NotImplementedError

    def reset_grad(self):
        """
        Set all gradients to zero
        """
        self.opt_f.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        self.opt_alpha.zero_grad()

    def extract_multi_source_hidden(self, sx):
        for i in range(self.n_sources):
            for hidden in self.feature_extractor:
                sx[i] = hidden(sx[i])
        return sx

    def extract_hidden(self, x):
        for hidden in self.feature_extractor:
            x = hidden(x)
        return x

    def forward(self, Xs, Xt, dropout_mode: str = 'Combine'):
        # Feature extractor
        sx, tx = Xs.copy(), Xt.clone()
        del Xs, Xt
        # torch.cuda.empty_cache()

        for i in range(self.n_sources):
            for hidden in self.feature_extractor:
                sx[i] = hidden(sx[i])

        y_spred = []
        for i in range(self.n_sources):
            y_sx = sx[i].clone()
            for hidden in self.c1:
                y_sx = hidden(y_sx)
            y_spred.append(y_sx)

        y_spred2 = []
        for i in range(self.n_sources):
            y_tmp = sx[i].clone()
            for hidden in self.c2:
                y_tmp = hidden(y_tmp)
            y_spred2.append(y_tmp)

        if dropout_mode == 'No':
            y_tpred = self.feature_cls(tx, self.c1)
            y_tpred2 = self.feature_cls(tx, self.c2)

            return y_spred, y_spred2, y_tpred, y_tpred2, None
        else:
            y_tpred, unc1 = self.perform_target_dropout(tx, self.c1, mode=dropout_mode)
            y_tpred2, unc2 = self.perform_target_dropout(tx, self.c2, mode=dropout_mode)
            unc = unc1 + unc2

            return y_spred, y_spred2, y_tpred, y_tpred2, unc

    def train_step1(self, Xs, Xt, ys, yt, clip=1):
        self.train()

        s_pred1, s_pred2, _, _, _ = self.forward(Xs, Xt, 'No')

        source_loss_c1 = self.weighted_loss(s_pred1, ys, self.alpha)
        source_loss_c2 = self.weighted_loss(s_pred2, ys, self.alpha)

        loss_pred = source_loss_c1 + source_loss_c2
        loss_pred.backward(retain_graph=True)

        # Clip gradient
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

        self.opt_f.step()
        self.opt_c1.step()
        self.opt_c2.step()

        self.reset_grad()

    def train_step2(self, Xs, Xt, ys, yt, clip=1):
        self.train()

        s_pred1, s_pred2, t_pred1, t_pred2, unc = self.forward(Xs, Xt, 'Combine')
        if self.unc_type == 'prob':
            unc = self.MinMaxScaler(unc)

        loss_disc = discrepancy(self.alpha, s_pred1, s_pred2,
                                t_pred1, t_pred2)

        loss = -loss_disc
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

        self.opt_c1.step()
        self.opt_c2.step()
        self.reset_grad()


    def train_step3(self, Xs, Xt, ys, yt, clip=1):
        self.train()

        s_pred1, s_pred2, t_pred1, t_pred2, unc = self.forward(Xs, Xt, 'Combine')
        if self.unc_type == 'prob':
            unc = self.MinMaxScaler(unc)

        unc = self.L1_norm(unc)
        loss_disc = discrepancy(self.alpha, s_pred1, s_pred2,
                                t_pred1, t_pred2)

        loss = loss_disc + self.unc_scale * unc
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

        self.opt_f.step()
        self.reset_grad()


    def train_alpha(self, Xs, Xt, ys, yt,
                    clip=1, mu=0, lam_alpha=1):
        """
        Train alpha to minimize the discrepancy
        Inputs:
            - Xs: list of torch.Tensor, source data
            - Xt: torch.Tensor, target data
            - ys: list of torch.Tensor, source y
            - clip: max values of the gradients
        """
        self.train()

        s_pred1, s_pred2, t_pred1, t_pred2, _ = self.forward(Xs, Xt, 'No')

        # loss for alpha
        source_loss_c1 = self.weighted_loss(s_pred1, ys, self.alpha)
        source_loss_c2 = self.weighted_loss(s_pred2, ys, self.alpha)

        loss_s = source_loss_c1 + source_loss_c2
        loss_disc = discrepancy(self.alpha, s_pred1, s_pred2,
                                t_pred1, t_pred2)

        loss_alpha = mu * loss_s + loss_disc + lam_alpha * torch.norm(self.alpha, p=2)
        loss_alpha.backward(retain_graph=True)

        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.alpha, clip)

        # optim step
        self.opt_alpha.step()
        self.reset_grad()

        # Normalization (||alpha||_1 = 1)
        with torch.no_grad():
            self.alpha.clamp_(1 / (self.n_sources * 10), 1 - 1 / (self.n_sources * 10))
            self.alpha.div_(torch.norm(F.relu(self.alpha), p=1))


    def predict(self, x):
        z = x.clone()
        for hidden in self.feature_extractor:
            if hidden.__class__.__name__ == 'LSTM' and hidden.input_size == self.input_dim:
                z = torch.transpose(z, -1, 1)
                z = hidden(z)[0]
            elif hidden.__class__.__name__ == 'LSTM':
                z = hidden(z)[0]
            else:
                z = hidden(z)

        c1 = z.clone()
        for hidden in self.c1:
            c1 = hidden(c1)

        c2 = z.clone()
        for hidden in self.c2:
            c2 = hidden(c2)

        out_ensemble = c1 + c2

        return c1, c2, out_ensemble

    def clamp(self, x):
        return torch.clamp_(x, self.min_pred, self.max_pred)

    def compute_loss(self, s_pred1, s_pred2, t_pred1, t_pred2, ys):

        source_loss_c1 = self.weighted_loss(s_pred1, ys, self.alpha).item()
        source_loss_c2 = self.weighted_loss(s_pred2, ys, self.alpha).item()
        source_loss = source_loss_c1 + source_loss_c2

        disc = discrepancy(self.alpha, s_pred1, s_pred2,
                           t_pred1, t_pred2).item()

        return source_loss, disc

    def perform_target_dropout(self, Xt, cls, mode):
        tx = Xt.clone()
        del Xt
        torch.cuda.empty_cache()

        if mode == 'Feature_extractor_only':
            for hidden in self.feature_extractor:
                if hidden.__class__.__name__ == 'LSTM' and hidden.input_size == self.input_dim:
                    tx = torch.transpose(tx, -1, 1)
                    tx = [hidden(tx)[0] for i in range(self.num_dropout)]

                elif hidden.__class__.__name__ == 'LSTM':
                    tx = [hidden(tx)[0] for i in range(self.num_dropout)]
                else:
                    tx = [hidden(tx) for i in range(self.num_dropout)]

            for hidden in cls:
                for i in range(self.num_dropout):
                    tx = [hidden(tx[i])]

        elif mode == 'Classifier_only':
            for hidden in self.feature_extractor:
                if hidden.__class__.__name__ == 'LSTM' and hidden.input_size == self.input_dim:
                    tx = torch.transpose(tx, -1, 1)
                    tx = hidden(tx)[0]

                elif hidden.__class__.__name__ == 'LSTM':
                    tx = hidden(tx)[0]
                else:
                    tx = hidden(tx)

            for hidden in cls:
                tx = [hidden(tx) for i in range(self.num_dropout)]

        elif mode == 'Combine':
            tx = [self.feature_cls(tx, cls) for i in range(self.num_dropout)]

        else:
            NotImplementedError

        if len(tx) != 0:
            tx = torch.stack(tx)
            tx_mean = tx.mean(axis=0)
            unc = tx.std(axis=0)

            return tx_mean, unc

    def feature_cls(self, Xt, cls):
        tx = Xt.clone()
        del Xt
        torch.cuda.empty_cache()

        for hidden in self.feature_extractor:
            if hidden.__class__.__name__ == 'LSTM' and hidden.input_size == self.input_dim:
                tx = torch.transpose(tx, -1, 1)
                tx = hidden(tx)[0]
            elif hidden.__class__.__name__ == 'LSTM':
                tx = hidden(tx)[0]
            else:
                tx = hidden(tx)
        for hidden in cls:
            tx = hidden(tx)
        return tx

    def L1_norm(self, x):
        x_norm = torch.abs(x)
        x_norm = torch.sum(x_norm)
        return x_norm

    def MinMaxScaler(self, x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))

