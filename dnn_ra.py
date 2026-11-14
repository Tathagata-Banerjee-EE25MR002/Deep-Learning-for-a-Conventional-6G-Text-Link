"""
dnn_ra.py -- unsupervised learning-to-optimize (L2O) DNN for multi-user resource allocation.

What it learns.  A permutation-equivariant deep network (see EqLayer) maps the channel-gain matrix (U x N) and the
power budget Pmax to
    * an RB assignment: for each RB n a soft-max over the U users (hard arg-max at test
      time -> exactly one user per RB, so  sum_u x_{u,n} = 1  always holds), and
    * a power split: p_n = Pmax * softmax(logits)_n   ->  sum_n p_n = Pmax  and p_n >= 0
      hold BY CONSTRUCTION (the power constraint never needs to be penalised).

UNSUPERVISED: there are no optimal labels (the MILP would be too slow to label a large
set).  The network is trained directly on the objective of the problem with
    loss = - (sum rate [Mbit/s])  +  rho * sum_u max(0, Rmin - R_u)^2   (Rmin in Mbit/s)
i.e. negative sum rate plus a quadratic penalty on unmet minimum rates, as the task
prescribes.  Fresh random channel drops and a random Pmax in [15,40] dBm are generated
at EVERY iteration, so there is no over-fitting and one network covers the whole sweep.

Discrete decision inside back-propagation: a hard arg-max has zero gradient, so we use the
straight-through estimator (ST): the FORWARD pass uses the one-hot arg-max assignment (the
loss is therefore evaluated on the very allocation that will be deployed) while the
BACKWARD pass uses the gradient of the soft-max probabilities.

Hybrid variant ("DNN + WF"): keep the DNN's assignment, throw away its power split and
re-compute the powers with the KKT water-filling of resource_alloc.kkt_waterfill.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from config import *
import resource_alloc as ra

torch.set_num_threads(2)
P_LO, P_HI = float(PMAX_DBM_SWEEP[0]), float(PMAX_DBM_SWEEP[-1])
RHO = 40.0                                           # weight of the minimum-rate penalty [1/(Mbit/s)^2]


def features(a, p_dbm):
    """
    a: (B,U,N) SNR-per-watt.  Four input channels per (user, RB) cell, all derived from
    the equal-power SNR in dB (standardised with FIXED constants so that training and
    test use identical scaling):
        c0 = (snr_dB - 20)/12                   absolute channel quality
        c1 = (snr_dB - max_u snr_dB)/10         how far the user is from the best user on this RB
        c2 = (snr_dB - mean_n snr_dB)/10        how good this RB is relative to the user's average RB
        c3 = (Pmax_dBm - 27.5)/12.5             power budget (same for all cells)
    Returns (B,U,N,4).
    """
    P_w = 10 ** ((p_dbm - 30.0) / 10.0)
    snr = 10.0 * torch.log10(a * (P_w[:, None, None] / N) + 1e-12)
    c0 = (snr - 20.0) / 12.0
    c1 = (snr - snr.max(1, keepdim=True).values) / 10.0
    c2 = (snr - snr.mean(2, keepdim=True)) / 10.0
    c3 = ((p_dbm - 27.5) / 12.5)[:, None, None].expand_as(snr)
    return torch.stack([c0, c1, c2, c3], dim=-1)


class EqLayer(nn.Module):
    """
    Permutation-EQUIVARIANT layer over the user axis and the RB axis (DeepSets on a
    matrix).  Each cell (u,n) combines: its own state, the mean over the user's RBs, the
    mean and max over the users of its RB, and the global mean:
        h'_{un} = ReLU( W0 h_{un} + W1 mean_n h_{un} + W2 mean_u h_{un} + W3 max_u h_{un} + W4 mean_{un} h + b )
    (identical to a linear layer on the concatenation of these five tensors, but the
    pooled terms are transformed BEFORE broadcasting, which is ~4x cheaper).
    Because the same weights are shared by all users and RBs the network is tiny
    (~25k parameters), needs no re-learning of the same rule at every (u,n) position, and
    its output permutes consistently when users or RBs are permuted.
    """
    def __init__(self, din, dout):
        super().__init__()
        self.w0 = nn.Linear(din, dout)
        self.w1 = nn.Linear(din, dout, bias=False)
        self.w2 = nn.Linear(din, dout, bias=False)
        self.w3 = nn.Linear(din, dout, bias=False)
        self.w4 = nn.Linear(din, dout, bias=False)

    def forward(self, h):                                       # (B,U,N,d)
        return torch.relu(self.w0(h)
                          + self.w1(h.mean(2, keepdim=True))            # context of the user (over its RBs)
                          + self.w2(h.mean(1, keepdim=True))            # context of the RB (over users)
                          + self.w3(h.max(1, keepdim=True).values)      # best user on this RB
                          + self.w4(h.mean((1, 2), keepdim=True)))      # global context


class RANet(nn.Module):
    """Features -> 4 equivariant layers -> (assignment logit per cell, power logit per RB)."""
    def __init__(self, hid=48, layers=4):
        super().__init__()
        dims = [4] + [hid] * layers
        self.layers = nn.ModuleList([EqLayer(dims[i], dims[i + 1]) for i in range(layers)])
        self.assign = nn.Linear(hid, 1)
        self.power = nn.Linear(2 * hid, 1)

    def forward(self, a, p_dbm):
        h = features(a, p_dbm)
        for layer in self.layers:
            h = layer(h)
        z = self.assign(h).squeeze(-1)                           # (B,U,N)
        pooled = torch.cat([h.mean(1), h.max(1).values], dim=-1)  # (B,N,2*hid) pooled over users
        w = self.power(pooled).squeeze(-1)                       # (B,N)
        return z, w


def allocate_torch(z, w, a, p_dbm):
    """Soft/hard assignment with straight-through gradient and power split -> (X, p)."""
    soft = torch.softmax(z, dim=1)                              # over users, per RB
    hard = torch.zeros_like(soft).scatter_(1, soft.argmax(1, keepdim=True), 1.0)
    X = hard + soft - soft.detach()                             # forward: hard, backward: soft
    P_w = 10 ** ((p_dbm - 30.0) / 10.0)
    p = P_w[:, None] * torch.softmax(w, dim=1)
    return X, p


def loss_fn(X, p, a, rho=RHO):
    r = B_RB * torch.log2(1.0 + p[:, None, :] * a) / 1e6        # (B,U,N) Mbit/s
    R = (X * r).sum(2)                                          # (B,U)
    sum_rate = R.sum(1)
    pen = (torch.relu(R_MIN / 1e6 - R) ** 2).sum(1)
    return (-sum_rate + rho * pen).mean(), sum_rate.mean().item(), pen.mean().item()


def train(iters=RA_TRAIN_ITERS, batch=RA_BATCH, seed=SEED, path=os.path.join(DATA_DIR, "ra_dnn.pt")):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 5)
    net = RANet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=2e-5)
    hist = []
    t0 = time.time()
    for it in range(iters):
        a = torch.tensor(ra.generate_drops(rng, batch), dtype=torch.float32)
        p_dbm = torch.empty(batch).uniform_(P_LO, P_HI)
        z, w = net(a, p_dbm)
        X, p = allocate_torch(z, w, a, p_dbm)
        loss, sr, pen = loss_fn(X, p, a)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
        if it % 100 == 0 or it == iters - 1:
            hist.append((it, float(loss.detach()), sr, pen))
        if it % 2000 == 0:
            print(f"[3] iter {it:6d} loss {float(loss):8.3f} sum-rate {sr:6.2f} Mbit/s penalty {pen:6.3f}  ({time.time()-t0:.0f}s)", flush=True)
    torch.save(net.state_dict(), path)
    print(f"[3] DNN trained in {time.time()-t0:.0f}s")
    return net, hist


@torch.no_grad()
def dnn_allocate(net, a_np, p_dbm_value):
    """Inference: returns (X hard, p) as numpy for a batch of drops at one Pmax."""
    a = torch.tensor(a_np, dtype=torch.float32)
    p_dbm = torch.full((a.shape[0],), float(p_dbm_value))
    z, w = net(a, p_dbm)
    X, p = allocate_torch(z, w, a, p_dbm)
    return X.numpy().astype(np.float64), p.numpy().astype(np.float64)


def load(path=os.path.join(DATA_DIR, "ra_dnn.pt")):
    net = RANet(); net.load_state_dict(torch.load(path)); net.eval(); return net


if __name__ == "__main__":
    import sys
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    net, hist = train(iters=iters, path=os.path.join(DATA_DIR, "ra_dnn_test.pt"))
    rng = np.random.default_rng(123)
    a = ra.generate_drops(rng, 100)
    for pdbm in (20, 30, 40):
        P = ra.dbm_to_w(pdbm)
        X, p = dnn_allocate(net, a, pdbm)
        print(pdbm, "DNN", ra.metrics(ra.user_rates(X, a, p)))
        p2, _ = ra.kkt_waterfill(a, X, P)
        print(pdbm, "DNN+WF", ra.metrics(ra.user_rates(X, a, p2)))
        Xm, pm = ra.milp_kkt_wf(a, P)
        print(pdbm, "MILP+WF", ra.metrics(ra.user_rates(Xm, a, pm)))
