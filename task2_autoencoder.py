"""
task2_autoencoder.py -- Task 2: deep-learning physical layer (end-to-end autoencoder).

Idea (O'Shea & Hoydis, 2017).  Treat transmitter + channel + receiver as ONE neural
network that is trained end to end:
    message s in {0..15} (k = 4 bits)
        -> encoder DNN -> n real channel uses x (power-normalised)
        -> channel (AWGN, or Rayleigh fading with receiver CSI)
        -> decoder DNN -> soft-max over the 16 messages
Loss = cross entropy between the soft-max output and the transmitted message, so the
network jointly invents a CODEBOOK (encoder) and a DECODER.  No bit-wise structure,
modulation or code is imposed.

Experiments
-----------
A. (7,4) AE in AWGN  vs  uncoded BPSK,  Hamming(7,4) + hard decision,
                          Hamming(7,4) + maximum-likelihood (soft) decoding.
B. (2,4) AE in AWGN  (16 messages on ONE complex symbol)  vs  16-QAM
   -> learned constellation vs 16-QAM.
C. (7,4) AE in Rayleigh fading with CSI at the receiver  vs the same baselines.

Conventions
-----------
* Eb/N0 is per INFORMATION bit.  With code rate R = k/n and real signalling
  (unit energy per real channel use on average) the noise std per real channel use is
        sigma = sqrt( 1 / (2 R Eb/N0) ) .
* Energy constraint: average over the 16 codewords  E||x||^2 = n  (the whole code-
  book is rescaled).  This is the average-power constraint of the system model, a
  *weaker* constraint than per-codeword constant energy, so the AE is free to beat
  a constant-energy code if such a code exists.
* Fading (C): h_i ~ Rayleigh, E[h^2]=1, independent for every channel use (ideal
  interleaving), receiver knows h_i (CSI); the decoder gets the vector [y, h].
* BLER = message (4-bit block) error probability.
"""
import os, json, time, itertools
import numpy as np
import torch
import torch.nn as nn
from scipy.special import erfc
from config import *

torch.set_num_threads(2)
DEV = "cpu"
K_BITS = 4
M_MSG = 1 << K_BITS


def Qf(x):
    return 0.5 * erfc(x / np.sqrt(2))


# ----------------------------------------------------------------------------- model
class AE(nn.Module):
    def __init__(self, n, hid=64, csi=False, const_energy=False):
        super().__init__()
        self.n, self.csi, self.const_energy = n, csi, const_energy
        self.enc = nn.Sequential(nn.Linear(M_MSG, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, n))
        din = 2 * n if csi else n
        self.dec = nn.Sequential(nn.Linear(din, hid * 2), nn.ReLU(), nn.Linear(hid * 2, hid), nn.ReLU(),
                                 nn.Linear(hid, M_MSG))
        self.register_buffer("eye", torch.eye(M_MSG))

    def codebook(self):
        """(16, n) codebook.  Two energy constraints are supported:
        * average  (default): rescaled so that the AVERAGE energy over the 16 codewords is n,
        * constant (const_energy=True): EVERY codeword has energy exactly n (all points on a
          sphere), the constraint of the original O'Shea-Hoydis autoencoder; Hamming(7,4)+BPSK
          satisfies it too, so this is the like-for-like comparison."""
        z = self.enc(self.eye)
        if self.const_energy:
            return z * torch.sqrt(self.n / (z ** 2).sum(1, keepdim=True))
        return z * torch.sqrt(self.n / (z ** 2).sum(1).mean())

    def channel(self, x, sigma, fading):
        if fading:
            h = torch.sqrt(torch.randn_like(x) ** 2 + torch.randn_like(x) ** 2) / np.sqrt(2.0)   # Rayleigh, E h^2 = 1
            y = h * x + sigma * torch.randn_like(x)
            return torch.cat([y, h], dim=-1)
        return x + sigma * torch.randn_like(x)

    def forward(self, idx, sigma, fading=False):
        x = self.codebook()[idx]
        return self.dec(self.channel(x, sigma, fading))


def sigma_of(ebn0_db, n, k=K_BITS):
    R = k / n
    return np.sqrt(1.0 / (2 * R * 10 ** (np.asarray(ebn0_db) / 10)))


def train_ae(n, fading=False, train_range=(0.0, 8.0), iters=AE_TRAIN_ITERS, seed=SEED, hid=64, tag="",
             const_energy=False):
    """Adam + cosine learning-rate decay; the training SNR is drawn per example in train_range."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = AE(n, hid=hid, csi=fading, const_energy=const_energy)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=1e-4)
    lossf = nn.CrossEntropyLoss()
    hist = []
    t0 = time.time()
    for it in range(iters):
        idx = torch.randint(0, M_MSG, (AE_BATCH,))
        eb = torch.empty(AE_BATCH, 1).uniform_(*train_range)
        sigma = torch.sqrt(1.0 / (2 * (K_BITS / n) * 10 ** (eb / 10)))
        loss = lossf(model(idx, sigma, fading), idx)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 200 == 0 or it == iters - 1:
            hist.append((it, float(loss.detach())))
    print(f"[2] trained AE(n={n}, fading={fading}) {tag} in {time.time()-t0:.0f}s, final loss {hist[-1][1]:.4f}", flush=True)
    return model, hist


@torch.no_grad()
def ae_bler(model, ebn0_db, n_msgs, fading=False, seed=0):
    torch.manual_seed(seed)
    sigma = float(sigma_of(ebn0_db, model.n))
    errs, done = 0, 0
    while done < n_msgs:
        b = min(500_000, n_msgs - done)
        idx = torch.randint(0, M_MSG, (b,))
        pred = model(idx, sigma, fading).argmax(1)
        errs += int((pred != idx).sum()); done += b
    return errs / n_msgs


# ----------------------------------------------------------------------------- baselines
def hamming74_codebook():
    """Systematic Hamming(7,4): codeword = [d1 d2 d3 d4 p1 p2 p3] (any equivalent code gives the same curves)."""
    P = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    msgs = np.array(list(itertools.product([0, 1], repeat=4)))
    cw = np.concatenate([msgs, (msgs @ P) % 2], axis=1)
    return cw.astype(np.float32)                              # (16, 7) in {0,1}


@torch.no_grad()
def hamming_blers(ebn0_db, n_msgs, fading=False, seed=0):
    """
    Simulated BLER of Hamming(7,4)+BPSK for hard decision and ML (soft) decoding,
    and of uncoded BPSK blocks of 4 bits.  Same noise/fading statistics as the AE.
    """
    g = torch.Generator().manual_seed(seed)
    cw = torch.tensor(hamming74_codebook())
    cwb = 1 - 2 * cw                                           # BPSK: bit 0 -> +1
    sigma = float(sigma_of(ebn0_db, 7))
    sigma_u = float(sigma_of(ebn0_db, 4))                      # uncoded: n = k = 4, R = 1
    e_h = e_m = e_u = 0; done = 0
    while done < n_msgs:
        b = min(400_000, n_msgs - done)
        idx = torch.randint(0, M_MSG, (b,), generator=g)
        x = cwb[idx]
        noise = sigma * torch.randn(b, 7, generator=g)
        if fading:
            h = torch.sqrt(torch.randn(b, 7, generator=g) ** 2 + torch.randn(b, 7, generator=g) ** 2) / np.sqrt(2.0)
        else:
            h = torch.ones(b, 7)
        y = h * x + noise
        # ML with CSI: min ||y - h c||^2  <=>  max  sum h y c   (all |c_i| = 1)
        e_m += int((torch.argmax((h * y) @ cwb.T, 1) != idx).sum())
        # hard decision: bit decisions, then nearest codeword in Hamming distance (= syndrome decoding)
        hard = (y < 0).float()
        dH = (hard[:, None, :] != cw[None, :, :]).sum(2)
        e_h += int((torch.argmin(dH, 1) != idx).sum())
        # uncoded: 4 independent BPSK bits, coherent detection
        ub = torch.randint(0, 2, (b, 4), generator=g).float()
        hu = (torch.sqrt(torch.randn(b, 4, generator=g) ** 2 + torch.randn(b, 4, generator=g) ** 2) / np.sqrt(2.0)) if fading else torch.ones(b, 4)
        yu = hu * (1 - 2 * ub) + sigma_u * torch.randn(b, 4, generator=g)
        e_u += int(((yu < 0).float() != ub).any(1).sum())
        done += b
    return e_h / n_msgs, e_m / n_msgs, e_u / n_msgs


def uncoded_bpsk_theory(ebn0_db, fading=False, k=4):
    g = 10 ** (np.asarray(ebn0_db) / 10)
    p = 0.5 * (1 - np.sqrt(g / (1 + g))) if fading else Qf(np.sqrt(2 * g))
    return 1 - (1 - p) ** k


def hamming_hard_theory(ebn0_db, fading=False):
    """BLER = 1 - [(1-p)^7 + 7 p (1-p)^6]  with p the raw BPSK bit error probability at Eb/N0 * 4/7."""
    g = 10 ** (np.asarray(ebn0_db) / 10) * 4 / 7
    p = 0.5 * (1 - np.sqrt(g / (1 + g))) if fading else Qf(np.sqrt(2 * g))
    return 1 - ((1 - p) ** 7 + 7 * p * (1 - p) ** 6)


# ----------------------------------------------------------------------------- (2,4) vs 16-QAM
def qam16_points_real2():
    """Gray-free 16-QAM in R^2 scaled so that E||x||^2 = n = 2 (same convention as the AE)."""
    lv = np.array([-3, -1, 1, 3], float)
    pts = np.array([(a, b) for a in lv for b in lv])
    return pts * np.sqrt(2.0 / np.mean((pts ** 2).sum(1)))


@torch.no_grad()
def qam16_ser(ebn0_db, n_msgs, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.tensor(qam16_points_real2(), dtype=torch.float32)
    sigma = float(sigma_of(ebn0_db, 2))
    errs, done = 0, 0
    while done < n_msgs:
        b = min(500_000, n_msgs - done)
        idx = torch.randint(0, M_MSG, (b,), generator=g)
        y = pts[idx] + sigma * torch.randn(b, 2, generator=g)
        errs += int((torch.cdist(y, pts).argmin(1) != idx).sum()); done += b
    return errs / n_msgs


@torch.no_grad()
def ae_ml_ser(model, ebn0_db, n_msgs, seed=0):
    """ML (nearest-neighbour) detection on the AE's *own* codebook (any n): the lower bound on the
    error probability of ANY decoder for that codebook, so the learned decoder can only approach it."""
    g = torch.Generator().manual_seed(seed)
    pts = model.codebook()
    sigma = float(sigma_of(ebn0_db, model.n))
    errs, done = 0, 0
    while done < n_msgs:
        b = min(500_000, n_msgs - done)
        idx = torch.randint(0, M_MSG, (b,), generator=g)
        y = pts[idx] + sigma * torch.randn(b, model.n, generator=g)
        errs += int((torch.cdist(y, pts).argmin(1) != idx).sum()); done += b
    return errs / n_msgs


def min_dist2(cb):
    d = torch.cdist(cb, cb) ** 2 + torch.eye(len(cb)) * 1e9
    return float(d.min())



def add_const_energy(res):
    """Extra AEs with the per-codeword (constant) energy constraint, (7,4) AWGN and Rayleigh+CSI."""
    NM = AE_EVAL_MSGS
    A, C = res["awgn_7_4"], res["rayleigh_7_4"]
    ae, _ = train_ae(7, fading=False, train_range=(0.0, 8.0), const_energy=True, tag="(const. energy)")
    A["ae_const"] = [ae_bler(ae, e, NM) for e in A["ebn0"]]
    cb = ae.codebook().detach()
    A["ae_const_min_dist2"] = min_dist2(cb)
    aef, _ = train_ae(7, fading=True, train_range=(4.0, 16.0), hid=96, const_energy=True, tag="(Rayleigh+CSI, const. energy)")
    C["ae_const"] = [ae_bler(aef, e, NM, fading=True) for e in C["ebn0"]]
    return res

# ----------------------------------------------------------------------------- main
def main():
    res = {}
    grid = [float(x) for x in AE_EBN0_GRID]
    NM = AE_EVAL_MSGS
    t0 = time.time()

    # --- A: (7,4) AWGN
    ae74, hist74 = train_ae(7, fading=False, train_range=(0.0, 8.0))
    A = dict(ebn0=grid, ae=[], ham_hard=[], ham_ml=[], uncoded=[], ham_hard_theory=[], uncoded_theory=[])
    for e in grid:
        A["ae"].append(ae_bler(ae74, e, NM))
        hh, hm, uu = hamming_blers(e, NM)
        A["ham_hard"].append(hh); A["ham_ml"].append(hm); A["uncoded"].append(uu)
        A["ham_hard_theory"].append(float(hamming_hard_theory(e)))
        A["uncoded_theory"].append(float(uncoded_bpsk_theory(e)))
    A["ae_ml_own"] = [ae_ml_ser(ae74, e, NM) for e in grid]          # ML on the AE's own codebook
    cb = ae74.codebook().detach()
    cwb = torch.tensor(1 - 2 * hamming74_codebook())
    A["ae_min_dist2"] = min_dist2(cb)
    A["hamming_min_dist2"] = min_dist2(cwb)
    A["ae_codeword_energy_std"] = float((cb ** 2).sum(1).std())
    A["loss_hist"] = hist74
    res["awgn_7_4"] = A
    torch.save(ae74.state_dict(), os.path.join(DATA_DIR, "ae_7_4_awgn.pt"))

    # --- B: (2,4) AWGN
    ae24, hist24 = train_ae(2, fading=False, train_range=(4.0, 12.0), hid=64)
    gridB = [float(x) for x in np.arange(0.0, 14.1, 1.0)]
    B = dict(ebn0=gridB, ae=[], ae_ml=[], qam16=[])
    for e in gridB:
        B["ae"].append(ae_bler(ae24, e, NM)); B["ae_ml"].append(ae_ml_ser(ae24, e, NM)); B["qam16"].append(qam16_ser(e, NM))
    B["ae_points"] = ae24.codebook().detach().numpy().tolist()
    B["qam_points"] = qam16_points_real2().tolist()
    B["ae_min_dist2"] = min_dist2(ae24.codebook().detach())
    B["qam_min_dist2"] = min_dist2(torch.tensor(qam16_points_real2(), dtype=torch.float32))
    B["loss_hist"] = hist24
    res["awgn_2_4"] = B
    torch.save(ae24.state_dict(), os.path.join(DATA_DIR, "ae_2_4_awgn.pt"))

    # --- C: (7,4) Rayleigh with CSI
    aef, histf = train_ae(7, fading=True, train_range=(4.0, 16.0), hid=96, tag="(Rayleigh+CSI)")
    gridC = [float(x) for x in np.arange(0.0, 24.1, 2.0)]
    C = dict(ebn0=gridC, ae=[], ham_hard=[], ham_ml=[], uncoded=[], ham_hard_theory=[], uncoded_theory=[])
    for e in gridC:
        C["ae"].append(ae_bler(aef, e, NM, fading=True))
        hh, hm, uu = hamming_blers(e, NM, fading=True)
        C["ham_hard"].append(hh); C["ham_ml"].append(hm); C["uncoded"].append(uu)
        C["ham_hard_theory"].append(float(hamming_hard_theory(e, True)))
        C["uncoded_theory"].append(float(uncoded_bpsk_theory(e, True)))
    C["loss_hist"] = histf
    res["rayleigh_7_4"] = C
    torch.save(aef.state_dict(), os.path.join(DATA_DIR, "ae_7_4_rayleigh.pt"))

    add_const_energy(res)
    with open(os.path.join(RES_DIR, "task2_results.json"), "w") as f:
        json.dump(res, f)
    print(f"[task2] done in {time.time()-t0:.0f}s")
    # headline numbers
    i = grid.index(6.0) if 6.0 in grid else 8
    print(f"[2A] Eb/N0={grid[i]} dB: AE {A['ae'][i]:.2e} | Ham-ML {A['ham_ml'][i]:.2e} | Ham-hard {A['ham_hard'][i]:.2e} | uncoded {A['uncoded'][i]:.2e}")
    print(f"[2A] d_min^2: AE {A['ae_min_dist2']:.2f} vs Hamming {A['hamming_min_dist2']:.2f}; AE codeword energy std {A['ae_codeword_energy_std']:.3f}")
    print(f"[2B] d_min^2: AE {B['ae_min_dist2']:.3f} vs 16QAM {B['qam_min_dist2']:.3f}")


if __name__ == "__main__":
    import sys
    if "--extras" in sys.argv:          # only add the constant-energy variants to an existing result file
        p = os.path.join(RES_DIR, "task2_results.json")
        r = add_const_energy(json.load(open(p)))
        json.dump(r, open(p, "w"))
    else:
        main()
