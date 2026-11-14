"""
task1_link.py -- Task 1 experiments: conventional end-to-end text link.

(a) validate the simulated uncoded BER against theory
(b) coded BER and BLER versus SNR for the six MCS (AWGN and Rayleigh)
(c) text quality: BLEU, word accuracy, sentence similarity (+ example garbled text)
(d) look-up tables (LUT) of BLER / similarity versus SNR and the AMC rule
    (highest-rate MCS with BLER <= 10 %)

All raw numbers are stored in results/task1_*.npz|json; figures are produced later by
make_figures.py (so plots can be restyled without re-running the simulations).
Run:   python task1_link.py
"""
import os, json, time
import numpy as np
from multiprocessing import Pool
from config import *
from corpus import load_corpus
from source_coding import CharHuffman
import link
from modem import ber_qam_awgn, ber_qam_rayleigh
from text_metrics import bleu, word_accuracy, SentenceEmbedder

# ----------------------------------------------------------------------------- (a)
def task1a_validate_ber():
    """Monte-Carlo uncoded BER of Gray QPSK/16QAM/64QAM vs exact theory, AWGN and Rayleigh."""
    rng = np.random.default_rng(SEED)
    out = dict(ebn0_db=np.arange(0.0, 30.1, 2.0))
    for chan in ("awgn", "rayleigh"):
        for m in (2, 4, 6):
            sim, th = [], []
            for ebn0 in out["ebn0_db"]:
                esn0 = ebn0 + 10 * np.log10(m)                 # Es/N0 = m * Eb/N0 (uncoded)
                sim.append(link.uncoded_ber(m, esn0, BER_VALID_BITS, rng, chan))
                g = 10 ** (esn0 / 10)
                th.append(ber_qam_awgn(g, m)[0] if chan == "awgn" else ber_qam_rayleigh(g, m)[0])
            out[f"{chan}_m{m}_sim"], out[f"{chan}_m{m}_theory"] = np.array(sim), np.array(th)
    np.savez(os.path.join(RES_DIR, "task1a_ber_validation.npz"), **out)
    # report the worst relative deviation where the BER is statistically reliable
    worst = 0.0
    for k in out:
        if k.endswith("_sim"):
            s, t = out[k], out[k.replace("_sim", "_theory")]
            ok = t > 2e-5 if not QUICK else t > 1e-3
            worst = max(worst, float(np.max(np.abs(s[ok] - t[ok]) / t[ok])))
    print(f"[1a] max relative deviation sim vs theory (reliable points): {worst:.3f}")
    return worst


# ----------------------------------------------------------------------------- (b,c)
_G = {}


def _init():
    tr, te = load_corpus()
    _G["huff"] = CharHuffman(tr)
    _G["test"] = te[:LINK_NFRAMES]
    _G["emb"] = SentenceEmbedder(tr)


def _sweep(args):
    """One worker job: one (channel, MCS) over the whole SNR grid."""
    chan, k = args
    mcs = MCS_TABLE[k]
    huff, sents, emb = _G["huff"], _G["test"], _G["emb"]
    grid = SNR_GRID_AWGN if chan == "awgn" else SNR_GRID_RAYL
    tx = link.TxFrames(huff, sents, mcs)
    rng = np.random.default_rng(SEED + 1000 * k + (7 if chan == "rayleigh" else 0))
    res = {n: [] for n in ("ber", "bler", "bleu", "wacc", "sim")}
    samples = []
    t0 = time.time()
    for snr in grid:
        r = link.run(tx, snr, rng, chan=chan, want_text=True)
        res["ber"].append(r["bit_errors"].sum() / tx.L.sum())
        res["bler"].append(r["err_frames"].mean())
        b, w, s = [], [], []
        for i, (ref, hyp) in enumerate(zip(sents, r["text"])):
            if hyp == ref:
                b.append(1.0); w.append(1.0); s.append(1.0)         # perfect recovery
            else:
                b.append(bleu(ref, hyp)); w.append(word_accuracy(ref, hyp)); s.append(emb.sim(ref, hyp))
        res["bleu"].append(np.mean(b)); res["wacc"].append(np.mean(w)); res["sim"].append(np.mean(s))
        samples.append([r["text"][i] for i in range(3)])
    print(f"[1b] {chan:8s} {mcs['name']:11s} done in {time.time()-t0:.0f}s", flush=True)
    out = {n: np.array(v) for n, v in res.items()}
    out.update(snr=np.array(grid), samples=samples,
               avg_info_bits=float(tx.L.mean()), avg_tx_bits=float(tx.n_tx_bits.mean()),
               avg_symbols=float(tx.n_sym.mean()))
    return chan, k, out


def task1bc_sweep():
    jobs = [(c, k) for c in ("awgn", "rayleigh") for k in range(len(MCS_TABLE))]
    n_workers = int(os.environ.get("EX1_WORKERS", "2"))     # 1 = serial (Windows/macOS/notebooks)
    if n_workers <= 1:
        _init()
        results = [_sweep(j) for j in jobs]
    else:
        with Pool(n_workers, initializer=_init) as p:
            results = p.map(_sweep, jobs, chunksize=1)
    store = {}
    for chan, k, out in results:
        store.setdefault(chan, {})[k] = out
    # serialise
    ser = {}
    for chan in store:
        ser[chan] = {}
        for k, o in store[chan].items():
            ser[chan][MCS_TABLE[k]["name"]] = {n: (v.tolist() if isinstance(v, np.ndarray) else v)
                                               for n, v in o.items()}
    with open(os.path.join(RES_DIR, "task1_sweeps.json"), "w") as f:
        json.dump(ser, f)
    return ser


# ----------------------------------------------------------------------------- (d)
def build_lut_and_amc(ser):
    """
    LUT = BLER and similarity versus SNR per MCS on the AWGN grid (0.5 dB steps).
    Threshold of an MCS = smallest SNR from which BLER stays <= 10 % (linear
    interpolation of log10(BLER) between grid points).  AMC picks the highest
    spectral efficiency among the MCS whose threshold <= SNR; below the lowest
    threshold the link is in outage (no transmission).
    Why AWGN LUT in Task 3?  gamma_{u,n} already contains the Rayleigh fading of the
    RB, and the channel is constant over the RB, so the instantaneous-SNR AWGN curve
    is the right look-up.
    """
    snr = np.array(ser["awgn"][MCS_TABLE[0]["name"]]["snr"])
    bler = np.array([ser["awgn"][m["name"]]["bler"] for m in MCS_TABLE])        # (6, nsnr)
    sim = np.array([ser["awgn"][m["name"]]["sim"] for m in MCS_TABLE])
    bleu_ = np.array([ser["awgn"][m["name"]]["bleu"] for m in MCS_TABLE])
    thr = []
    for i in range(len(MCS_TABLE)):
        ok = bler[i] <= BLER_TARGET
        # last index where the target is violated -> threshold is the next grid point
        bad = np.flatnonzero(~ok)
        j = (bad[-1] + 1) if len(bad) else 0
        if j >= len(snr):
            thr.append(np.inf); continue
        if j == 0:
            thr.append(float(snr[0])); continue
        # interpolate in log domain between grid points j-1 (BLER>target) and j (<= target)
        y0, y1 = np.log10(max(bler[i, j - 1], 1e-6)), np.log10(max(bler[i, j], 1e-6))
        t = (np.log10(BLER_TARGET) - y0) / (y1 - y0) if y1 != y0 else 1.0
        thr.append(float(snr[j - 1] + t * (snr[j] - snr[j - 1])))
    thr = np.array(thr)
    eta = np.array([m["eta"] for m in MCS_TABLE])
    np.savez(os.path.join(DATA_DIR, "amc_lut.npz"), snr_db=snr, bler=bler, sim=sim, bleu=bleu_,
             thr_db=thr, eta=eta,
             avg_info_bits=ser["awgn"][MCS_TABLE[0]["name"]]["avg_info_bits"])
    # CSV versions for humans
    with open(os.path.join(RES_DIR, "lut_bler_similarity.csv"), "w") as f:
        f.write("snr_db," + ",".join(f"BLER[{m['name']}]" for m in MCS_TABLE) + "," +
                ",".join(f"SIM[{m['name']}]" for m in MCS_TABLE) + "\n")
        for j, s in enumerate(snr):
            f.write(f"{s:.1f}," + ",".join(f"{bler[i, j]:.4f}" for i in range(6)) + "," +
                    ",".join(f"{sim[i, j]:.4f}" for i in range(6)) + "\n")
    with open(os.path.join(RES_DIR, "amc_thresholds.json"), "w") as f:
        json.dump({m["name"]: dict(threshold_snr_db=float(t), eta=m["eta"]) for m, t in zip(MCS_TABLE, thr)}, f, indent=1)
    print("[1d] AMC thresholds (SNR dB for BLER<=10%):",
          {m["name"]: round(float(t), 2) for m, t in zip(MCS_TABLE, thr)})
    return thr


class AMC:
    """Adaptive modulation and coding built from the saved LUT (used by Task 3)."""

    def __init__(self, path=os.path.join(DATA_DIR, "amc_lut.npz")):
        d = np.load(path)
        self.snr, self.bler, self.sim = d["snr_db"], d["bler"], d["sim"]
        self.thr, self.eta, self.L = d["thr_db"], d["eta"], float(d["avg_info_bits"])

    def select(self, snr_db):
        """snr_db array -> (mcs index or -1 for outage, spectral efficiency [info bit/symbol])."""
        snr_db = np.asarray(snr_db, float)
        feas = snr_db[..., None] >= self.thr                  # (..., 6)
        score = np.where(feas, self.eta, -1.0)
        idx = np.argmax(score, axis=-1)
        none = ~feas.any(axis=-1)
        idx = np.where(none, -1, idx)
        eta = np.where(none, 0.0, self.eta[np.maximum(idx, 0)])
        return idx, eta

    def bler_at(self, idx, snr_db):
        """BLER of the chosen MCS at the actual SNR (interpolated from the LUT)."""
        snr_db = np.asarray(snr_db, float)
        out = np.ones_like(snr_db)
        for i in range(len(self.eta)):
            sel = idx == i
            if sel.any():
                out[sel] = np.interp(snr_db[sel], self.snr, self.bler[i])
        return out


def main():
    t0 = time.time()
    task1a_validate_ber()
    ser = task1bc_sweep()
    build_lut_and_amc(ser)
    print(f"[task1] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
