"""
task3_ra.py -- Task 3 experiments: DNN-based multi-user resource allocation.

Compared methods (same random channel drops for every method and every Pmax):
  1 Round-robin + equal power              (benchmark)
  2 Strongest user + water-filling         (benchmark)
  3 MILP + equal power                     (benchmark, exact assignment)
  4 MILP + KKT water-filling (Rmin-aware)  (benchmark, "best known" reference)
  5 DNN (own assignment and power split)   (proposed, unsupervised L2O)
  6 DNN + KKT water-filling                (proposed hybrid)

Metrics: sum rate, % of users meeting Rmin, Jain's fairness index, 5th-percentile user
rate, computation time, and -- after applying the Task-1 AMC rule to every allocated RB --
the number of CORRECT sentences delivered per second.

Sentence throughput model.  An RB carries B = 180 ksymbol/s.  A user served on RB n with
SNR gamma_{u,n} = p_n a_{u,n} uses the AMC rule of Task 1 (highest-rate MCS whose AWGN
BLER <= 10 %; outage -> RB carries nothing).  A sentence needs on average S_mcs channel
symbols (measured in Task 1: Huffman bits + 6 tail bits, divided by bits/symbol x rate,
incl. modulation padding), so the RB delivers
        B / S_mcs  *  (1 - BLER_mcs(gamma))        correct sentences per second.
The BLER is read from the Task-1 LUT at the ACTUAL SNR (it is <= 10 % by construction and
much smaller well above the MCS threshold).  The AWGN LUT is the right one because
gamma_{u,n} already contains the Rayleigh fading of the RB, which is constant over it.
"""
import os, json, time
import numpy as np
import torch
from config import *
import resource_alloc as ra
import dnn_ra
from task1_link import AMC

METHODS = ["Round-robin + equal power", "Strongest user + water-filling", "MILP + equal power",
           "MILP + KKT water-filling", "DNN", "DNN + KKT water-filling"]


def sentences_per_second(X, a, p, amc, avg_sym):
    """Expected number of correct sentences per second (per drop) -> (D,) and per user (D,U)."""
    gamma = p[:, None, :] * a                                           # (D,U,N)
    snr_db = 10.0 * np.log10(np.maximum(gamma, 1e-30))
    used = (X > 0.5) & (p[:, None, :] > 0)
    idx, eta = amc.select(snr_db)                                       # (D,U,N)
    idx = np.where(used, idx, -1)
    bler = amc.bler_at(idx, snr_db)
    sym = np.where(idx >= 0, avg_sym[np.maximum(idx, 0)], np.inf)
    sps = np.where(idx >= 0, B_RB / sym * (1.0 - bler), 0.0)
    per_user = sps.sum(2)
    return per_user.sum(1), per_user


def single_instance_time(fn, a, P, n=30):
    """Mean wall-clock time per drop when drops are processed ONE AT A TIME (latency)."""
    t = time.perf_counter()
    for d in range(n):
        fn(a[d:d + 1], P)
    return (time.perf_counter() - t) / n


def main():
    t_all = time.time()
    amc = AMC()
    with open(os.path.join(RES_DIR, "task1_sweeps.json")) as f:
        sw = json.load(f)
    avg_sym = np.array([sw["awgn"][m["name"]]["avg_symbols"] for m in MCS_TABLE])

    path = os.path.join(DATA_DIR, "ra_dnn.pt")
    hist = None
    if os.path.exists(path) and os.environ.get("EX1_RETRAIN", "0") != "1":
        net = dnn_ra.load(path); print("[3] loaded trained DNN")
    else:
        net, hist = dnn_ra.train()
        net.eval()
    if hist is not None:
        json.dump(hist, open(os.path.join(RES_DIR, "task3_train_hist.json"), "w"))

    rng = np.random.default_rng(SEED + 77)
    a = ra.generate_drops(rng, RA_TEST_DROPS)
    D = a.shape[0]
    out = {m: {k: [] for k in ("sum_rate_mbps", "pct_users_rmin", "jain", "p5_rate_mbps", "sentences_per_s",
                               "sps_p5_user")} for m in METHODS}
    out["assign_agreement_dnn_vs_milp"] = []
    out["pmax_dbm"] = [float(x) for x in PMAX_DBM_SWEEP]
    for pdbm in PMAX_DBM_SWEEP:
        P = ra.dbm_to_w(pdbm)
        alloc = {}
        alloc[METHODS[0]] = ra.round_robin(a, P)
        alloc[METHODS[1]] = ra.strongest_wf(a, P)
        Xm, pm_eq = ra.milp_equal(a, P)                                  # MILP solved once, used twice
        alloc[METHODS[2]] = (Xm, pm_eq)
        alloc[METHODS[3]] = (Xm, ra.kkt_waterfill(a, Xm, P)[0])
        Xd, pd = dnn_ra.dnn_allocate(net, a, pdbm)
        alloc[METHODS[4]] = (Xd, pd)
        alloc[METHODS[5]] = (Xd, ra.kkt_waterfill(a, Xd, P)[0])
        for name in METHODS:
            X, p = alloc[name]
            R = ra.user_rates(X, a, p)
            mt = ra.metrics(R)
            sps, sps_u = sentences_per_second(X, a, p, amc, avg_sym)
            for k in ("sum_rate_mbps", "pct_users_rmin", "jain", "p5_rate_mbps"):
                out[name][k].append(mt[k])
            out[name]["sentences_per_s"].append(float(sps.mean()))
            out[name]["sps_p5_user"].append(float(np.percentile(sps_u, 5)))
        out["assign_agreement_dnn_vs_milp"].append(float(np.mean((Xd * Xm).sum(1))))   # fraction of RBs with same user
        print(f"[3] Pmax {pdbm:5.1f} dBm | " + " | ".join(
            f"{n.split()[0][:6]}{n.split()[-1][:3]} {out[n]['sum_rate_mbps'][-1]:5.1f}Mb {out[n]['pct_users_rmin'][-1]:5.1f}% {out[n]['sentences_per_s'][-1]:6.1f}s/s"
            for n in METHODS), flush=True)

    # ---------------- computation time at Pmax = 30 dBm (one drop at a time) --------------
    P = ra.dbm_to_w(30.0)
    fns = {
        METHODS[0]: ra.round_robin,
        METHODS[1]: ra.strongest_wf,
        METHODS[2]: ra.milp_equal,
        METHODS[3]: ra.milp_kkt_wf,
        METHODS[4]: lambda aa, PP: dnn_ra.dnn_allocate(net, aa, 30.0),
        METHODS[5]: lambda aa, PP: (lambda XP: (XP[0], ra.kkt_waterfill(aa, XP[0], PP)[0]))(dnn_ra.dnn_allocate(net, aa, 30.0)),
    }
    ntime = 10 if QUICK else 40
    out["time_ms_per_drop"] = {n: 1e3 * single_instance_time(fns[n], a, P, ntime) for n in METHODS}
    # batched DNN throughput (what a base-station scheduler would see with many cells)
    t = time.perf_counter(); dnn_ra.dnn_allocate(net, a, 30.0); dt = time.perf_counter() - t
    out["dnn_batched_ms_per_drop"] = 1e3 * dt / D
    # per-Pmax timing curve (DNN vs MILP) for the figure
    out["time_vs_pmax_ms"] = {}
    for n in (METHODS[2], METHODS[3], METHODS[4], METHODS[5]):
        out["time_vs_pmax_ms"][n] = []
    for pdbm in PMAX_DBM_SWEEP:
        PP = ra.dbm_to_w(pdbm)
        for n in (METHODS[2], METHODS[3]):
            f = fns[n]
            out["time_vs_pmax_ms"][n].append(1e3 * single_instance_time(f, a, PP, 8))
        for n in (METHODS[4], METHODS[5]):
            def g(aa, PP_, n=n, pdbm=pdbm):
                X, p = dnn_ra.dnn_allocate(net, aa, pdbm)
                return X, (ra.kkt_waterfill(aa, X, PP_)[0] if n == METHODS[5] else p)
            out["time_vs_pmax_ms"][n].append(1e3 * single_instance_time(g, a, PP, 8))
    out["n_drops"] = D
    out["noise_dbm"] = float(10 * np.log10(ra.noise_power_w()) + 30)
    out["avg_symbols_per_mcs"] = avg_sym.tolist()
    out["avg_info_bits"] = float(amc.L)
    json.dump(out, open(os.path.join(RES_DIR, "task3_results.json"), "w"))
    print("[3] time/drop at 30 dBm [ms]:", {k.split()[0]+k.split()[-1][:3]: round(v, 2) for k, v in out["time_ms_per_drop"].items()})
    print(f"[task3] done in {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
