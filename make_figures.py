"""
make_figures.py -- turns the saved result files into every figure of the paper and the
slides.  Each figure is rendered twice:
    figures/paper/<name>.pdf   compact, 7 pt fonts, sized for IEEE columns (3.4 in) / page (7.1 in)
    figures/slide/<name>.png   large fonts for 16:9 slides
Nothing is simulated here, so the figures can be restyled quickly.
Colour logic (colour-blind safe, Okabe-Ito): modulation = colour (QPSK blue, 16-QAM green,
64-QAM vermillion); code rate = line style/marker (lower rate solid/circle, 3/4 dashed/square).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

BLUE, GREEN, VERM, ORANGE, SKY, PURPLE, GREY, BLACK = "#0072B2", "#009E73", "#D55E00", "#E69F00", "#56B4E9", "#CC79A7", "#7f7f7f", "#000000"
MOD_COL = {"QPSK": BLUE, "16QAM": GREEN, "64QAM": VERM}
MCS_STYLE = {}
for _m in MCS_TABLE:
    hi = _m["rate"] == "3/4"
    MCS_STYLE[_m["name"]] = dict(color=MOD_COL[_m["mod"]], ls="--" if hi else "-", marker="s" if hi else "o")
KIND = "paper"
SC = 1.0


def setup(kind):
    global KIND, SC
    KIND = kind
    SC = 1.0 if kind == "paper" else 1.75
    plt.rcParams.update({
        "font.size": 7 * SC, "axes.labelsize": 7.5 * SC, "legend.fontsize": 6 * SC if kind == "paper" else 8.5,
        "xtick.labelsize": 6.5 * SC, "ytick.labelsize": 6.5 * SC, "axes.titlesize": 7.5 * SC,
        "lines.linewidth": 1.1 * SC, "lines.markersize": 3.2 * SC, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 100, "savefig.dpi": 200, "font.family": "DejaVu Sans"})


def fig(w, h, **kw):
    return plt.subplots(figsize=(w * SC, h * SC), **kw)


def save(f, name):
    d = os.path.join(FIG_DIR, KIND)
    os.makedirs(d, exist_ok=True)
    f.tight_layout(pad=0.4)
    f.savefig(os.path.join(d, name + (".pdf" if KIND == "paper" else ".png")), bbox_inches="tight")
    plt.close(f)


def L(path):
    return json.load(open(path))


# ------------------------------------------------------------------------------------------ Task 1
def fig_ber_validation():
    d = np.load(os.path.join(RES_DIR, "task1a_ber_validation.npz"))
    f, ax = fig(3.4, 2.6)
    e = d["ebn0_db"]
    for m, nm in ((2, "QPSK"), (4, "16QAM"), (6, "64QAM")):
        for chan, ls, mk in (("awgn", "-", "o"), ("rayleigh", "--", "s")):
            th, sim = d[f"{chan}_m{m}_theory"], d[f"{chan}_m{m}_sim"]
            ax.semilogy(e, th, ls=ls, color=MOD_COL[nm], lw=0.9 * SC)
            ok = sim > 0
            ax.semilogy(e[ok], sim[ok], ls="", marker=mk, mfc="none", color=MOD_COL[nm], ms=3.5 * SC)
    ax.set_ylim(1e-6, 1); ax.set_xlim(0, 30)
    ax.set_xlabel("$E_b/N_0$ [dB] (uncoded)"); ax.set_ylabel("BER")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=MOD_COL[n], label=n.replace("QAM", "-QAM")) for n in MOD_COL]
    h += [Line2D([], [], color="k", ls="-", marker="o", mfc="none", label="AWGN"),
          Line2D([], [], color="k", ls="--", marker="s", mfc="none", label="Rayleigh + ZF")]
    ax.legend(handles=h, loc="lower left", ncol=1, frameon=False)
    ax.text(0.97, 0.97, "lines: exact theory\nmarkers: simulation", transform=ax.transAxes, ha="right", va="top", fontsize=6 * SC if KIND == "paper" else 8.5)
    save(f, "t1_ber_validation")


def fig_coded_curves():
    sw = L(os.path.join(RES_DIR, "task1_sweeps.json"))
    f, axs = fig(7.1, 2.35, ncols=3)
    panels = [("awgn", "ber", "Coded BER, AWGN"), ("awgn", "bler", "BLER, AWGN"), ("rayleigh", "bler", "BLER, Rayleigh block fading")]
    for ax, (ch, key, ttl) in zip(axs, panels):
        for m in MCS_TABLE:
            r = sw[ch][m["name"]]
            y = np.array(r[key], float); x = np.array(r["snr"])
            ok = y > 0
            ax.semilogy(x[ok], y[ok], label=m["name"], **MCS_STYLE[m["name"]], markevery=2 if ch == "awgn" else 1)
        ax.set_xlabel("SNR $E_s/N_0$ [dB]"); ax.set_title(ttl)
        ax.set_ylim(1e-4 if key == "ber" else 3e-4, 1.2 if key == "bler" else 0.6)
        if key == "bler":
            ax.axhline(BLER_TARGET, color="k", lw=0.6, ls=":")
            ax.text(0.98, 0.5, "10 % target", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5 * SC if KIND == "paper" else 8)
    axs[0].set_ylabel("error rate")
    axs[1].legend(loc="lower left", frameon=False, ncol=1)
    save(f, "t1_coded_curves")


def fig_text_quality():
    sw = L(os.path.join(RES_DIR, "task1_sweeps.json"))
    name = "16-QAM 1/2"
    r = sw["awgn"][name]; x = np.array(r["snr"])
    f, ax = fig(3.4, 2.4)
    ax.plot(x, r["bleu"], color=BLUE, marker="o", markevery=2, label="BLEU")
    ax.plot(x, r["wacc"], color=GREEN, marker="s", markevery=2, label="word accuracy")
    ax.plot(x, r["sim"], color=VERM, marker="^", markevery=2, label="sentence similarity")
    ax.plot(x, 1 - np.array(r["bler"]), color=BLACK, ls=":", label="1 - BLER")
    ax.set_xlim(2, 12); ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("SNR $E_s/N_0$ [dB]"); ax.set_ylabel("score"); ax.set_title(f"Cliff effect ({name}, AWGN)")
    ax.legend(loc="center right", frameon=False)
    save(f, "t1_text_quality")


def fig_text_quality_all():
    """Similarity vs SNR of all six MCS (used on slides / LUT illustration)."""
    sw = L(os.path.join(RES_DIR, "task1_sweeps.json"))
    f, ax = fig(3.4, 2.4)
    for m in MCS_TABLE:
        r = sw["awgn"][m["name"]]
        ax.plot(r["snr"], r["sim"], label=m["name"], **MCS_STYLE[m["name"]], markevery=3)
    ax.set_xlabel("SNR $E_s/N_0$ [dB]"); ax.set_ylabel("sentence similarity"); ax.set_title("Similarity LUT (AWGN)")
    ax.legend(frameon=False, loc="lower right"); ax.set_xlim(0, 24)
    save(f, "t1_similarity_lut")


def fig_amc():
    d = np.load(os.path.join(DATA_DIR, "amc_lut.npz"))
    from task1_link import AMC
    amc = AMC()
    snr = np.arange(-2, 28, 0.1)
    idx, eta = amc.select(snr)
    f, ax = fig(3.4, 2.4)
    ax.step(snr, eta, where="post", color=BLACK, label="AMC: spectral efficiency")
    for m, t in zip(MCS_TABLE, d["thr_db"]):
        ax.axvline(t, color=MCS_STYLE[m["name"]]["color"], ls=":", lw=0.8 * SC)
        ax.text(t + 0.15, 0.05 + 0.0 * t, m["name"], rotation=90, fontsize=5.2 * SC if KIND == "paper" else 8,
                color=MCS_STYLE[m["name"]]["color"], va="bottom")
    ax.set_xlabel("SNR $E_s/N_0$ [dB]"); ax.set_ylabel("info bit / symbol"); ax.set_title("AMC rule (BLER $\\leq$ 10 %)")
    ax.set_xlim(-2, 24); ax.set_ylim(0, 5)
    save(f, "t1_amc")


# ------------------------------------------------------------------------------------------ Task 2
def fig_ae_awgn(wide=False):
    r = L(os.path.join(RES_DIR, "task2_results.json"))
    A = r["awgn_7_4"]; e = np.array(A["ebn0"])
    f, ax = fig(3.4, 2.5)
    def p(y, **k):
        y = np.array(y, float); ok = y > 0
        ax.semilogy(e[ok], y[ok], **k)
    p(A["uncoded"], color=GREY, ls=":", marker="x", label="uncoded BPSK (4 bit)")
    p(A["ham_hard"], color=BLUE, ls="--", marker="s", mfc="none", label="Hamming(7,4) hard")
    p(A["ham_ml"], color=GREEN, ls="-", marker="o", mfc="none", label="Hamming(7,4) ML")
    if "ae_const" in A:
        p(A["ae_const"], color=PURPLE, ls="-.", marker="v", label="AE (7,4), const. energy")
    p(A["ae"], color=VERM, ls="-", marker="^", label="AE (7,4), avg. energy")
    ax.set_xlabel("$E_b/N_0$ [dB]"); ax.set_ylabel("BLER"); ax.set_ylim(1e-5, 1); ax.set_title("(7,4) in AWGN")
    ax.legend(frameon=False, loc="lower left")
    save(f, "t2_ae74_awgn")


def fig_ae_rayleigh():
    r = L(os.path.join(RES_DIR, "task2_results.json"))
    C = r["rayleigh_7_4"]; e = np.array(C["ebn0"])
    f, ax = fig(3.4, 2.5)
    def p(y, **k):
        y = np.array(y, float); ok = y > 0
        ax.semilogy(e[ok], y[ok], **k)
    p(C["uncoded"], color=GREY, ls=":", marker="x", label="uncoded BPSK (4 bit)")
    p(C["ham_hard"], color=BLUE, ls="--", marker="s", mfc="none", label="Hamming(7,4) hard")
    p(C["ham_ml"], color=GREEN, ls="-", marker="o", mfc="none", label="Hamming(7,4) ML + CSI")
    if "ae_const" in C:
        p(C["ae_const"], color=PURPLE, ls="-.", marker="v", label="AE, const. energy + CSI")
    p(C["ae"], color=VERM, ls="-", marker="^", label="AE, avg. energy + CSI")
    ax.set_xlabel("average $E_b/N_0$ [dB]"); ax.set_ylabel("BLER"); ax.set_ylim(1e-4, 1); ax.set_title("(7,4) in Rayleigh fading, CSI at Rx")
    ax.legend(frameon=False, loc="lower left")
    save(f, "t2_ae74_rayleigh")


def fig_constellation():
    r = L(os.path.join(RES_DIR, "task2_results.json"))["awgn_2_4"]
    ae, q = np.array(r["ae_points"]), np.array(r["qam_points"])
    f, ax = fig(3.4, 3.0)
    ax.scatter(q[:, 0], q[:, 1], s=34 * SC ** 1.2, facecolors="none", edgecolors=GREY, linewidths=1.0, label="16-QAM")
    ax.scatter(ae[:, 0], ae[:, 1], s=20 * SC ** 1.2, color=VERM, label="AE (2,4)")
    for i, (x, y) in enumerate(ae):
        ax.annotate(str(i), (x, y), fontsize=5 * SC if KIND == "paper" else 8, xytext=(3, 3), textcoords="offset points", color=VERM)
    ax.set_aspect("equal"); ax.set_xlabel("channel use 1 (I)"); ax.set_ylabel("channel use 2 (Q)")
    ax.set_title("Learned constellation vs 16-QAM (same avg. energy)")
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=6 * SC if KIND == "paper" else 9)
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.3, 2.3)
    save(f, "t2_constellation")


def fig_ae24_ser():
    r = L(os.path.join(RES_DIR, "task2_results.json"))["awgn_2_4"]
    e = np.array(r["ebn0"])
    f, ax = fig(3.4, 2.5)
    for key, lab, c, mk, ls in (("qam16", "16-QAM (ML)", GREY, "s", "--"), ("ae_ml", "AE constellation, ML", GREEN, "o", "-"),
                                ("ae", "AE (learned decoder)", VERM, "^", "-")):
        y = np.array(r[key], float); ok = y > 0
        ax.semilogy(e[ok], y[ok], color=c, marker=mk, ls=ls, mfc="none" if key != "ae" else None, label=lab)
    ax.set_xlabel("$E_b/N_0$ [dB]"); ax.set_ylabel("symbol (block) error rate"); ax.set_ylim(1e-5, 1)
    ax.set_title("(2,4) autoencoder in AWGN"); ax.legend(frameon=False, loc="lower left")
    save(f, "t2_ae24_ser")


def fig_ae_training():
    r = L(os.path.join(RES_DIR, "task2_results.json"))
    f, ax = fig(3.4, 2.3)
    for key, lab, c in (("awgn_7_4", "(7,4) AWGN", VERM), ("awgn_2_4", "(2,4) AWGN", BLUE), ("rayleigh_7_4", "(7,4) Rayleigh", GREEN)):
        h = np.array(r[key]["loss_hist"]); ax.semilogy(h[:, 0], h[:, 1], label=lab, color=c)
    ax.set_xlabel("iteration"); ax.set_ylabel("cross-entropy loss"); ax.legend(frameon=False); ax.set_title("AE training")
    save(f, "t2_training")


# ------------------------------------------------------------------------------------------ Task 3
RA_STYLE = {
    "Round-robin + equal power": dict(color=GREY, ls="-", marker="v"),
    "Strongest user + water-filling": dict(color=SKY, ls="-", marker="D"),
    "MILP + equal power": dict(color=BLUE, ls="--", marker="s"),
    "MILP + KKT water-filling": dict(color=BLUE, ls="-", marker="o"),
    "DNN": dict(color=VERM, ls="--", marker="^"),
    "DNN + KKT water-filling": dict(color=ORANGE, ls="-", marker="P"),
}


def fig_ra_main():
    r = L(os.path.join(RES_DIR, "task3_results.json"))
    p = np.array(r["pmax_dbm"])
    f, axs = fig(7.1, 4.0, nrows=2, ncols=3)
    panels = [("sum_rate_mbps", "sum rate [Mbit/s]"), ("pct_users_rmin", "users with $R_u\\geq R_{min}$ [%]"),
              ("jain", "Jain's fairness index"), ("p5_rate_mbps", "5th-percentile user rate [Mbit/s]"),
              ("sentences_per_s", "correct sentences / s"), (None, "computation time [ms/drop]")]
    for ax, (k, yl) in zip(axs.ravel(), panels):
        for name, st in RA_STYLE.items():
            if k is None:
                if name in r["time_vs_pmax_ms"]:
                    ax.semilogy(p, r["time_vs_pmax_ms"][name], label=name, **st)
            else:
                ax.plot(p, r[name][k], label=name, **st)
        ax.set_xlabel("$P_{max}$ [dBm]"); ax.set_ylabel(yl)
        if k == "pct_users_rmin":
            ax.set_ylim(0, 102)
    axs[0, 0].legend(frameon=False, loc="upper left", fontsize=5.3 * SC if KIND == "paper" else 8)
    save(f, "t3_main")


def fig_ra_time_bar():
    r = L(os.path.join(RES_DIR, "task3_results.json"))
    names = list(r["time_ms_per_drop"].keys())
    f, ax = fig(3.4, 2.4)
    vals = [r["time_ms_per_drop"][n] for n in names]
    cols = [RA_STYLE[n]["color"] for n in names]
    ax.barh(range(len(names)), vals, color=cols)
    ax.set_xscale("log"); ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=5.5 * SC if KIND == "paper" else 8)
    for i, v in enumerate(vals):
        ax.text(v * 1.1, i, f"{v:.2f} ms", va="center", fontsize=6 * SC if KIND == "paper" else 9)
    ax.set_xlabel("time per drop at 30 dBm [ms]"); ax.invert_yaxis(); ax.set_xlim(right=max(vals) * 8)
    save(f, "t3_time_bar")


def fig_ra_training():
    p = os.path.join(RES_DIR, "task3_train_hist.json")
    if not os.path.exists(p):
        return
    h = np.array(L(p))
    f, ax = fig(3.4, 2.3)
    ax.plot(h[:, 0], h[:, 2], color=BLUE, label="sum rate [Mbit/s]")
    ax2 = ax.twinx(); ax2.plot(h[:, 0], h[:, 3], color=VERM, label="rate penalty"); ax2.grid(False)
    ax.set_xlabel("iteration"); ax.set_ylabel("sum rate [Mbit/s]", color=BLUE); ax2.set_ylabel("$\\sum_u (R_{min}-R_u)_+^2$", color=VERM)
    ax.set_title("Unsupervised DNN training")
    save(f, "t3_training")


ALL = [fig_ber_validation, fig_coded_curves, fig_text_quality, fig_text_quality_all, fig_amc,
       fig_ae_awgn, fig_ae_rayleigh, fig_constellation, fig_ae24_ser, fig_ae_training,
       fig_ra_main, fig_ra_time_bar, fig_ra_training]


def main():
    for kind in ("paper", "slide"):
        setup(kind)
        for fn in ALL:
            try:
                fn()
            except FileNotFoundError as e:
                print("[figures] skipped", fn.__name__, "(missing results):", e.filename)
    print("[figures] written to", FIG_DIR)


if __name__ == "__main__":
    main()
