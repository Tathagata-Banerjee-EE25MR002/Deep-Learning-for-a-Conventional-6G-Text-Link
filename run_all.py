"""
run_all.py -- ONE command reproduces every number and figure of Exercise 1:

        python run_all.py              # full run (~45 min on 2 CPU cores)
        EX1_QUICK=1 python run_all.py  # smoke test (~10 min)
        python run_all.py --only 3 figures   # run selected stages

Stages:  1 = conventional text link (Task 1)       -> results/task1_*.json, data/amc_lut.npz
         2 = autoencoder PHY (Task 2)              -> results/task2_results.json
         3 = multi-user RA with DNN (Task 3)       -> results/task3_results.json
         figures = every figure (paper PDF + slide PNG)
Stage 3 needs the AMC look-up table of stage 1.  All randomness is seeded (config.SEED).
"""
import sys, time, runpy, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STAGES = {
    "1": "task1_link",
    "2": "task2_autoencoder",
    "3": "task3_ra",
    "figures": "make_figures",
}


def main():
    args = sys.argv[1:]
    todo = list(STAGES)
    if "--only" in args:
        todo = [a for a in args[args.index("--only") + 1:] if a in STAGES]
    t0 = time.time()
    for s in todo:
        print(f"\n===== stage {s}: {STAGES[s]} =====", flush=True)
        runpy.run_module(STAGES[s], run_name="__main__")
    print(f"\nall done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
