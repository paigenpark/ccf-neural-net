"""Check the outputs of run_all.py.

Two levels, cheapest first (run from the repo root):

    .venv/bin/python code/future_forecasts/verify_future.py
        Audits the CSVs in data/future_forecasts without re-running anything
        (seconds). Checks each file's jump-off year, horizon, age range and rate
        values against the same model in the main run, and checks that the
        methods agree where they must (cohorts with only one forecast age).

    .venv/bin/python code/future_forecasts/verify_future.py --reproduce
        Re-runs the R benchmarks through run_all.py's patching at jump-off 2010,
        a year the main analysis also ran, into a scratch folder, and compares
        them cell by cell with the main outputs (minutes). Exact equality shows
        the patching changes only the jump-off year and output location, not the
        model. This can't be checked at 2019 because the main run has no 2019.

The DL model is left out of --reproduce by default: it is stochastic, so it only
reproduces to within seed noise (~8e-5 per cell), and re-training it is slow.
`--reproduce --with-dl` includes it and reports the largest difference.
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

FUTURE_JOY = 2019
CF_AGE_LO, CF_AGE_HI = 15, 40          # matches the eval notebooks
N_CF_AGES = CF_AGE_HI - CF_AGE_LO + 1
FORECAST_LEN = 30
AGE_LO, AGE_HI = 15, 44

# label -> (forecast basename, observed basename). Mirrors MODEL_REGISTRY in
# eval_figures_future.ipynb; if that registry changes, change this too.
REGISTRY = {
    "Lee":          ("lee2_forecasts_cohort",            "lee2_obs_cohort"),
    "CCF-Net":      ("dl_forecasts_future_joy2019",      "dl_obs_cohort_future_joy2019"),
    "Freeze":       ("freeze_forecasts_cohort",          "freeze_obs_cohort"),
    "Myrskyla":     ("myrskyla_forecasts_cohort",        "myrskyla_obs_cohort"),
    "de Beer":      ("debeer2_forecasts_cohort",         "debeer2_obs_cohort"),
    "Schmertmann":  ("schmertmann2013_forecasts_cohort", "schmertmann2013_obs_cohort"),
    "Sevcikova":    ("sevcikova_forecasts_cohort",       "sevcikova_obs_cohort"),
}

# Each method's output in the main run. The models differ in shape (de Beer
# carries cohorts 59 years past the jump-off; Schmertmann and Sevcikova start at
# age 16 and stop at JOY+29), so each future file is compared with the same
# model's main-run file rather than one common shape.
PRIMARY_OF = {
    "Lee":         "lee2_forecasts_cohort",
    "CCF-Net":     "dl_forecasts_CCF_Net_u110_l4_d0.0_w40",
    "Freeze":      "freeze_forecasts_cohort",
    "Myrskyla":    "myrskyla_forecasts_cohort",
    "de Beer":     "debeer2_forecasts_cohort",
    "Schmertmann": "schmertmann2013_forecasts_cohort",
    "Sevcikova":   "sevcikova_forecasts_cohort",
}

# Hyperparameters of the network the paper calls CCF-Net. The future run must use
# the same ones, or Figure 6 shows a different network from the rest of the paper.
PAPER_DL_VARIANT = "CCF_Net_u110_l4_d0.0_w40"   # VARIANT in train_ensemble.py

# Main-run outputs are in data/; the DL runs may be in data/sensitivity_checks/.
PRIMARY_DIRS = [DATA_DIR, os.path.join(DATA_DIR, "sensitivity_checks")]

# For --reproduce: run_all.py model name -> primary CSV basename it should match.
REPRO_MODELS = {
    "freeze":      "freeze_forecasts_cohort",
    "myrskyla":    "myrskyla_forecasts_cohort",
    "lee":         "lee2_forecasts_cohort",
    "debeer":      "debeer2_forecasts_cohort",
    "schmertmann": "schmertmann2013_forecasts_cohort",
}

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_tally = {PASS: 0, WARN: 0, FAIL: 0}


def check(status, label, detail=""):
    _tally[status] += 1
    mark = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"[{mark}] {label}" + (f"  --  {detail}" if detail else ""))


def load(out_dir, basename):
    df = pd.read_csv(os.path.join(out_dir, basename + ".csv"))
    df["Country"] = df["Country"].astype(int)      # some scripts quote it
    return df


# ---------------------------------------------------------------------------
# Tier 1: audit the files that are already there
# ---------------------------------------------------------------------------
def spliced_cf(pred, obs, joy):
    """CF per (Country, cohort): observed ages 15..AAF plus forecast ages AAF+1..40.

    Same calculation as `spliced_cf` in eval_figures_future.ipynb.
    """
    o = (obs[obs["Age"].between(CF_AGE_LO, CF_AGE_HI)]
         [["Country", "Year", "Age", "Rate"]]
         .drop_duplicates(["Country", "Year", "Age"]))
    hist = o[o["Year"] + o["Age"] <= joy]
    f = pred[pred["JumpOffYear"] == joy][["Country", "Year", "Age", "Rate"]]
    f = f[(f["Year"] + f["Age"] > joy) & f["Age"].between(CF_AGE_LO, CF_AGE_HI)]
    traj = pd.concat([hist, f])
    n = traj.groupby(["Country", "Year"])["Age"].transform("nunique")
    res = (traj[n == N_CF_AGES].groupby(["Country", "Year"])["Rate"]
           .sum().reset_index(name="cf"))
    res["AAF"] = joy - res["Year"]
    return res


def parse_variant(name):
    """Recover hyperparameters from a train_ensemble VARIANT name.

    'CCF_Net_u110_l4_d0.0_w40' -> units 110, 4 layers, dropout 0.0, 40-year window.
    """
    m = re.search(r"u(\d+)_l(\d+)_d([\d.]+)_w(\d+|None)$", name)
    if not m:
        raise ValueError(f"cannot parse variant name {name!r}")
    return dict(units=int(m.group(1)), n_layers=int(m.group(2)),
                dropout=float(m.group(3)),
                train_window=None if m.group(4) == "None" else int(m.group(4)))


def fingerprint(pred, joy):
    """The shape of one method's forecast at one jump-off year.

    Period-year span is stored as an OFFSET from the jump-off year, which is what
    makes it comparable between the primary run's jump-off years and this one.
    """
    f = pred[(pred["JumpOffYear"] == joy) & (pred["Year"] + pred["Age"] > joy)]
    if not len(f):
        return None
    period = f["Year"] + f["Age"]
    r = f["Rate"]
    return dict(
        h_lo=int(period.min()) - joy, h_hi=int(period.max()) - joy,
        age_lo=int(f["Age"].min()), age_hi=int(f["Age"].max()),
        n_neg=int((r < 0).sum()), n_big=int((r > 1).sum()),
        n_nan=int(r.isna().sum()), n=len(f),
    )


def find_primary(basename):
    """Locate a primary-run CSV, which may sit in data/ or data/sensitivity_checks/."""
    for d in PRIMARY_DIRS:
        path = os.path.join(d, basename + ".csv")
        if os.path.exists(path):
            return path
    return None


def primary_fingerprints(basename):
    """Fingerprint the primary run's file at each of its own jump-off years."""
    path = find_primary(basename)
    if path is None:
        return None
    p = pd.read_csv(path)
    p["Country"] = p["Country"].astype(int)
    return {int(j): fingerprint(p, int(j)) for j in sorted(p["JumpOffYear"].unique())}


def audit(out_dir, joy):
    print(f"\n{'=' * 72}\nTIER 1  file audit -- {os.path.relpath(out_dir, REPO_ROOT)}"
          f", jump-off {joy}\n{'=' * 72}\n")

    missing = [b for pair in REGISTRY.values() for b in pair
               if not os.path.exists(os.path.join(out_dir, b + ".csv"))]
    if missing:
        check(FAIL, "all registry files present", f"missing: {', '.join(missing)}")
        return
    check(PASS, "all 14 registry files present")

    frames, cfs = {}, {}
    print()
    for label, (pf, of) in REGISTRY.items():
        pred, obs = load(out_dir, pf), load(out_dir, of)
        frames[label] = (pred, obs)

        # -- the jump-off year is the one we asked for, and only that one ----
        joys = sorted(pred["JumpOffYear"].unique())
        if joys != [joy]:
            check(FAIL, f"{label}: jump-off year", f"expected [{joy}], found {joys}")
        else:
            check(PASS, f"{label}: jump-off year is {joy} only")

        # -- shape, judged against this same model in the primary run --------
        fp = fingerprint(pred, joy)
        prim = primary_fingerprints(PRIMARY_OF[label])

        if fp["n_nan"]:
            check(FAIL, f"{label}: no NaN rates", f"{fp['n_nan']} NaN forecast cells")
        else:
            check(PASS, f"{label}: {fp['n']:,} forecast cells, none NaN")

        if prim is None:
            check(WARN, f"{label}: convention vs. primary run",
                  f"data/{PRIMARY_OF[label]}.csv not found; shape unverified")
        else:
            # Horizon and age span should match the primary run exactly. They are
            # model constants, so a difference here is this runner's doing.
            spans = {(p["h_lo"], p["h_hi"]) for p in prim.values() if p}
            ages = {(p["age_lo"], p["age_hi"]) for p in prim.values() if p}
            mine_span, mine_age = (fp["h_lo"], fp["h_hi"]), (fp["age_lo"], fp["age_hi"])
            if len(spans) == 1 and mine_span not in spans:
                s = next(iter(spans))
                check(FAIL, f"{label}: forecast horizon",
                      f"JOY+{mine_span[0]}..JOY+{mine_span[1]}, but the primary run "
                      f"always gives JOY+{s[0]}..JOY+{s[1]}")
            elif len(spans) == 1:
                check(PASS, f"{label}: horizon JOY+{mine_span[0]}..JOY+{mine_span[1]}"
                            f" (= primary)", f"{joy + mine_span[0]}-{joy + mine_span[1]}")
            if len(ages) == 1 and mine_age not in ages:
                a = next(iter(ages))
                check(FAIL, f"{label}: forecast age span",
                      f"{mine_age[0]}-{mine_age[1]}, but the primary run gives "
                      f"{a[0]}-{a[1]}")
            elif len(ages) == 1:
                check(PASS, f"{label}: ages {mine_age[0]}-{mine_age[1]} (= primary)")

            # Out-of-range rates are a known behaviour of some models at some
            # jump-off years, so the question is only whether this run is worse.
            prim_worst = max((p["n_neg"] + p["n_big"]) / max(p["n"], 1)
                             for p in prim.values() if p)
            mine_bad = fp["n_neg"] + fp["n_big"]
            mine_rate = mine_bad / max(fp["n"], 1)
            if not mine_bad:
                check(PASS, f"{label}: all rates in [0, 1]")
            elif mine_rate <= max(prim_worst, 1e-9) * 3:
                check(WARN, f"{label}: rates outside [0, 1]",
                      f"{fp['n_neg']} negative, {fp['n_big']} > 1 "
                      f"({100 * mine_rate:.2f}% of cells); this model also does this "
                      f"in the primary run (worst JOY: {100 * prim_worst:.2f}%)")
            else:
                check(FAIL, f"{label}: rates outside [0, 1]",
                      f"{100 * mine_rate:.2f}% of cells, vs. at most "
                      f"{100 * prim_worst:.2f}% at any primary jump-off year")

        cfs[label] = spliced_cf(pred, obs, joy)
        print()

    # -- country coverage ---------------------------------------------------
    print(f"{'-' * 72}\ncountry coverage (each method drops the countries it "
          f"cannot support)\n{'-' * 72}")
    sets = {}
    for label, (pred, _) in frames.items():
        s = set(pred[pred["Year"] + pred["Age"] > joy]["Country"])
        sets[label] = s
        print(f"    {label:<14} {len(s):>3} countries")
    shared = set.intersection(*sets.values())
    union = set.union(*sets.values())
    print(f"    {'-' * 30}\n    {'shared by all':<14} {len(shared):>3} countries"
          f"   (union {len(union)})")
    if len(shared) < 20:
        check(WARN, "shared country set", f"only {len(shared)} countries in common; "
              "cross-method means rest on a thin sample")
    else:
        check(PASS, "shared country set", f"{len(shared)} countries in every method")

    # -- the strongest cheap check: forced agreement at short horizons -------
    print(f"\n{'-' * 72}\nspliced CF, mean over the {len(shared)} shared countries\n"
          f"{'-' * 72}")
    print("At AAF 39 a cohort has ONE forecast age out of 26, so every method is\n"
          "~96% observed data and they must nearly coincide. Divergence there is a\n"
          "bug (misaligned years, wrong scaling), not a modelling difference. The\n"
          "fan should then open smoothly as AAF falls.\n")
    aafs = [39, 35, 30, 25, 20, 15]
    hdr = "    AAF  fcast ages  " + "".join(f"{l:>13}" for l in REGISTRY)
    print(hdr)
    table = {}
    for aaf in aafs:
        vals = {}
        for label, c in cfs.items():
            sel = c[(c["AAF"] == aaf) & c["Country"].isin(shared)]
            vals[label] = sel["cf"].mean() if len(sel) else np.nan
        table[aaf] = vals
        print(f"    {aaf:>3}  {CF_AGE_HI - aaf:>10}  "
              + "".join(f"{vals[l]:>13.3f}" for l in REGISTRY))
    print()

    v39 = np.array([table[39][l] for l in REGISTRY], dtype=float)
    spread = np.nanmax(v39) - np.nanmin(v39)
    if spread > 0.05:
        outlier = max(REGISTRY, key=lambda l: abs(table[39][l] - np.nanmedian(v39)))
        check(FAIL, "one-forecast-age agreement (AAF 39)",
              f"spread {spread:.3f} across methods; {outlier} is the outlier "
              f"({table[39][outlier]:.3f} vs median {np.nanmedian(v39):.3f})")
    elif spread > 0.005:
        outlier = max(REGISTRY, key=lambda l: abs(table[39][l] - np.nanmedian(v39)))
        check(WARN, "one-forecast-age agreement (AAF 39)",
              f"spread {spread:.3f}; {outlier} sits {table[39][outlier] - np.nanmedian(v39):+.3f} "
              f"off the median -- expected if it uses a different observed base")
    else:
        check(PASS, "one-forecast-age agreement (AAF 39)", f"spread {spread:.4f}")

    for aaf in aafs:
        offenders = [l for l in REGISTRY
                     if not (0.2 <= table[aaf][l] <= 4.0) and not np.isnan(table[aaf][l])]
        if offenders:
            check(WARN, f"CF plausibility at AAF {aaf}",
                  ", ".join(f"{l}={table[aaf][l]:.2f}" for l in offenders)
                  + " outside 0.2-4.0 children/woman")

    # -- DL-specific: ensemble spread and config echo ------------------------
    print()
    sd_path = os.path.join(out_dir, f"dl_forecasts_future_joy{joy}_seedsd.csv")
    if os.path.exists(sd_path):
        sd = pd.read_csv(sd_path)
        sd_col = "RateSD" if "RateSD" in sd.columns else "Rate"
        m, p99 = sd[sd_col].mean(), sd[sd_col].quantile(0.99)
        if p99 > 0.02:
            check(WARN, "DL seed spread", f"mean SD {m:.2e}, p99 {p99:.2e} -- "
                  "members disagree more than usual; check dl.log for a bad seed")
        else:
            check(PASS, "DL seed spread", f"mean per-cell SD {m:.2e}, p99 {p99:.2e}")

    cfg_path = os.path.join(out_dir, f"dl_config_future_joy{joy}.json")
    if os.path.exists(cfg_path):
        import json
        cfg = json.load(open(cfg_path))
        jo = cfg.get("jump_off_years")
        if jo != [joy]:
            check(FAIL, "DL config jump-off", f"config says {jo}, expected [{joy}]")
        else:
            check(PASS, "DL config jump-off", f"{jo}, forecast_len "
                  f"{cfg.get('forecast_len')}")

        # run_dl_future.py uses whatever config is currently in train_ensemble.py.
        # If a variant was left there, the future run used a different network
        # from the paper's CCF-Net.
        want = parse_variant(PAPER_DL_VARIANT)
        got = dict(cfg.get("hparams", {}))
        got["train_window"] = cfg.get("train_window")
        diffs = {k: (want[k], got.get(k)) for k in want if got.get(k) != want[k]}
        if diffs:
            check(FAIL, "DL config matches the paper's CCF-Net",
                  "future run used " + ", ".join(
                      f"{k}={g} (paper: {w})" for k, (w, g) in diffs.items())
                  + f" -- the future figures' CCF-Net is not the "
                    f"'{PAPER_DL_VARIANT}' network the cohort notebook scores")
        else:
            check(PASS, "DL config matches the paper's CCF-Net",
                  f"units/layers/dropout/window all equal {PAPER_DL_VARIANT}")
        print(f"           future hparams : {cfg.get('hparams')}, "
              f"train_window={cfg.get('train_window')}")
        print(f"           paper CCF-Net  : {PAPER_DL_VARIANT}")


# ---------------------------------------------------------------------------
# Tier 2: re-run the machinery at a year the primary analysis also ran
# ---------------------------------------------------------------------------
def reproduce(joy, with_dl):
    subdir = f"_verify_joy{joy}"
    out_dir = os.path.join(DATA_DIR, subdir)
    print(f"\n{'=' * 72}\nTIER 2  reproduction test at jump-off {joy}\n{'=' * 72}\n")
    print(f"Re-running the R benchmarks through run_all.py's patcher at {joy},")
    print(f"into data/{subdir}, then diffing against the primary outputs in data/.")
    print("Any nonzero difference means the patching changed the model, not just")
    print("the calendar.\n")

    models = list(REPRO_MODELS) + (["dl"] if with_dl else [])
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "run_all.py"),
           "--joys", str(joy), "--models", ",".join(models),
           "--out-subdir", subdir]
    print(f"$ {' '.join(cmd)}\n")
    if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
        check(FAIL, "re-run completed", "run_all.py exited nonzero; see "
              f"data/{subdir}/logs/")
        return
    check(PASS, "re-run completed")

    print(f"\n{'-' * 72}\ncell-by-cell diff vs. the primary run\n{'-' * 72}")
    keys = ["Country", "Year", "Age"]
    for name, basename in REPRO_MODELS.items():
        prim_path = os.path.join(DATA_DIR, basename + ".csv")
        if not os.path.exists(prim_path):
            check(WARN, f"{name}: primary output", f"{basename}.csv not in data/, "
                  "nothing to diff against")
            continue
        new = load(out_dir, basename)
        old = load(DATA_DIR, basename)
        new = new[new["JumpOffYear"] == joy][keys + ["Rate"]]
        old = old[old["JumpOffYear"] == joy][keys + ["Rate"]]
        if not len(old):
            check(WARN, f"{name}: primary output",
                  f"no JumpOffYear=={joy} rows in the primary file")
            continue

        merged = old.merge(new, on=keys, how="outer",
                           suffixes=("_old", "_new"), indicator=True)
        only_old = int((merged["_merge"] == "left_only").sum())
        only_new = int((merged["_merge"] == "right_only").sum())
        both = merged[merged["_merge"] == "both"]
        d = (both["Rate_new"] - both["Rate_old"]).abs()
        maxd = float(d.max()) if len(d) else 0.0

        if only_old or only_new:
            check(FAIL, f"{name}: same cells",
                  f"{only_old} cells only in primary, {only_new} only in re-run")
        elif maxd > 1e-9:
            worst = both.loc[d.idxmax()]
            check(FAIL, f"{name}: identical rates",
                  f"max |diff| {maxd:.3e} at country {int(worst.Country)} "
                  f"cohort {int(worst.Year)} age {int(worst.Age)} "
                  f"({worst.Rate_old:.6f} -> {worst.Rate_new:.6f})")
        else:
            check(PASS, f"{name}: {len(both):,} cells reproduce exactly",
                  f"max |diff| {maxd:.1e}")

    if with_dl:
        new_p = os.path.join(out_dir, f"dl_forecasts_future_joy{joy}.csv")
        old_p = find_primary(PRIMARY_OF["CCF-Net"])
        if old_p is None or not os.path.exists(new_p):
            check(WARN, "CCF-Net: reproduction",
                  f"missing {PRIMARY_OF['CCF-Net']}.csv or the re-run output; "
                  "nothing to compare")
        else:
            new = pd.read_csv(new_p); old = pd.read_csv(old_p)
            for df in (new, old):
                df["Country"] = df["Country"].astype(int)
            new = new[new["JumpOffYear"] == joy][keys + ["Rate"]]
            old = old[old["JumpOffYear"] == joy][keys + ["Rate"]]
            m = old.merge(new, on=keys, suffixes=("_old", "_new"))
            d = (m["Rate_new"] - m["Rate_old"]).abs()
            maxd, meand = float(d.max()), float(d.mean())
            # Stochastic: judged against the ensemble's own noise floor, not zero.
            if maxd > 5e-3:
                check(WARN, "CCF-Net: reproduces within seed noise",
                      f"max |diff| {maxd:.2e}, mean {meand:.2e} -- larger than the "
                      "~8e-5 noise floor; worth a look")
            else:
                check(PASS, "CCF-Net: reproduces within seed noise",
                      f"max |diff| {maxd:.2e}, mean {meand:.2e}")

    print(f"\nScratch outputs left in data/{subdir}/ -- delete when done.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-subdir", default="future_forecasts",
                    help="subfolder of data/ to audit (default: future_forecasts)")
    ap.add_argument("--joy", type=int, default=FUTURE_JOY,
                    help=f"jump-off year the audited run used (default: {FUTURE_JOY})")
    ap.add_argument("--reproduce", action="store_true",
                    help="also re-run the R benchmarks at --repro-joy and diff "
                         "against the primary outputs")
    ap.add_argument("--repro-joy", type=int, default=2010,
                    help="jump-off year for the reproduction test; must be one the "
                         "primary analysis ran (default: 2010)")
    ap.add_argument("--with-dl", action="store_true",
                    help="include the DL ensemble in the reproduction test (slow, "
                         "and compares within seed noise rather than exactly)")
    ap.add_argument("--skip-audit", action="store_true",
                    help="run only the reproduction test")
    args = ap.parse_args()

    if not args.skip_audit:
        audit(os.path.join(DATA_DIR, args.out_subdir), args.joy)
    if args.reproduce:
        reproduce(args.repro_joy, args.with_dl)

    print(f"\n{'=' * 72}")
    print(f"  {_tally[PASS]} passed, {_tally[WARN]} warnings, {_tally[FAIL]} failures")
    print(f"{'=' * 72}")
    if _tally[FAIL]:
        sys.exit(1)


if __name__ == "__main__":
    main()
