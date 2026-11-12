"""Re-run every model from a recent jump-off year and forecast past the data.

This produces the forecasts in Figure 6 of the paper. The main analysis scores
forecasts from jump-off years 1985-2013 against observed data. This run instead
jumps off from 2019 and forecasts forward, so there is nothing yet to score
against; the methods can only be compared with each other.

The model scripts are not edited. The R benchmarks in code/pnas_benchmarks/ and
the DL trainer in code/dl_training_with_uncertainty/ keep their own jump-off
years and output paths.

How the R benchmarks are re-pointed
-----------------------------------
Each script is read, two kinds of line are rewritten in memory, and the result
is written to data/future_forecasts/_patched/ and run from there with Rscript:

    joys <- c(1985, ...)              ->  joys <- c(2019)
    file.path(path, "<name>.csv")     ->  file.path("<out_dir>", "<name>.csv")

Every CSV a script writes is redirected (forecasts, observed rates and
uncertainty intervals), so the main outputs in data/ are never overwritten.
`path` is also the scripts' input directory, so only .csv paths are rewritten;
the inputs are .txt files. Both rewrites are checked, so a script whose layout
has changed fails loudly instead of forecasting the wrong years.

Everything else (the model, the country checks, the 30-year horizon, the output
columns) is unchanged, so these forecasts are comparable to the main ones.
`verify_future.py --reproduce` checks this by re-running the patched scripts at
2010 and comparing against the main outputs.

Why 2019
--------
Country series end in different years (2013 to 2024), so a single shared
jump-off year is needed. At 2019, 32 of 39 countries have a complete series up
to the jump-off; at 2022 only 24 do. The scripts skip countries they cannot
support, so a later year silently drops countries. Use --joys to change it.

Usage (from the repo root)
--------------------------
    .venv/bin/python code/future_forecasts/run_all.py                # 5 R models + DL
    .venv/bin/python code/future_forecasts/run_all.py --models dl
    .venv/bin/python code/future_forecasts/run_all.py --all          # adds Sevcikova
    .venv/bin/python code/future_forecasts/run_all.py --joys 2017

Sevcikova is left out by default because it runs bayesTFR MCMC (3 chains of
20,000 iterations), which takes hours. Add it with --all or --models sevcikova.

Outputs (in data/future_forecasts/)
-----------------------------------
    <model>_forecasts_cohort.csv     forecast cohort rates, same columns as the main run
    <model>_obs_cohort.csv           observed cohort rates
    dl_forecasts_future_joy<J>.csv   DL ensemble forecast (see run_dl_future.py)
    logs/<model>.log                 stdout/stderr for each model
    _patched/<model>.R               the exact R code that ran
"""

import argparse
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
BENCH_DIR = os.path.join(REPO_ROOT, "code/pnas_benchmarks")

# R benchmarks, roughly cheapest first. Paths are relative to BENCH_DIR.
R_MODELS = {
    "freeze":       "FreezeRates/FreezeRates.R",
    "myrskyla":     "Myrskyla2013/Myrskyla2013.R",
    "lee":          "Lee1993Log/Lee1993.R",
    "debeer":       "deBeer1985and1989/deBeer1985and1989.R",
    "schmertmann":  "Schmertmann2013/Schmertmann2013.R",
    "sevcikova":    "Sevcikova2016/Sevcikova2016.r",
}

# Left out of the default run; opt in with --all or --models sevcikova.
EXPENSIVE = {"sevcikova"}

DEFAULT_JOYS = [2019]

# The jump-off-year assignment at the top level of each script (Sevcikova calls
# it JOYS). Anchored to the start of a line so nothing inside a function matches.
JOYS_RE = re.compile(r"^(joys|JOYS)[ \t]*<-[ \t]*c\([\d,\s]+\)[ \t]*$", re.MULTILINE)

# Every CSV path under `path`. The scripts only write CSVs there (their inputs
# are .txt), so this catches every output and no input.
OUT_CSV_RE = re.compile(r'file\.path\(\s*path\s*,\s*"([A-Za-z0-9_]+\.csv)"\s*\)')


def patch_script(src_path, dst_path, joys, out_dir):
    """Write a copy of an R benchmark that jumps off at `joys` and writes to `out_dir`.

    Raises RuntimeError if a pattern isn't found as expected, which means the
    original script has changed and this runner needs updating.
    """
    with open(src_path) as f:
        code = f.read()

    joys_literal = ", ".join(str(j) for j in joys)

    # keep the original variable name (joys or JOYS)
    code, n_joys = JOYS_RE.subn(lambda m: f"{m.group(1)} <- c({joys_literal})", code)
    if n_joys != 1:
        raise RuntimeError(
            f"{os.path.basename(src_path)}: expected 1 jump-off-year assignment, "
            f"found {n_joys}. Update JOYS_RE in run_all.py.")

    code, n_out = OUT_CSV_RE.subn(lambda m: f'file.path("{out_dir}", "{m.group(1)}")', code)
    if n_out < 2:
        raise RuntimeError(
            f"{os.path.basename(src_path)}: expected at least 2 output CSV paths, "
            f"found {n_out}. Update OUT_CSV_RE in run_all.py.")

    header = (
        f"## GENERATED by code/future_forecasts/run_all.py -- do not edit.\n"
        f"## Source: {os.path.relpath(src_path, REPO_ROOT)}\n"
        f"## Rewritten: jump-off year(s) -> c({joys_literal}); "
        f"{n_out} CSV path(s) -> {out_dir}\n\n"
    )
    with open(dst_path, "w") as f:
        f.write(header + code)
    return n_out


def run(name, cmd, log_path, env=None):
    """Run one model, writing its output to a log. Returns (ok, seconds)."""
    print(f"  running {name} -> {os.path.relpath(log_path, REPO_ROOT)}", flush=True)
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, **(env or {})})
    elapsed = time.time() - start
    ok = proc.returncode == 0
    print(f"  {name}: {'ok' if ok else f'FAILED (exit {proc.returncode})'} "
          f"in {elapsed:.0f}s", flush=True)
    return ok, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joys", default=",".join(str(j) for j in DEFAULT_JOYS),
                    help="comma-separated jump-off year(s) (default: 2019)")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset: " + ", ".join(list(R_MODELS) + ["dl"]))
    ap.add_argument("--all", action="store_true",
                    help="include slow models (" + ", ".join(EXPENSIVE) + ")")
    ap.add_argument("--out-subdir", default="future_forecasts",
                    help="subfolder of data/ to write into (default: future_forecasts)")
    ap.add_argument("--patch-only", action="store_true",
                    help="write the patched R scripts and stop without running them")
    args = ap.parse_args()

    joys = [int(j) for j in args.joys.split(",")]

    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        unknown = set(selected) - set(R_MODELS) - {"dl"}
        if unknown:
            ap.error(f"unknown model(s): {', '.join(sorted(unknown))}")
    else:
        selected = [m for m in R_MODELS if args.all or m not in EXPENSIVE] + ["dl"]

    out_dir = os.path.join(DATA_DIR, args.out_subdir)
    patched_dir = os.path.join(out_dir, "_patched")
    log_dir = os.path.join(out_dir, "logs")
    for d in (out_dir, patched_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    print("Future forecasts")
    print(f"  jump-off year(s) : {joys}")
    print(f"  models           : {', '.join(selected)}")
    print(f"  output           : {os.path.relpath(out_dir, REPO_ROOT)}\n")

    # ---- patch the R benchmarks -------------------------------------------
    r_selected = [m for m in selected if m in R_MODELS]
    patched = {}
    for name in r_selected:
        dst = os.path.join(patched_dir, f"{name}.R")
        n_out = patch_script(os.path.join(BENCH_DIR, R_MODELS[name]), dst, joys, out_dir)
        patched[name] = dst
        print(f"  patched {name}: jump-off year and {n_out} CSV path(s)")

    if args.patch_only:
        print(f"\n--patch-only: wrote {len(patched)} script(s) to "
              f"{os.path.relpath(patched_dir, REPO_ROOT)}; nothing run.")
        return

    # ---- run ---------------------------------------------------------------
    results = {}
    for name in r_selected:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        env = {}
        if name == "sevcikova":
            # Sevcikova also reads its jump-off years from SEV_JOYS, so set it to
            # match the patch. Keep its MCMC cache separate from the main run's.
            env["SEV_JOYS"] = ",".join(str(j) for j in joys)
            env["SEV_MCMC_ROOT"] = os.path.join(out_dir, "sevcikova_mcmc")
        results[name] = run(name, ["Rscript", patched[name]],
                            os.path.join(log_dir, f"{name}.log"), env)

    if "dl" in selected:
        print(f"\n{'=' * 60}\ndl\n{'=' * 60}")
        results["dl"] = run(
            "dl",
            [sys.executable, os.path.join(SCRIPT_DIR, "run_dl_future.py"),
             "--joys", ",".join(str(j) for j in joys), "--out-dir", out_dir],
            os.path.join(log_dir, "dl.log"),
        )

    # ---- summary -----------------------------------------------------------
    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    for name, (ok, elapsed) in results.items():
        print(f"  {name:<14} {'ok' if ok else 'FAILED':<8} {elapsed:>7.0f}s")
    print(f"\nOutputs in {os.path.relpath(out_dir, REPO_ROOT)}")
    failed = [n for n, (ok, _) in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} model(s) failed: {', '.join(failed)}")
        print(f"See {os.path.relpath(log_dir, REPO_ROOT)}/<model>.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
