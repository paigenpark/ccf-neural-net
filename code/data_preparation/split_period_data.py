"""Convert the raw HFD Lexis-square ASFR file into the numeric format the models read.

Run this first, from anywhere (paths resolve relative to this file):

    .venv/bin/python code/data_preparation/split_period_data.py

Input (download from the Human Fertility Database, which needs registration):

    data/asfr/asfrRR.txt        columns: Code Year Age ASFR, 3 header lines

Outputs:

    data/asfr_1891_to_2024.txt  numeric rows [country_idx, year, age, rate]
    data/geos_key.npy           (country code, country_idx) pairs

Country index = order of first appearance in asfrRR.txt. The R benchmarks
rebuild the same mapping from the raw file (shared_helpers.R,
Schmertmann2013.R), and split_vv_data.py reuses geos_key.npy, so index k means
the same country everywhere.

Cleaning, identical for every method:
  - non-numeric ages ("12-", "55+") and ages above 54 are dropped
  - non-numeric rates are dropped
  - rates above 1 are capped at 1

The file keeps every numeric age up to 54; load_data() and each R script keep
ages 15-44.
"""

import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../data"))

RAW_RR = os.path.join(DATA_DIR, "asfr", "asfrRR.txt")
OUT = os.path.join(DATA_DIR, "asfr_1891_to_2024.txt")
GEOS_KEY = os.path.join(DATA_DIR, "geos_key.npy")
N_HEADER = 3
MAX_AGE = 54


def main():
    data = []
    countries = []   # codes in order of first appearance

    with open(RAW_RR) as f:
        for row_index, row in enumerate(f):
            row = row.strip()
            if row_index < N_HEADER:
                print(row)
                continue

            country, year, age, rate = row.split()
            if country not in countries:
                countries.append(country)
            try:
                age = int(age)
            except ValueError:
                continue
            try:
                rate = float(rate)
            except ValueError:
                continue
            if age > MAX_AGE:
                continue
            data.append([countries.index(country), int(year), age, min(rate, 1.0)])

    asfr = np.array(data)
    np.savetxt(OUT, asfr)
    np.save(GEOS_KEY, np.column_stack((np.array(countries), np.arange(len(countries)))))

    print(f"Saved {OUT}: {asfr.shape[0]:,} rows")
    print(f"  years {int(asfr[:, 1].min())}-{int(asfr[:, 1].max())}, "
          f"ages {int(asfr[:, 2].min())}-{int(asfr[:, 2].max())}, "
          f"{len(countries)} countries")
    print(f"Saved {GEOS_KEY}")


if __name__ == "__main__":
    main()
