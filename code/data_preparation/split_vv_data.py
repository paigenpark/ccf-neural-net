"""Convert the raw HFD vertical-parallelogram ASFR file into the models' numeric format.

Run after split_period_data.py, which writes the geos_key.npy used here:

    .venv/bin/python code/data_preparation/split_vv_data.py

Input:

    data/asfr/asfrVV.txt          columns: Code Year ARDY Cohort ASFR, 3 header lines

Output:

    data/asfrVV_1891_to_2024.txt  numeric rows [country_idx, year, age, rate]

ARDY (age reached during the year) takes the place of age. The Cohort column is
dropped because it always equals Year - ARDY. Countries are indexed through the
shared geos_key.npy, so index k is the same country as in asfr_1891_to_2024.txt.
Cleaning is the same as in split_period_data.py.
"""

import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../data"))

RAW_VV = os.path.join(DATA_DIR, "asfr", "asfrVV.txt")
GEOS_KEY = os.path.join(DATA_DIR, "geos_key.npy")
OUT = os.path.join(DATA_DIR, "asfrVV_1891_to_2024.txt")
N_HEADER = 3
MAX_AGE = 54


def main():
    code_to_index = {code: int(idx) for code, idx in np.load(GEOS_KEY)}

    data = []
    unmapped = set()

    with open(RAW_VV) as f:
        for row_index, row in enumerate(f):
            row = row.strip()
            if row_index < N_HEADER:
                print(row)
                continue

            country, year, ardy, _cohort, rate = row.split()
            if country not in code_to_index:
                unmapped.add(country)
                continue
            try:
                age = int(ardy)
            except ValueError:
                continue
            try:
                rate = float(rate)
            except ValueError:
                continue
            if age > MAX_AGE:
                continue
            data.append([code_to_index[country], int(year), age, min(rate, 1.0)])

    asfr_vv = np.array(data)
    np.savetxt(OUT, asfr_vv)

    print(f"Saved {OUT}: {asfr_vv.shape[0]:,} rows")
    print(f"  years {int(asfr_vv[:, 1].min())}-{int(asfr_vv[:, 1].max())}, "
          f"ages (ARDY) {int(asfr_vv[:, 2].min())}-{int(asfr_vv[:, 2].max())}, "
          f"{len(np.unique(asfr_vv[:, 0]))} countries")
    if unmapped:
        print(f"WARNING: {len(unmapped)} country codes not in geos_key, "
              f"skipped: {sorted(unmapped)}")


if __name__ == "__main__":
    main()
