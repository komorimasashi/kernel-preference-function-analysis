# Empirical Data Dictionary

The `data/` directory contains one CSV file for each of 20 participants. Each
file contains the 50 experimental trials for that participant. The file number
is an arbitrary study identifier and is not linked to a name or contact detail.

| Variable | Description |
| --- | --- |
| `aspect_ratio` | Width-to-height ratio of the presented rectangle, equal to `4 ** random_value`. |
| `beauty` | Beauty rating on a seven-point ordinal scale (`1` = not at all beautiful; `7` = very beautiful). |
| `random_value` | Analysis coordinate `x = log_4(aspect_ratio)`, sampled from `[-1, 1]`. |

The analysis reads `random_value` as the predictor and z-standardizes `beauty`
separately within each participant. No names, contact details, dates, or other
direct identifiers are included in these files.
