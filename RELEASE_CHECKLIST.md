# Release Checklist

## Validation for version 0.2.0 (September 9, 2026)

- [x] Confirm that sharing the anonymized participant-level ratings is
  consistent with the ethics approval and participant consent.
- [x] Confirm code license (MIT) and data/output license (CC BY 4.0).
- [x] Run the empirical selected-parameter fit and regenerate all empirical
  figures.
- [x] Reproduce the complete 231-condition empirical cross-validation.
- [x] Regenerate all simulation figures from the saved result tables.
- [x] Run at least one reduced smoke test for each Monte Carlo workflow.
- [x] Validate the manuscript-facing numerical results and recompute simulation
  summary statistics from the saved replication tables.
- [x] Check the empirical model against an independent feature-space solution
  and PCA invariants (four numerical checks).
- [x] Label the historical standardization check as a working record outside
  the current manuscript.

## Publication steps still to complete

- [ ] Commit and push the current data, code, results, and documentation to GitHub.
- [ ] Change the GitHub repository to Public.
- [ ] Create the GitHub release `v0.2.0` from the verified commit.
- [ ] Update the manuscript data availability statement after public access is verified.

## Later archival and article metadata

- [ ] Replace the provisional citation metadata with the accepted article
  citation when available.
- [ ] If archiving on Zenodo, enable the integration and archive the chosen
  release when the manuscript and reproducibility materials are finalized.
- [ ] Add the Zenodo DOI to `README.md`, `CITATION.cff`, and the manuscript data
  availability statement.
