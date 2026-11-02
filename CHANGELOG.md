# Changelog

Versions follow [semantic versioning](https://semver.org): a new major version when the matcher's decisions change in
substance (new models, a new threshold), a minor version for new tools or data, a patch for fixes.

## 1.0.0 (3 November 2026)

First release, the version OpenAlex runs from the nightly build of 3 November 2026.

- Identifiers, then six candidate searches, hard rules (104 vetoed type pairs, undeclared Kaggle versions), XGBoost
  (95 features, 781 trees; a record whose best candidate scores under 0.01 is a new work), then Qwen3.5-9B with a LoRA
  adapter on the top 10 (joins at 0.910, or when copies of one paper together reach 0.910).
- Trained on 100,000 incoming records labelled by Claude Opus 5.5 plus 11,728 shortlists of older works.
- Benchmarks: 99.1% right when it merges, 92.2% of duplicates found, on 8 judged nights (the title + author rule it replaces:
  78.6% / 74.3%).
- `matcher/`: the same stack for any record through the public API; `pipeline/`: the production job.
