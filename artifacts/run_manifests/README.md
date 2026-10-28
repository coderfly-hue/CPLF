# Genuine-run manifests (not yet populated)

No original run manifests were supplied in this release. Do not create a
five-seed manifest merely by copying the manuscript's results table.

For each genuinely completed prediction or generator training run, capture
seed, actual model revision, configuration, software versions, source split
hashes and hashes/locations of the produced checkpoint/adapter and output
files. Use `scripts/capture_run_manifest.py` with existing source files.

Model checkpoints and adapters may be hosted at immutable external locations
when redistribution and license terms permit. Include the source URI, size,
SHA-256 and retrieval instructions in the corresponding records. Until then,
the strict canonical evidence gate is expected to fail.
