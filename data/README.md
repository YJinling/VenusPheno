# Data Layout

- `data/raw/`: original raw files, large inputs, or private datasets. Usually not committed.
- `data/processed/`: cleaned metadata, small examples, or lightweight shared inputs.

Keep large HDF5 files, raw embeddings, and private metadata out of the public repository unless they are intended for release.
