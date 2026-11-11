# Dataset storage

Datasets are obtained separately under their release terms. Image files and annotations are not versioned.

```text
data/
  DGM4/{train,val,test}.json
  DGM4/images/...
  SAMM/{train,val,test}.json
  SAMM/images/...
  sources/SAMM/{train,val,test}.json   # original annotations for reconstruction
```

Use `python -B scripts/heri.py reconstruct-samm --source-root data/sources/SAMM --output-root data/SAMM` to reconstruct the paper subset, then run `prepare-data` with the appropriate dataset config. Reconstruction validates the frozen byte hashes before writing; it does not copy images.
