# Dataset files

[The repository README](../README.md#data-protocols) provides the complete data workflow.

DGM4 is the primary benchmark and is obtained from its [official release](https://huggingface.co/datasets/rshaojimmy/DGM4). The annotation files distributed here support the supplementary SAMM-20K protocol.

- [SAMM-20K manifest](SAMM-20K/manifest.json) contains the 20,000 selected training IDs, seed777, source identities and expected reconstructed split hashes.
- [Original SAMM annotations](sources/SAMM) are unchanged publisher files for SAMM-without-CAP at revision `9ff6eafbe04f13e9ad8df6bc09a63f9f7a24f3f4`: 190,970 training, 20,000 validation and 50,000 test rows.
- [LICENSE](sources/SAMM/LICENSE) and [NOTICE](sources/SAMM/NOTICE) apply to those original annotation files.

```bash
python -B scripts/heri.py reconstruct-samm --source-root data/sources/SAMM --output-root data/SAMM
```

The runtime protocol uses 20,000 training, 20,000 validation and 50,000 test rows. Reconstructed annotations, images, other datasets and generated products are excluded from version control. Images must be obtained separately from the [official dataset](https://huggingface.co/datasets/SJJ0854/SAMM).
