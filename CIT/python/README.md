# CIT Python replication

Reference Python implementation for the numerical results accompanying the Consensus Instability Theorem.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
cit-run --fast
```

Publication run:

```bash
cit-run --publication
```

Outputs include the paired response kernel, confidence bands, susceptibility metrics, generator cross-check, threshold estimates, and the four-panel numerical figure.
