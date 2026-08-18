# Sealed result artifacts

This directory contains compact, machine-readable copies of the formal summaries and their independent-verification receipts.

- `final14/`: fresh MP3D mixed Novel/Revisit evaluation and latency audit.
- `hm3d/`: held-out HM3D Revisit transfer.
- `three_leg/`: actual-online Novel-Novel-Revisit strict retest.

The raw RGB/depth traces, Habitat assets and model checkpoints are intentionally excluded. Run `python paper/verify_release.py` from the repository root to verify hashes and headline counts.
