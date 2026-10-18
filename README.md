# draftfm-training

The GPU half of DraftFM stage 3. Everything lives in [`training/`](training/)
— the trainer, the one-shot runner a rented box executes, the Ollama
Modelfile, and `draftfm-train-synth-v1.zip` (14,062 rows from 385 **synthetic**
foundation plans; the 20 frozen Golden-Set test sources are excluded by id).

Start at [`training/README.md`](training/README.md).

Until 2026-10-18 every file here existed twice, at the root and under
`training/`, and the two copies had drifted (the root trainer was two fixes
behind). The pods ran `training/`, so that is the copy that survived.

Nothing confidential belongs in this repository. It is public, and only the
synthetic set may be pushed to it or uploaded to a rented machine; the
sanitised BVES corpus stays on `bvesws1`. The orchestrator in the sibling
repository (`bves-drafting`, `tools/rent_and_train.py`) enforces that as a
check on every upload rather than as a flag.
