# SynBio2026

Computational pipeline for ranking **thermostable GFP variants** from designed sequence libraries. Combines sequence generation, fluorescence prediction, and thermostability screening to prioritize candidates for experimental validation.

---

## Pipeline

```
ProteinMPNN  →  brightness scoring (ESM-2)  →  thermostability screen  →  ranked candidates
   FASTA              regression model              PPtStab + ML
```

1. **Design** — generate mutant sequences with [ProteinMPNN](https://github.com/dauparas/ProteinMPNN).
2. **Brightness** — predict relative fluorescence from sequence using an ESM-2 regression head (`ResLinear` on pooled embeddings).
3. **Thermostability** — estimate melting temperature with [PPtStab](https://github.com/raghavagps/pptstab), enrich features with Biopython descriptors and ESM-2 embeddings, then classify with an ExtraTrees ensemble (stable if functional thermostability > 72 °C).
4. **Select** — rank variants by predicted thermostability probability and filter for downstream assays.

---

## Repository layout

```
SynBio2026/
├── src/
│   ├── FluoreModel/     # ESM-2 GFP regression models and training utilities
│   └── MPNN/            # FASTA dataset loaders for designed sequences
├── scripts/
│   ├── score_sequences.py   # batch brightness inference on .fa files
│   └── train_model.py       # train GFP regression model
├── notebooks/           # preprocessing, training, thermostability, validation
├── slurm/               # cluster job scripts
├── envs/                # per-task conda environments
└── data/model_weights/  # pre-trained brightness model weights
```

---

## Setup

Each stage uses its own conda environment. Install the one you need:

| Task | Environment file |
|------|------------------|
| GFP brightness model | `environment.yml` or `envs/gfpmodel.yml` |
| Thermostability (PPtStab) | `envs/therm.yml` |
| ProteinMPNN | `envs/proteinmpnn_cu12.4.yml` |
| ThermoMPNN | `envs/thermompnn.yaml` |

```bash
conda env create -f environment.yml
conda activate esm2gfp_env
```

For PPtStab, clone and install separately:

```bash
git clone https://github.com/raghavagps/pptstab.git
conda env create -f envs/therm.yml
```

**Requirements:** Python 3.11+, CUDA-capable GPU recommended for ESM-2 inference and training.

---

## Usage

### Score sequences for brightness

Place ProteinMPNN output (`.fa` files) in an input directory, then run:

```bash
cd scripts
python score_sequences.py \
  --input_dir ../data/mpnn_sequences/MPNN_seqs/ \
  --model_weights ../data/model_weights/regression_model.weights \
  --output ../data/scored_sequences/scores.csv
```

Output CSV columns: `id`, `struct`, `brightness`, `sequence`.

### Train the brightness model

Requires training/validation CSVs with `seq` and `rel_brightness` columns (see `notebooks/preprocess_data.ipynb`):

```bash
cd scripts
python train_model.py
```

### Thermostability screening

Run `notebooks/therm.ipynb` for PPtStab-based Tm prediction, then `notebooks/classification.ipynb` for the ExtraTrees thermostability classifier. See `notebooks/anal_scored_seqs.ipynb` for filtering and ranking results.

### Cluster jobs

```bash
sbatch slurm/score_sequences.sh
sbatch slurm/train_model.sh
```

Adjust partition, GPU, and conda environment names in the SLURM scripts for your cluster.

---

## Method summary

Thermostability labels follow experimental functional thermostability ([Nat Commun, 2023](https://doi.org/10.1038/s41467-023-38099-z)): variants above **72 °C** are thermostable. The classifier uses PPtStab-predicted Tm ([Sci Rep, 2025](https://doi.org/10.1038/s41598-025-98667-9)), physicochemical descriptors (Biopython `ProteinAnalysis`), amino acid composition, sequence length, and mean-pooled [ESM-2](https://doi.org/10.1101/2022.12.21.521521) embeddings. Final model: **ExtraTrees** (1,500 trees), stratified 75/25 train–test split.

Brightness prediction uses ESM-2 (`esm2_t33_650M`) embeddings with a residual linear regression head trained on relative brightness data.

---

## References

- PPtStab: [10.1038/s41598-025-98667-9](https://doi.org/10.1038/s41598-025-98667-9)
- GFP thermostability dataset: [10.1038/s41467-023-38099-z](https://doi.org/10.1038/s41467-023-38099-z)
- ESM-2: [10.1101/2022.12.21.521521](https://doi.org/10.1101/2022.12.21.521521)

---

## License

See repository maintainers for usage terms. Third-party tools (PPtStab, ProteinMPNN, ESM-2) carry their own licenses.
