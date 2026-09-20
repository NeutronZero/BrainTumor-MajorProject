# User Guide — SB-1.x System (research prototype, not clinical)

## What the system does

Accepts a brain-MRI slice image and returns: predicted class with
calibrated confidence, tumor segmentation + bbox/centroid/area when present,
a system state, and descriptive engineering-observer signals (input
quality, perturbation consistency, reliability summary, visual explanation).

## Reading a result

1. **Model output** (the prediction): class, confidence, classification
   state. `uncertain` means the frozen confidence rule did not pass.
2. **System state**: healthy / tumor_localized / tumor_unlocalized /
   uncertain / degraded, by frozen precedence. `tumor_unlocalized` means a
   tumor class was predicted but no region segmented — a documented
   limitation case, not an error.
3. **Observers** (never change the above): quality verdict, consistency
   agreement, reliability stable/review + basis, Grad-CAM overlay.
4. **Limitations** travel with every report: single-dataset validity,
   noise sensitivity, glioma lower tail, no clinical meaning.

## Running locally

```powershell
pip install -r requirements.txt  # plus torch CPU wheel for your platform
python -m streamlit run app/streamlit/app.py     # UI on :8501
uvicorn app.api.main:app --port 8000             # API
python -m pytest tests/unit tests/integration tests/data  # CI
```

Weights ship via Git LFS (`checkpoints/*/best.pt`); BRISC bytes are external
(Kaggle `briscdataset/brisc2025`) and required only for training/evaluation
reproduction — inference and CI run on synthetic inputs.
