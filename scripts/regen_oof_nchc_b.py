"""Regenerate OOF .npz files for the 3 NCHC B-plan archs.

User scp'd 15 best.pt checkpoints (cwt_filtered/cwt_paul/stft × maxvit_tiny@384 × 5 folds)
but the corresponding _oof.npz files weren't transferred. We reload each ckpt and run
inference on the val fold to recover the OOF logits + labels + sample_ids.

GPU 0 (free), since GPU 1+2 are running heavy aug.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.datasets.fold_datasets import FoldSpecDataset
from neuromm26_baseline.models.spec_cnn_concat import ConcatSpecCNN

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
RES = REPO / "neuromm26_results"
FOLD_CSV = str(REPO / "fold_df_fixed.csv")
DEVICE = "cuda:0"
BATCH = 64

# (spec_type, exp prefix without fold/seed suffix)
SPECS = [
    ("cwt_filtered", "concat_cwt_filtered_fold__maxvit_tiny_tf_384_in1k"),
    ("cwt_paul",     "concat_cwt_paul_fold__maxvit_tiny_tf_384_in1k"),
    ("stft",         "concat_stft_fold__maxvit_tiny_tf_384_in1k"),
]


def regen_one(spec, prefix, fold):
    exp = f"{prefix}__fold{fold}__seed0"
    ckpt_path = RES / "checkpoints" / exp / "best.pt"
    out_path = RES / "predictions" / f"{exp}_oof.npz"
    if out_path.exists():
        print(f"  ✓ already exists: {out_path.name}")
        return
    if not ckpt_path.exists():
        print(f"  ✗ MISSING ckpt: {ckpt_path}")
        return

    st = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ConcatSpecCNN(num_classes=1, backbone="maxvit_tiny_tf_384.in1k",
                          target_size=384, pretrained=False).to(DEVICE).eval()
    model.load_state_dict(st["model_state_dict"])

    spec_root = str(REPO / "neuromm26_datasets/processed/features" / spec)
    ds = FoldSpecDataset(spec_root=spec_root, fold_csv=FOLD_CSV,
                         fold_idx=fold, split="val", preload_in_memory=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0,
                        collate_fn=neuromm_collate)

    L, Y, S = [], [], []
    with torch.no_grad():
        for b in loader:
            b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            with torch.amp.autocast("cuda", enabled=True):
                out = model({"spec": b["spec"]})
            L.append(out["main"].view(-1).float().cpu().numpy())
            Y.append(b["label"].view(-1).cpu().numpy())
            S.extend(b["sample_id"])
    L = np.concatenate(L); Y = np.concatenate(Y).astype(np.int32)
    np.savez(out_path, sample_ids=np.array(S), logits=L, labels=Y)

    # quick AUPRC
    from sklearn.metrics import average_precision_score
    p = 1 / (1 + np.exp(-L.astype(np.float64)))
    ap = average_precision_score(Y, p)
    n_pos = int(Y.sum()); n = len(Y)
    print(f"  ✓ {exp}  n={n} n_pos={n_pos} val_AUPRC={ap:.4f}")
    del model
    torch.cuda.empty_cache()


def main():
    RES.joinpath("predictions").mkdir(parents=True, exist_ok=True)
    for spec, prefix in SPECS:
        print(f"\n=== {prefix} ===")
        for f in range(5):
            regen_one(spec, prefix, f)


if __name__ == "__main__":
    main()
