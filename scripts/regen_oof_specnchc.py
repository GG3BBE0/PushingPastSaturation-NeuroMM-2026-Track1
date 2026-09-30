"""Regenerate OOF .npz for ONE specnchc concat arch (5 folds) from its ckpts (ckpt-only return).
Reload each best.pt, infer on the val fold, save OOF (sample_ids, logits, labels).
Usage: python scripts/regen_oof_specnchc.py <spec_type> <prefix> <backbone> <target_size> <gpu>
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score as aps
from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.datasets.fold_datasets import FoldSpecDataset
from neuromm26_baseline.models.spec_cnn_concat import ConcatSpecCNN

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
RES = REPO / "neuromm26_results"
FOLD_CSV = str(REPO / "fold_df_fixed.csv")
spec_type, prefix, backbone, target_size, gpu = sys.argv[1:6]
target_size = int(target_size); DEV = f"cuda:{gpu}"

for fold in range(5):
    exp = f"{prefix}__fold{fold}__seed0"
    out = RES / "predictions" / f"{exp}_oof.npz"
    if out.exists():
        print(f"  ✓ {exp} exists", flush=True); continue
    ck = RES / "checkpoints" / exp / "best.pt"
    if not ck.exists():
        print(f"  ✗ MISSING {ck}", flush=True); continue
    st = torch.load(ck, map_location=DEV, weights_only=False)
    m = ConcatSpecCNN(num_classes=1, backbone=backbone, target_size=target_size, pretrained=False).to(DEV).eval()
    m.load_state_dict(st["model_state_dict"])
    ds = FoldSpecDataset(spec_root=str(REPO / "neuromm26_datasets/processed/features" / spec_type),
                         fold_csv=FOLD_CSV, fold_idx=fold, split="val", preload_in_memory=True)
    ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    L, Y, S = [], [], []
    with torch.no_grad():
        for b in ld:
            b = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            with torch.amp.autocast("cuda", enabled=True):
                o = m({"spec": b["spec"]})
            L.append(o["main"].view(-1).float().cpu().numpy()); Y.append(b["label"].view(-1).cpu().numpy()); S.extend(b["sample_id"])
    L = np.concatenate(L); Y = np.concatenate(Y).astype(np.int32)
    np.savez(out, sample_ids=np.array(S), logits=L, labels=Y)
    ap = aps(Y, 1 / (1 + np.exp(-L.astype(np.float64))))
    print(f"  ✓ {exp} n={len(Y)} val_AUPRC={ap:.4f}", flush=True)
    del m; torch.cuda.empty_cache()
