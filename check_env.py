from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
SCR = ROOT / "scr"
if str(SCR) not in sys.path:
    sys.path.insert(0, str(SCR))

from pot import pot_eval_stable  # type: ignore


def main() -> None:
    x_train = np.random.rand(200).astype(float)
    x_test = np.random.rand(120).astype(float)
    labels = np.zeros(len(x_test), dtype=int)
    result, pred = pot_eval_stable(x_train, x_test, labels, q=1e-5, level=0.01, th_scale=1.0)

    out = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pot_threshold": float(result["threshold"]),
        "pot_pred_len": int(len(pred)),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
