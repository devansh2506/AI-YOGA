from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.mcls import load_model_and_labels
from poseguru_core.losses import PoseGuruLoss, PoseGuruLossConfig
from poseguru_core.metrics import compute_mpi_jad_and_pcik
from poseguru_core.recourse import run_recourse
from poseguru_core.refinement import refine_pose


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def main():
    incorrect_csv = DATA_DIR / "incorrect_pose_landmarks.csv"
    if not incorrect_csv.is_file():
        raise FileNotFoundError(
            f"Cannot find incorrect_pose_landmarks.csv at {incorrect_csv}. "
            f"Run scripts/generate_incorrect_poses.py first."
        )

    df = pd.read_csv(incorrect_csv)

    model, label_mapping = load_model_and_labels()
    model.eval()

    # load exemplars
    ex_path = DATA_DIR / "exemplars_yoga.npz"
    ex_data = np.load(ex_path, allow_pickle=True)
    ex_landmarks = ex_data["landmarks"]  # (C,12,2)
    ex_labels = ex_data["labels"]  # (C,)

    # build mapping from class_id -> exemplar landmarks
    exemplar_by_class = {}
    for i, class_id in enumerate(ex_labels):
        exemplar_by_class[int(class_id)] = ex_landmarks[i]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_module = PoseGuruLoss(model, PoseGuruLossConfig())

    all_mpi = []
    all_pcp = []
    all_pcik = []

    batch_size = 32

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]

        x_gt_list = []
        x_prime_list = []
        x_ex_list = []
        tgt_list = []

        for _, row in batch.iterrows():
            pose_name = row["pose_name"]
            class_id = label_mapping[pose_name]

            # correct and incorrect landmarks from row
            correct = []
            incorrect = []
            for j in range(12):
                correct.append([row[f"correct_{j}_x"], row[f"correct_{j}_y"]])
                incorrect.append([row[f"incorrect_{j}_x"], row[f"incorrect_{j}_y"]])

            x_gt_list.append(correct)
            x_prime_list.append(incorrect)
            x_ex_list.append(exemplar_by_class[class_id])
            tgt_list.append(class_id)

        if not x_gt_list:
            continue

        x_gt = torch.tensor(x_gt_list, dtype=torch.float32)
        x_prime = torch.tensor(x_prime_list, dtype=torch.float32)
        x_ex = torch.tensor(x_ex_list, dtype=torch.float32)
        target = torch.tensor(tgt_list, dtype=torch.long)

        x_star, _ = run_recourse(
            x_prime=x_prime,
            x_ex=x_ex,
            target_class=target,
            loss_module=loss_module,
            num_steps=150,
            lr=5e-2,
            device=device,
        )

        x_final, _ = refine_pose(x_prime, x_star)

        mpi, pcp, pcik = compute_mpi_jad_and_pcik(
            x_gt=x_gt,
            x_prime=x_prime,
            x_final=x_final,
            threshold=0.1,
        )

        all_mpi.append(mpi)
        all_pcp.append(pcp)
        all_pcik.append(pcik)

        print(
            f"Batch {start // batch_size}: "
            f"MPIJAD={mpi:.4f}, PCP@MPIJAD={pcp:.4f}, PCIK={pcik:.4f}"
        )

    if all_mpi:
        mpi_mean = float(np.mean(all_mpi))
        pcp_mean = float(np.mean(all_pcp))
        pcik_mean = float(np.mean(all_pcik))

        print("\n=== PoseGuru Evaluation (Yoga) ===")
        print(f"Mean MPIJAD: {mpi_mean:.4f}")
        print(f"Mean PCP@MPIJAD (T=0.1): {pcp_mean:.4f}")
        print(f"Mean PCIK (T=0.1): {pcik_mean:.4f}")
    else:
        print("No valid incorrect joints found for evaluation.")


if __name__ == "__main__":
    main()

