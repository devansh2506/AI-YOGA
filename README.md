## PoseGuru Yoga – Explainable Pose Correction

This repo implements a PoseGuru-style human pose correction system on your Yoga dataset, following the CVPRW 2025 paper *“PoseGuru: Landmarks for Explainable Pose Correction using Exemplar-Guided Algorithmic Recourse”*.

It uses:
- MediaPipe BlazePose as the pose estimator `Mest`
- A 12-keypoint landmark-based classifier `Mcls`
- Exemplar-guided algorithmic recourse with four losses (prediction, stick-length, landmarks, angles)
- A refinement step and action vector
- A real-time webcam demo for live pose correction

---

### 🚀 Quick Start (Running the System)

If you just want to run the real-time AI Yoga assistant:

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python run.py
   ```

3. **Follow the On-Screen Instructions**
   - Enter the name of the yoga pose you want to practice (e.g., "tree pose", "cobra").
   - Confirm the match found by the system.
   - The webcam feed will open. A skeleton overlay will show your pose (in red) and provide real-time textual feedback.
   - An "Ideal Pose" image will appear in the top-right corner to guide you.
   - Press **`q`** to quit.

---

### 🛠️ Manual Setup & Pipeline Details

If you want to retrain the model, generate new data, or understand the internals, follow the steps below.

### 1. Environment Setup

- **Python**: 3.9+ recommended
- It is recommended to use a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

Make sure your Yoga images live under:
- `yoga data/train`
- `yoga data/val`
- `yoga data/test`

And ensure `pose_landmarker_full.task` (MediaPipe model) is in the repo root.

### 2. Generate Landmarks (12-keypoint) CSVs

This step runs MediaPipe on your yoga images and writes split-wise landmark CSVs under `data/`.

From the repo root:

```bash
python scripts/generate_landmarks.py
```

This produces:
- `data/yoga_pose_landmarks_train.csv`
- `data/yoga_pose_landmarks_val.csv`
- `data/yoga_pose_landmarks_test.csv`

Each row contains `pose_name` and 12 keypoints (x,y) normalized to [0,1].

### 3. Train the Pose Classifier `Mcls`

Train a shallow MLP on the landmark data:

```bash
python scripts/train_classifier.py
```

This saves:
- `models/mcls_yoga.pt` – classifier weights
- `models/label_mapping.json` – mapping from `pose_name` to class index

### 4. Select Class Exemplars

For each pose class, we pick the highest-confidence training sample as an exemplar:

```bash
python scripts/select_exemplars.py
```

Output:
- `data/exemplars_yoga.npz` containing:
  - `landmarks`: (C, 12, 2) landmark arrays
  - `labels`: class ids
  - `pose_names`: pose names for each exemplar

### 5. Generate Incorrect Poses for Evaluation

We synthetically create biomechanically plausible incorrect poses by perturbing a single joint angle per sample, following the paper’s constraints:

```bash
python scripts/generate_incorrect_poses.py
```

This writes:
- `data/incorrect_pose_landmarks.csv`

Each row contains:
- `pose_name`, `segment`, `joint_name`
- 12×(x,y) for the **correct** pose
- 12×(x,y) for the **incorrect** pose

### 6. Offline PoseGuru Evaluation (MPIJAD & PCIK)

Run the full PoseGuru pipeline (exemplar-guided recourse + refinement) on the incorrect poses and compute the paper’s metrics:

```bash
python scripts/evaluate_poseguru.py
```

This computes and prints:
- **MPIJAD** – Mean Per Incorrect Joint Absolute Deviation
- **PCP@MPIJAD (T=0.1)** – fraction of poses whose mean incorrect-joint deviation is ≤ 0.1
- **PCIK (T=0.1)** – Percentage of Corrected Incorrect Keypoints

### 7. Real-Time PoseGuru Demo (Webcam)

`run.py` is the main entry point, but you can also run the script directly:

```bash
python scripts/realtime_poseguru.py
```

What it does per frame:
- Captures a frame from your webcam
- Runs MediaPipe BlazePose to get 33 keypoints and reduces them to 12 landmarks
- Classifies the pose with `Mcls`
- Fetches the class exemplar
- Runs a short gradient-based recourse optimization and the refinement step
- Overlays:
  - Original skeleton (red)
  - Corrected skeleton (green)
  - Predicted pose name in text

Press **`q`** to exit the demo window.

### 8. Code Layout

- `yoga data/` – your yoga images for train/val/test
- `pose_landmarker_full.task` – MediaPipe pose model
- `data/`
  - `landmarks_dataset.py` – PyTorch dataset for 12-keypoint CSVs
  - `yoga_pose_landmarks_*.csv` – generated landmark splits
  - `exemplars_yoga.npz` – pose exemplars per class
  - `incorrect_pose_landmarks.csv` – generated incorrect poses
- `models/`
  - `mcls.py` – landmark MLP classifier + save/load helpers
- `poseguru_core/`
  - `losses.py` – `Cpred`, `Cstick`, `Cland`, `Cangle`, `PoseGuruLoss`
  - `recourse.py` – gradient-based exemplar-driven optimization
  - `refinement.py` – one-segment-at-a-time refinement + action vector
  - `metrics.py` – MPIJAD, PCP@MPIJAD, PCIK
- `scripts/`
  - `generate_landmarks.py` – MediaPipe → landmark CSVs
  - `train_classifier.py` – train `Mcls`
  - `select_exemplars.py` – build exemplar set
  - `generate_incorrect_poses.py` – synthetic incorrect pose generation
  - `evaluate_poseguru.py` – offline metric evaluation
  - `realtime_poseguru.py` – real-time webcam PoseGuru demo

