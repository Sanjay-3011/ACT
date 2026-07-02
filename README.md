# Visual Imitation Learning for Robotic Pick-and-Place with Action Chunking Transformers

An end-to-end implementation of **ACT (Action Chunking Transformer)** for closed-loop, vision-based robotic manipulation — trained on self-collected expert demonstrations of a Franka Panda arm performing pick-and-place in PyBullet. The project follows a staged validation methodology (1 → 5 → 55 episodes) to isolate and debug failure modes before committing to full-scale training, rather than training blind on the full dataset and hoping it works.

*(This workspace also includes a separate, completed sub-project: kinematics calibration and URDF modeling for a KUKA KR16-2 industrial arm in ROS 2 / Gazebo — see [Supplementary Work](#supplementary-work-kuka-kr16-gazebo-simulation).)*

---

## Project Status: In Progress — Stage 1 (Single-Episode Validation)

**Current milestone:** Debugging single-episode overfitting to confirm the full pipeline — visual backbone, CVAE latent space, action de-normalization, and closed-loop simulator control — is correct before scaling up.

| Stage | Status |
|---|---|
| 1. Environment setup, data collection (55 expert episodes) | ✅ Complete |
| 2. Single-episode overfitting (pipeline validation) | 🔄 In progress |
| 3. 5-episode overfitting (generalization check) | ⏳ Pending |
| 4. Full 55-episode training | ⏳ Pending |
| 5. Evaluation on unseen seeds + deliverables | ⏳ Pending |

---

## Why Staged Validation

Training an 83.92M-parameter vision transformer directly on the full dataset initially produced a **0% success rate**, with no clear signal for *why*. Two silent bugs were compounding: a frozen visual backbone (`lr_backbone = 0.0`) that prevented the model from ever learning to localize the cube from pixels, and a collapsed CVAE latent space that caused constant zero-action outputs. Debugging both simultaneously on the full dataset would have been slow and ambiguous.

Instead, the project deliberately overfits to 1 episode first: if the model can't perfectly reproduce a single trajectory it was trained on, there's a pipeline bug, not a generalization problem — and it's far cheaper to diagnose on one episode than fifty-five.

---

## Key Technical Findings So Far

**1. Frozen visual backbone → zero learning signal from pixels.**
Default config trains the ACT transformer but leaves ResNet-18 frozen (`lr_backbone = 0.0`), so the policy has no way to learn where the cube is. Fixed by unfreezing with `lr_backbone = 2e-5`.

**2. `kl_weight = 0` → rollout collapses to a constant mean action.**
This was the most instructive bug. With `kl_weight = 0`, training loss converges fine (the CVAE encoder sees ground-truth actions and reconstructs them accurately), but at rollout time there's no ground truth — `z` is instead sampled from the prior `N(0, I)`. With KL unregularized, the encoder's posterior never learns to resemble that prior, so the sampled `z` is out-of-distribution for the decoder, and it falls back to predicting the dataset mean action for every step regardless of input. Setting `kl_weight = 10` resolved the collapse: predicted actions now vary meaningfully over the rollout instead of freezing.

**3. Diagnosed L1 loss dilution across action chunks.**
Even after the KL fix, rollout still fails — the model underpredicts the single largest, most consequential action in the trajectory (the initial reach) while matching the many near-zero "settling" actions later in the chunk. Since L1 loss is averaged uniformly across the full predicted action chunk, and most of any chunk is low-magnitude settling motion, the model can achieve deceptively low aggregate loss while getting the one action that matters most badly wrong. Currently isolating this with per-position loss logging and a decayed learning rate schedule before considering a position-weighted loss function.

**4. Disk I/O was bottlenecking GPU utilization.**
Reading HDF5 image frames from disk at every training step left the GPU idle between batches. Built a custom `PreloadedEpisodicDataset` that loads all episode data into RAM at startup, pushing GPU utilization to ~99%.

---

## Repository Structure

```
.
├── shaka_act/                  # Core ACT implementation
│   ├── train.py                # Full-dataset training entrypoint
│   ├── evaluate_sim.py         # Policy rollout & success-rate evaluation
│   └── training/utils.py       # Dataset stats, policy factory, image preprocessing
├── config/
│   └── config.py               # POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
├── scripts/
│   └── overfit_one_episode.py  # Stage 1: single-episode overfitting + rollout
├── data/
│   └── panda_pick_and_place/   # episode_0.hdf5 ... episode_54.hdf5 (55 expert demos)
├── checkpoints/
│   └── panda_pick_and_place/   # Saved policy checkpoints per experiment
└── kuka_description/           # Supplementary: KUKA KR16 URDF + kinematics calibration
```

---

## Setup

**Requirements:** Python 3.10, PyTorch (GPU), Torchvision, Gymnasium, `panda-gym`, `h5py`, `einops`

```bash
python -m venv venv_act
source venv_act/bin/activate
pip install -r requirements.txt
```

## How to Run

**Stage 1 — Overfit validation (single episode):**
```bash
./venv_act/bin/python scripts/overfit_one_episode.py --task panda_pick_and_place
```

**Full training (55 episodes):**
```bash
./venv_act/bin/python shaka_act/train.py --task panda_pick_and_place
```

**Evaluation (success rate over unseen seeds):**
```bash
./venv_act/bin/python shaka_act/evaluate_sim.py --task panda_pick_and_place
```

---

## Task Definition

* **Environment:** `PandaPickAndPlace-v3` (panda-gym / PyBullet), Franka Emika Panda arm
* **Goal:** relocate a cube to a randomized target position; success if cube-to-target distance ≤ 5 cm
* **Episode length:** max 120 steps, auto-reset on termination
* **Expert data:** 55 successful demonstrations from a custom IK-based state-machine controller (approach → grasp → lift → translate → release)
* **Observations:** 14-dim state (7 joint positions, 1 gripper width, 3 cube coords, 3 target coords) + 640×480 RGB front camera frames
* **Actions:** 14-dim target joint values + gripper command

## Model Configuration

* **Visual backbone:** ResNet-18, fine-tuned (`lr_backbone = 2e-5`)
* **Policy:** ACT (CVAE-based Action Chunking Transformer), `kl_weight = 10`
* **Chunk size:** 100 steps (`num_queries`)
* **Inference:** temporal aggregation with exponential-decay blending across overlapping chunk predictions
* **Optimizer:** AdamW

---

## Planned Deliverables

- [ ] Trained checkpoint on full 55-episode dataset
- [ ] Training loss curves (L1, KL, total)
- [ ] Success rate evaluation on unseen seeds
- [ ] Rollout demonstration videos (MP4)
- [ ] 2-page project report: setup, hyperparameters, results, and failure-mode analysis

---

## Supplementary Work: KUKA KR16 Gazebo Simulation

A separate, completed sub-project simulating a **KUKA KR16-2** industrial arm in ROS 2 / Gazebo:

* **Kinematics calibration:** realigned skewed joint frames from raw CAD coordinates to clean bases parallel to the robot's base frame
* **Inertial realignment:** rotated link inertia matrices (`I_new = R_vis · I_old · R_vis^T`) to satisfy physics constraints and eliminate simulation instability
* **URDF implementation:** standardized package structure under `kuka_description`, with tuned damping/friction thresholds
