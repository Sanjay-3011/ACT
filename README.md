# Imitation Learning on Franka Panda Pick-and-Place using Action Chunking Transformer (ACT)

An end-to-end implementation of **ACT (Action Chunking Transformer)** for closed-loop, vision-based robotic manipulation — trained on self-collected expert demonstrations of a Franka Panda arm performing pick-and-place in PyBullet. The project follows a staged validation methodology (1 → 5 → 55 episodes) to isolate and debug failure modes before committing to full-scale training, rather than training blind on the full dataset and hoping it works.

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

**3. Diagnosed L1 loss dilution across action chunks — current blocker.**
Even after the KL fix, rollout still fails — 0/1 success on the single episode the policy was directly overfit to. L1 loss plateaus around 0.05–0.06 and does not improve further with additional epochs, while the model badly underpredicts the single largest, most consequential action in the trajectory (the initial reach):

| Step | True gripper action | Predicted | True joint 0 | Predicted |
|---|---|---|---|---|
| 0 | 0.4000 | -0.1921 | 1.0000 | -0.0070 |
| 1 | 0.0486 | -0.0868 | 0.7904 | -0.0082 |

From step 2 onward, predicted and true actions both hover near zero — but that's because the true trajectory *also* settles by then, not because the model is tracking it. Working theory: since L1 loss is averaged uniformly across the full 100-step predicted action chunk, and most of any chunk is low-magnitude settling motion, the model can achieve deceptively low aggregate loss by nailing the easy majority while getting the one action that actually matters badly wrong. Not yet confirmed — currently isolating with per-position loss logging.

**4. Disk I/O was bottlenecking GPU utilization.**
Reading HDF5 image frames from disk at every training step left the GPU idle between batches. Built a custom `PreloadedEpisodicDataset` that loads all episode data into RAM at startup, pushing GPU utilization to ~99%.

---

## Open Questions

Currently seeking input on the loss-dilution blocker above:

- Does the loss-dilution theory hold, or is there a more likely explanation given the L1/KL numbers observed?
- Is a position-weighted L1 loss (upweighting early chunk positions) the standard fix for this, or is there a more established approach in the ACT / imitation learning literature?
- Is near-perfect single-episode overfitting a reasonable bar to clear before moving to the 5-episode stage, or is this staged validation plan too strict?

---

## Repository Structure

```
.
shaka_act/
├── config/
│   └── config.py            # Hyperparameters and environment configurations (LR, state dimension, camera names, nheads)
├── checkpoints/
│   └── .gitkeep             # Directory where trained checkpoints (.ckpt files) are saved during train.py execution
├── data/
│   └── .gitkeep             # Directory where PyBullet expert datasets are saved
├── training/
│   ├── policy.py            # ACTPolicy and CNNMLPPolicy wrapper classes that orchestrate DETR model updates
│   └── utils.py             # Dataloader utilities (EpisodicDataset) and dataset normalizer (get_norm_stats)
│
│  # --- Simulator Scripts (PyBullet) ---
├── record_episodes_sim.py   # State-machine IK expert script used to generate and log 55 demo episodes
├── train.py                 # Main training script to train the model on the generated HDF5 dataset
├── evaluate_sim.py          # Evaluation rollout script in simulator incorporating closed-loop temporal aggregation
│
│  # --- Physical Robot Scripts (Dynamixel hardware setup) ---
├── record_episodes.py       # Data logger script for human demonstrations on physical arms
├── evaluate.py              # Rolout execution script on physical Follower arm
├── teleoperation.py         # Leader-to-Follower direct torque controller mapping script
├── robot.py                 # Low-level serial communication configurations for hardware joint encoders
├── dynamixel.py             # Lower-level command packet mapping for Dynamixel servo protocols
│
│  # --- Testing & Documentation ---
├── cam.ipynb                # Jupyter notebook for checking camera feeds
├── replay_episode.ipynb     # Jupyter notebook to play back and verify logged demonstration files
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
