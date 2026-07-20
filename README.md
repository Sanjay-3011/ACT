# Imitation Learning on Franka Panda Pick-and-Place using Action Chunking Transformer (ACT)

An end-to-end implementation of **ACT (Action Chunking Transformer)** for closed-loop, vision-based robotic manipulation — trained on self-collected expert demonstrations of a Franka Panda arm performing pick-and-place in PyBullet. The project follows a staged validation methodology (1 → 5 → 55 episodes) to isolate and debug failure modes before committing to full-scale training, rather than training blind on the full dataset and hoping it works.

---

## Project Status: In Progress — Stage 1 (Single-Episode Validation Complete)

**Current milestone:** Single-episode overfitting on Episode 1 successfully validated with a 100% success rate (**4.62 cm placement error** at Step 67, below the 5.0 cm success threshold). Resolved initial-reach underprediction by implementing a fixed step-function chunk-position weighting scheme (`weight = 5.0` for `t < 5`, `weight = 0.2` for `t >= 5`). Ready to scale to multi-episode training.

| Stage | Status |
|---|---|
| 1. Environment setup, data collection (55 expert episodes) | ✅ Complete |
| 2. Single-episode overfitting (pipeline validation) | ✅ Complete (Success at Step 67, 4.62 cm error) |
| 3. 5-episode overfitting (generalization check) | 🔄 Next |
| 4. Full 55-episode training | ⏳ Pending |
| 5. Evaluation on unseen seeds + deliverables | ⏳ Pending |

---

## Why Staged Validation

Training an 83.92M-parameter vision transformer directly on the full dataset initially produced a **0% success rate**, with no clear signal for *why*. Two silent bugs were compounding: a frozen visual backbone (`lr_backbone = 0.0`) that prevented the model from ever learning to localize the cube from pixels, and a collapsed CVAE latent space that caused constant zero-action outputs. Debugging both simultaneously on the full dataset would have been slow and ambiguous.

Instead, the project deliberately overfits to 1 episode first: if the model can't perfectly reproduce a single trajectory it was trained on, there's a pipeline bug, not a generalization problem — and it's far cheaper to diagnose on one episode than fifty-five.

---

## Key Technical Findings So Far

**1. Frozen visual backbone → zero learning signal from pixels.**
Default config trains the ACT transformer but leaves ResNet-18 frozen (`lr_backbone = 0.0`), so the policy has no way to learn where the cube is. Fixed by unfreezing the backbone, currently set to `lr_backbone = 1e-5` for stable visual feature learning.

**2. `kl_weight = 0` → rollout collapses to a constant mean action.**
This was the most instructive bug. With `kl_weight = 0`, training loss converges fine (the CVAE encoder sees ground-truth actions and reconstructs them accurately), but at rollout time there's no ground truth — `z` is instead sampled from the prior `N(0, I)`. With KL unregularized, the encoder's posterior never learns to resemble that prior, so the sampled `z` is out-of-distribution for the decoder, and it falls back to predicting the dataset mean action for every step regardless of input. Setting `kl_weight = 10` resolved the collapse: predicted actions now vary meaningfully over the rollout instead of freezing.

**3. Diagnosed a real placement bug, and made a deliberate scoping decision.**
The `DESCEND_TO_TARGET` state was computing the drop position using only the target's (x, y) coordinates combined with a fixed table-level height, ignoring the target's actual z-coordinate — even though `PandaPickAndPlace-v3` randomizes targets across a full 3D range including elevated, mid-air positions. This caused a measured 15.24 cm placement error, almost entirely in the z-axis (cube placed on the table at z≈2cm while the true target was floating at z≈17cm; x/y error alone was under 2cm). Fixed the underlying logic to compute the descent trajectory using the target's full 3D position, not a fixed table height.

As a deliberate scoping decision for this stage, the environment's target range is currently restricted to table-level only (z=0), to isolate and validate ground-level placement before reintroducing elevated targets as a planned next step. This is a temporary simplification for debugging purposes, not a permanent constraint of the approach.

**4. Re-collected 55-Episode High-Precision Dataset.**
Re-recorded the entire demonstration dataset (all 55 episodes) with the new 150-step trajectory, including explicit descend-to-target, gripper release, and retract states — ensuring demonstrations end with an actual completed placement (gripper open, arm withdrawn), not just proximity while still gripping. All files verified at 138 MB each.

**5. Image Downsampling Speedup.**
Downsampled visual inputs from 480×640 to 240×320 on-the-fly during dataloading and rollouts, reducing ResNet-18 computation and yielding a ~30% training speedup.

**6. Confirmed Per-Position Loss Mismatch & Dilution.**
Implemented per-position loss logging to isolate the initial-reach underprediction. Confirmed that L1 loss at chunk positions 0 and 1 was originally **~3.5x higher** than the average L1 loss across the whole 100-step chunk. Because the majority of any 100-step action chunk is spent standing still or making micro-adjustments near the target, the network achieved low aggregate L1 loss by fitting the flat zero region, while under-optimizing high-amplitude reach actions.

**7. Disk I/O was bottlenecking GPU utilization.**
Reading HDF5 image frames from disk at every training step left the GPU idle between batches. Built a custom `PreloadedEpisodicDataset` that loads all episode data into RAM at startup, pushing GPU utilization to ~99%.

**8. Fixed Step-Function Loss Weighting Solved Initial-Reach Dilution & Achieved 100% Success.**
Implemented a fixed step-function loss weighting scheme across the 100-step action chunk: `weight = 5.0` for `t < 5` and `weight = 0.2` for `t >= 5`. This gave direct, exact control over the early-to-tail loss ratio without unpredicted cumulative decay effects. Retraining Episode 1 with this weighting (`kl_weight = 10`, `lr = 5e-5`, `lr_backbone = 5e-6`, 300 epochs) achieved a **100% success rate** with **4.62 cm placement error** at **Step 67**.

---

## Next Steps

With Stage 1 (single-episode overfitting) cleanly validated and reproducible:

- **Stage 3 (5-Episode Training)**: Scale training to a 5-episode subset to evaluate multi-trajectory policy learning and intermediate generalization.
- **Stage 4 (Full 55-Episode Training)**: Scale to the entire 55-episode dataset for 800 epochs.

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
├── evaluate.py              # Rollout execution script on physical Follower arm
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
./venv_act/bin/python scratch/overfit_one_episode.py --task panda_pick_and_place
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
* **Goal:** relocate a cube to a target position; success if cube-to-target distance ≤ 5 cm
* **Episode length:** max 150 steps, auto-reset on termination
* **Note on success criterion:** the environment's default check is proximity-only (distance ≤ 5cm), which does not require the gripper to actually release the cube. To ensure demonstrations reflect a genuinely completed placement, the expert controller's own state machine additionally requires the gripper to open and the arm to retract before an episode is logged as complete.
* **Current scope:** target positions are currently restricted to table-level (z=0) placements as a debugging simplification; elevated/mid-air target placement (part of the environment's default task) is a planned extension once ground-level policy training is validated.
* **Expert data:** 55 successful demonstrations from a custom IK-based state-machine controller (approach → grasp → lift → translate → descend to target → release → retract)
* **Observations:** 14-dim state (7 joint positions, 1 gripper width, 3 cube coords, 3 target coords) + 640×480 RGB front camera frames
* **Actions:** 14-dim target joint values + gripper command

## Model Configuration

* **Visual backbone:** ResNet-18, fine-tuned (`lr_backbone = 1e-5`)
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