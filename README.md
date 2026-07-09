# Imitation Learning on Franka Panda Pick-and-Place using Action Chunking Transformer (ACT)

---

## What This Repo Is

This repository implements **ACT (Action Chunking Transformer)** — a vision-based imitation learning policy — for a robotic pick-and-place task using a simulated Franka Panda arm in PyBullet (`PandaPickAndPlace-v3`). The end goal is a closed-loop policy that takes in a camera image and joint state at every timestep and outputs low-level joint commands to autonomously pick up a cube and place it at a target location, trained entirely from self-collected expert demonstrations — no hand-coded control at inference time.

It also contains a separate, completed sub-project: kinematics calibration and URDF modeling for a KUKA KR16-2 industrial arm in ROS 2 / Gazebo (see bottom of this document).

---

## What I'm Doing

Building the full pipeline end to end:
1. Collecting expert demonstrations of pick-and-place using a custom IK-based controller.
2. Training an ACT policy on those demonstrations to imitate the behavior visually (from camera images), not from privileged simulator state.
3. Evaluating the trained policy in closed-loop rollout to see if it can actually complete the task on its own.

---

## How I'm Doing It

**Staged validation approach (1 → 5 → 55 episodes):** rather than training directly on the full 55-episode dataset and hoping it works, I deliberately overfit to a single episode first. If the model can't perfectly reproduce one trajectory it was trained on, that points to a pipeline bug — not a generalization problem — and it's far cheaper to diagnose on one episode than fifty-five. Only once single-episode overfitting succeeds do I move to 5 episodes (checking generalization across varied cube/target positions), and then the full 55.

**Pipeline components:**
- **Expert data collection:** a custom IK-based finite state machine (approach → grasp → lift → translate → descend to target → release → retract) generates 55 demonstration episodes, logged as HDF5 files with camera frames, joint states, and actions.
- **Model:** ResNet-18 visual backbone + ACT transformer with a CVAE-based policy head, using temporal aggregation at inference (blending overlapping predicted action chunks via exponential-decay weighting).
- **Training speedups:** a custom `PreloadedEpisodicDataset` caches all episode data in RAM to eliminate disk I/O bottlenecks, and camera images are downsampled (480×640 → 240×320) to reduce backbone compute.

---

## What I'm Currently Doing

**Stage 1 of 4 — single-episode overfitting, in progress.** Specifically debugging why the policy underpredicts the single most important action in the trajectory — the initial reach — even when trained repeatedly on just one demonstration episode. Currently isolating this to a loss-averaging problem across the predicted action chunk (see Current Issues below), and deciding on a fix before moving to Stage 2.

| Stage | Status |
|---|---|
| 1. Environment setup, data collection (55 expert episodes) | ✅ Complete |
| 2. Single-episode overfitting (pipeline validation) | 🔄 In progress — current blocker below |
| 3. 5-episode overfitting (generalization check) | ⏳ Pending |
| 4. Full 55-episode training | ⏳ Pending |
| 5. Evaluation on unseen seeds + deliverables | ⏳ Pending |

---

## Issues I'm Facing Right Now

**Current blocker: chunk loss dilution causing initial-reach underprediction.**

ACT predicts a chunk of ~100 future actions at once and trains with L1 loss averaged uniformly across the whole chunk. Since most of any chunk is low-magnitude "settling" motion after the initial big reach movement, the model can achieve deceptively low *aggregate* loss while badly underpredicting the one action that actually matters most for task success.

This has been confirmed with direct measurement, not just theory: at convergence (epoch 300), the average L1 loss across the whole chunk is 0.046, but the loss specifically at chunk position 0 (the very next action) is 0.160 — roughly **3.5x higher**, and this gap does not close with more training; it plateaus independently of the overall average. In closed-loop rollout this shows up concretely: the expert commands a sharp joint-0 swing of -1.0 and a wide gripper opening of 0.40 at Step 0, while the policy predicts near-zero for both (joint 0: -0.003, gripper: 0.011).

**Already fixed along the way (worth noting since these were real, non-obvious bugs):**
- Frozen visual backbone (`lr_backbone = 0`) silently prevented any learning from camera pixels — fixed by unfreezing (`lr_backbone = 1e-5`).
- `kl_weight = 0` caused the CVAE's latent space to go unregularized, so at rollout the policy received an out-of-distribution latent vector and collapsed to predicting a constant mean action — fixed with `kl_weight = 10`.
- The expert controller's placement logic used a fixed table height instead of the target's real 3D position, causing large placement errors when targets were elevated — fixed the descent logic to use the target's true (x, y, z); elevated targets are temporarily disabled (table-level only) as a debugging simplification, to be reintroduced once ground-level training is validated.

**Currently deciding on the fix** for the loss dilution issue — most likely a position-weighted L1 loss that upweights early chunk positions relative to later ones — before proceeding further.

---

## What's Still Pending

- [ ] Resolve chunk loss dilution (position-weighted loss) and achieve a genuine successful rollout on a single overfit episode
- [ ] Re-introduce elevated (non-table-level) target placement
- [ ] Stage 2: overfit to 5 episodes, validate generalization across varied cube/target positions
- [ ] Stage 3: full training on all 55 episodes
- [ ] Evaluation: success rate over unseen seeds, rollout videos
- [ ] Trained checkpoint, loss curve plots, and final project report

---

## How to Clone and Run

```bash
git clone <your-repo-url>
cd ACT
```

**Requirements:** Python 3.10, PyTorch (GPU), Torchvision, Gymnasium, `panda-gym`, `h5py`, `einops`

```bash
python -m venv venv_act
source venv_act/bin/activate
pip install -r requirements.txt
```

**Stage 1 — Overfit validation (single episode):**
```bash
./venv_act/bin/python scratch/overfit_one_episode.py --task panda_pick_and_place
```

**Full training (55 episodes) — not yet recommended until Stage 1 passes:**
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
* **Note on success criterion:** the environment's default check is proximity-only (≤5cm), which doesn't require the gripper to actually release the cube. The expert controller's own state machine additionally requires the gripper to open and the arm to retract before an episode is logged as complete.
* **Current scope:** target positions are currently restricted to table-level (z=0) as a debugging simplification; elevated/mid-air targets are a planned next step.
* **Expert data:** 55 demonstrations from a custom IK-based state-machine controller (approach → grasp → lift → translate → descend to target → release → retract)
* **Observations:** 14-dim state (7 joint positions, 1 gripper width, 3 cube coords, 3 target coords) + 640×480 RGB front camera frames
* **Actions:** 14-dim target joint values + gripper command

## Model Configuration

* **Visual backbone:** ResNet-18, fine-tuned (`lr_backbone = 1e-5`)
* **Policy:** ACT (CVAE-based Action Chunking Transformer), `kl_weight = 10`
* **Chunk size:** 100 steps (`num_queries`)
* **Inference:** temporal aggregation with exponential-decay blending across overlapping chunk predictions
* **Optimizer:** AdamW

---

## Repository Structure

```
.
shaka_act/
├── config/
│   └── config.py            # Hyperparameters and environment configurations (LR, state dimension, camera names, nheads)
├── checkpoints/
│   └── .gitkeep             # Trained checkpoints (.ckpt) saved during train.py execution
├── data/
│   └── .gitkeep             # PyBullet expert datasets
├── training/
│   ├── policy.py            # ACTPolicy and CNNMLPPolicy wrapper classes
│   └── utils.py             # Dataloader utilities (EpisodicDataset) and dataset normalizer (get_norm_stats)
│
│  # --- Simulator Scripts (PyBullet) ---
├── record_episodes_sim.py   # State-machine IK expert script used to generate and log 55 demo episodes
├── train.py                 # Main training script on the generated HDF5 dataset
├── evaluate_sim.py          # Evaluation rollout script with closed-loop temporal aggregation
│
│  # --- Physical Robot Scripts (Dynamixel hardware setup) ---
├── record_episodes.py       # Data logger for human demonstrations on physical arms
├── evaluate.py              # Rollout execution on physical Follower arm
├── teleoperation.py         # Leader-to-Follower direct torque controller mapping
├── robot.py                 # Low-level serial communication for hardware joint encoders
├── dynamixel.py              # Command packet mapping for Dynamixel servo protocols
│
│  # --- Testing & Documentation ---
├── cam.ipynb                # Notebook for checking camera feeds
├── replay_episode.ipynb     # Notebook to play back and verify logged demonstration files
```

---

## Supplementary Work: KUKA KR16 Gazebo Simulation

A separate, completed sub-project simulating a **KUKA KR16-2** industrial arm in ROS 2 / Gazebo:

* **Kinematics calibration:** realigned skewed joint frames from raw CAD coordinates to clean bases parallel to the robot's base frame
* **Inertial realignment:** rotated link inertia matrices (`I_new = R_vis · I_old · R_vis^T`) to satisfy physics constraints and eliminate simulation instability
* **URDF implementation:** standardized package structure under `kuka_description`, with tuned damping/friction thresholds
