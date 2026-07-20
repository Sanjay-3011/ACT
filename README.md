# Imitation Learning on Franka Panda Pick-and-Place using Action Chunking Transformer (ACT)

An end-to-end implementation of **ACT (Action Chunking Transformer)** for closed-loop, vision-based robotic manipulation — trained on self-collected expert demonstrations of a Franka Panda arm performing pick-and-place in PyBullet. The project follows a staged validation methodology (1 → 5 → 55 episodes) to isolate and debug failure modes before committing to full-scale training, rather than training blind on the full dataset and hoping it works.

---

## Project Status: In Progress — Stage 1 (Single-Episode Validation Complete)

**Current milestone:** Single-episode overfitting on Episode 1 successfully validated with a 100% success rate (**4.62 cm placement error** at Step 67, below the 5.0 cm success threshold). Resolved initial-reach underprediction by implementing a fixed step-function chunk-position weighting scheme (`weight = 5.0` for `t < 5`, `weight = 0.2` for `t >= 5`). Ready to scale to multi-episode training.

| Stage | Status |
|---|---|
| 1. Environment setup, data collection (55 expert episodes) | ✅ Complete |
| 2. Single-episode overfitting (pipeline validation) | ✅ Complete (Success at Step 67, 4.62 cm error) |
| 3. 5-episode overfitting (generalization check) | 🔄 In progress |
| 4. Full 55-episode training | ⏳ Pending |
| 5. Evaluation on unseen seeds + deliverables | ⏳ Pending |

---

## Why Staged Validation

Training an 83.92M-parameter vision transformer directly on the full dataset initially produced a **0% success rate**, with no clear signal for *why*. Two silent bugs were compounding: a frozen visual backbone (`lr_backbone = 0.0`) that prevented the model from ever learning to localize the cube from pixels, and a collapsed CVAE latent space that caused constant zero-action outputs. Debugging both simultaneously on the full dataset would have been slow and ambiguous.

Instead, the project deliberately overfits to 1 episode first: if the model can't perfectly reproduce a single trajectory it was trained on, there's a pipeline bug, not a generalization problem — and it's far cheaper to diagnose on one episode than fifty-five.

---

## Setup

**Requirements:** Python 3.10, PyTorch (GPU), Torchvision, Gymnasium, `panda-gym`, `h5py`, `einops`

```bash
python -m venv venv_act
source venv_act/bin/activate
pip install -r requirements.txt
```

---

## How to Run

**1. Play expert trajectory simulation (dataset demonstration):**
```bash
./venv_act/bin/python scratch/play_expert_gui.py
```

**2. Train single-episode overfitting policy (Episode 1):**
```bash
./venv_act/bin/python scratch/overfit_one_episode.py --task panda_pick_and_place
```

---

## Next Steps

- **Stage 3 (5-Episode Overfitting)**: Scale training to a 5-episode subset to evaluate multi-trajectory policy learning and intermediate generalization.
- **Stage 4 (Full 55-Episode Training)**: Scale to the entire 55-episode dataset for full policy training.
- **Stage 5 (Evaluation & Deliverables)**: Evaluate success rate over unseen random seeds and record rollout videos.