# Multi-Robot Simulation and Learning Repository

This repository hosts two primary robotic simulation and learning projects:
1. **Visual Imitation Learning on Franka Panda Pick-and-Place**: Implementing, training, and validating an Action Chunking Transformer (ACT) model in a PyBullet simulator (`panda-gym`).
2. **URDF Modeling & ROS/Gazebo Simulation of the Kuka KR16 Arm**: Calibrating mechanical CAD coordinates, rotating inertia matrices, and simulating a Kuka KR16 manipulator in ROS 2 and Gazebo.

---

## 1. Visual Imitation Learning (Franka Panda & ACT)

### Project Status
I am currently executing **Step 4 (Prepare Data and Train ACT)**. To ensure maximum pipeline stability and verify coordinate matching/visual features, I am using a staged training approach:
1. **1-Episode Overfitting (Current)**: Overfitting the ACT policy on exactly 1 expert demonstration (Episode 0) to verify visual backbone integration, action de-normalization, and simulator control outputs.
2. **5-Episode Overfitting**: Validating visual generalization on a small subset with randomized block/target positions.
3. **55-Episode Full Training**: Training the final policy on the complete expert dataset.

### Detailed Step-by-Step Implementation

#### Step 1: Environment Setup
I set up a dedicated virtual environment with all required dependencies:
* **Python Version**: `3.10`
* **Core Libraries**: PyTorch (with GPU acceleration), Torchvision, Gymnasium, `h5py` (dataset handling), `einops` (tensor manipulations).
* **Simulator Backend**: `panda-gym` (utilizes PyBullet to simulate the Franka Emika Panda robot arm).
* **Environment Verification**: Verified rendering of the robot arm, table surface, cube object, and destination target marker.

#### Step 2: Pick-and-Place Task Definition
The task is defined in the `PandaPickAndPlace-v3` environment:
* **Task Goal**: Relocate a cube on the table to a specified target destination.
* **Randomization**:
  * The cube's initial position is randomized on the table surface at each reset.
  * The destination target position is randomly generated in the air/on the table.
* **Success Condition**: The episode is successful if the distance between the cube center and the target location is **less than 5 cm** (`distance <= 0.05m`).
* **Auto-Reset**: The environment automatically resets to generate a new randomized configuration upon episode termination/timeout (max 120 steps).

#### Step 3: Demonstration Collection
To train the ACT policy, I generated expert demonstration trajectories:
* **Expert Controller**: Built an Inverse Kinematics (IK)-based state-machine expert controller that plans joint trajectories to approach, grasp, lift, translate, and release the cube.
* **Dataset Size**: Collected **55 successful episodes**.
* **Recording Specifications**:
  * **Observations (`qpos`)**: 14-dimensional state containing 7 arm joint positions, 1 gripper finger width, 3 cube Cartesian coordinates, and 3 target Cartesian coordinates.
  * **Observations (`images`)**: `640x480` RGB camera frames from a front-facing viewpoint.
  * **Actions**: 14-dimensional target joint values and gripper command targets.
  * **Format**: Logged as structured HDF5 dataset files (`episode_0.hdf5` to `episode_54.hdf5`) in `data/panda_pick_and_place/`.

#### Step 4: Data Preparation and ACT Training
I preprocessed the collected demonstrations and initialized training:
* **Data Normalization**: Normalization statistics (mean and standard deviation) are computed across the active timesteps of all training trajectories to scale observations and actions into a standard Gaussian distribution.
* **Training Speed Optimization**: Created a custom `PreloadedEpisodicDataset` that loads dataset frames into RAM during startup, bypassing disk-read bottlenecks and speeding up GPU utilization to **99%**.
* **ACT Hyperparameters**:
  * **Image Encoder**: ResNet-18 visual backbone.
  * **Optimizer**: AdamW (`lr = 1e-5`, `lr_backbone = 2e-5`).
  * **Chunk Size (`num_queries`)**: `100` steps.
  * **Policy Mode**: Deterministic Decoder Policy (`kl_weight = 0` / CVAE `kl_weight = 10` for validation).
  * **Temporal Aggregation**: Enabled to query the policy at every step and blend overlapping action predictions using exponential decay weights.

#### Step 5: Policy Evaluation
Once training is complete, the policy is evaluated on unseen seeds:
* **Metrics Tracked**: Success Rate (%), Average Episode Length (steps), Standard Deviation of Episode Length.
* **Rollout Architecture**: Closed-loop control. The policy receives the current front camera image and joint states at every step and executes the blended action chunk in the PyBullet simulator.
* **Video Recording**: Successful runs are automatically saved as MP4 files for visual verification.

#### Step 6: Deliverables Checklist
Upon completion, the following deliverables will be prepared:
1. **GitHub Repository**: Organized source code.
2. **Demonstration Dataset**: HDF5 files containing the 55 successful expert episodes.
3. **Trained Checkpoint**: The optimal ACT checkpoint `.ckpt` file.
4. **Training Plots**: Loss curves showing L1 and total losses.
5. **Evaluation Videos**: Demonstration MP4 rollouts showing successful pick-and-place behaviors.
6. **Project Report**: A 2-page document detailing environment setup, hyperparameters, success metrics, and failure modes.

### How to Run (Franka Panda ACT)

1. **Training (Overfit Validation)**:
   ```bash
   ./venv_act/bin/python scratch/overfit_one_episode.py --task panda_pick_and_place
   ```
2. **Main Training (Full Dataset)**:
   ```bash
   ./venv_act/bin/python shaka_act/train.py --task panda_pick_and_place
   ```
3. **Evaluation**:
   ```bash
   ./venv_act/bin/python shaka_act/evaluate_sim.py --task panda_pick_and_place
   ```

---

## 2. Kuka KR16 Mechanical Modeling & Gazebo Simulation

This project involved setting up a standard ROS 2 / Gazebo simulation environment for a **Kuka KR16-2 industrial robot manipulator**. The primary challenge resolved was aligning coordinates from a raw, non-standard mechanical CAD export into a simulation-ready URDF format.

### Key Accomplishments & Implementation Details

1. **Homogeneous Kinematic Coordinate Transformation**:
   * Raw CAD exports typically contain skewed RPY orientation frames for joints.
   * I recalculated the relative homogeneous transformations to align all joint axes parallel to standard bases (setting joint roll, pitch, and yaw to `0.0` where possible).
   * Updated `visual` and `collision` link offsets (`xyz` and `rpy`) dynamically to counteract the joint frame adjustments, preventing the meshes from visually breaking.

2. **Inertia Tensor Rotation & Calibrations**:
   * Rotated the 3x3 inertia tensor matrices of each link by the orientation rotation matrix $R_{vis}$ between old and new joint frames:
     $$I_{new} = R_{vis} \cdot I_{old} \cdot R_{vis}^T$$
   * Zeroed out skewed off-diagonal elements in the inertia matrix to satisfy standard physics triangle inequalities, preventing the Kuka arm from vibrating or shaking inside the Gazebo physics engine.

3. **URDF Calibrations**:
   * Standardized packages from `package://kuka arm URDF` to `package://kuka_description` for standard ROS package management.
   * Added damping and friction values (`0.1`) on joint dynamic tags to stabilize Gazebo joint velocity commands.
   * Fixed joint limits (e.g. correcting `joint3`'s lower limit from positive to negative values).

### Mechanical Packages Structure
* **`kuka_description`**: Contains the URDF models, meshes (CAD DAE/STL models), and coordinate transformations.
* **`kuka_gazebo`**: Launches the Gazebo world and spawns the robot manipulator.
* **`kuka_bringup`**: Manages ROS 2 control configurations and script interfaces.