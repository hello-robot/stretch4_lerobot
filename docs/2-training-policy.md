# 2. Training a Policy

This file goes over the second step: training a policy on the dataset you recorded in
[1-collecting-and-preparing-data.md](1-collecting-and-preparing-data.md). Training uses stock
`lerobot-train`, so any LeRobot policy works. We use diffusion, ACT, and pi0.5.

**Where things run:** everything in this file runs on the **desktop / PC with an NVIDIA GPU**.

**Before you start:** you have a recorded dataset
([file 1](1-collecting-and-preparing-data.md)), and you have chosen the policy you will train. The
policy decides what you install (Step 1) and how the dataset is prepared:

| policy | `--policy.type` | extra to install | prepare the dataset with |
|---|---|---|---|
| Diffusion | `diffusion` | `diffusion` | absolute actions |
| pi0.5 | `pi05` | `pi` | relative actions (`--relative`) |
| ACT | `act` | none | absolute actions |
| another policy | see LeRobot's docs | see LeRobot's install docs | absolute actions, unless you train with `--policy.use_relative_actions=true` |

Preparing the dataset ([Step 4 of file 1](1-collecting-and-preparing-data.md#step-4-prepare-the-dataset-recommended))
is recommended, not required. Only diffusion, pi0.5 and ACT have been tested on Stretch 4.

## Contents

- [Step 1: Set up the training machine](#step-1-set-up-the-training-machine)
- [Step 2: Get the dataset](#step-2-get-the-dataset)
  - [2.1 From the Hub (recommended)](#21-from-the-hub-recommended)
  - [2.2 From the robot, without the Hub](#22-from-the-robot-without-the-hub)
  - [2.3 Prepare the dataset on the desktop (optional)](#23-prepare-the-dataset-on-the-desktop-optional)
- [Step 3: Train](#step-3-train)
  - [3.1 Run the training](#31-run-the-training)
  - [3.2 Checkpoints](#32-checkpoints)

**Previous:** [1-collecting-and-preparing-data.md](1-collecting-and-preparing-data.md) ·
**Next:** [3-deploy-policy.md](3-deploy-policy.md)

---

## Step 1: Set up the training machine

You need Python 3.12.

```bash
mkdir -p ~/repos && cd ~/repos
git clone https://github.com/hello-robot/stretch4_lerobot.git
cd stretch4_lerobot

python3.12 -m venv .venv
source .venv/bin/activate
pip install "lerobot[dataset,training,async,peft]==0.6.*"
pip install --no-deps -e packages/lerobot_camera_oak -e packages/lerobot_robot_stretch4
```

Then install your policy's extra (from the table at the top; ACT needs none):

```bash
pip install "lerobot[<extra>]==0.6.*"      # e.g. lerobot[diffusion] or lerobot[pi]
```

To train another policy later, install its extra in the same venv.

Check that the GPU works:

```bash
nvidia-smi                                                   # lists your GPU; if not, install the NVIDIA driver
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

**In every new terminal**, activate the venv before the commands in the next steps (otherwise:
`command not found`):

```bash
source ~/repos/stretch4_lerobot/.venv/bin/activate
```

---

## Step 2: Get the dataset

Do **2.1** (if you pushed the dataset to the Hub) **or 2.2** (if you did not). Then do **2.3** only
if the dataset is not yet prepared for the policy you will train.

### 2.1 From the Hub (recommended)

There is nothing to copy: `lerobot-train` downloads the dataset when you train. In Step 3, use the
name you pushed it as for `<user>/<folder_name>`.

Datasets are pushed private by default, so log in on the desktop once:

```bash
hf auth login
```

Paste a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Next:** 2.3 if the dataset is not prepared, otherwise [Step 3](#step-3-train).

### 2.2 From the robot, without the Hub

Copy the dataset from the robot with `rsync`:

```bash
rsync -aP --mkpath \
    hello-robot@<robot_ip>:.cache/huggingface/lerobot/<user>/<folder_name>_YYYYMMDD_HHMMSS/ \
    ~/.cache/huggingface/lerobot/<user>/<folder_name>/
```

**CHANGE THIS:**

- `<robot_ip>`: the robot's IP address. `rsync` asks for the robot's password.
- `<user>/<folder_name>_YYYYMMDD_HHMMSS`: the dataset's folder on the robot (`ls
  ~/.cache/huggingface/lerobot/<user>` on the robot).
- `<user>/<folder_name>`: the name you want on the desktop. You use it as `--dataset.repo_id` in
  Step 3.

The `/` at the end of the source copies the folder's contents, so the date and time are not in the
desktop's folder name.

**Next:** 2.3 if the dataset is not prepared, otherwise [Step 3](#step-3-train).

### 2.3 Prepare the dataset on the desktop (optional)

Training works without this, but a prepared dataset is recommended. Do it here if you did not
prepare the dataset on the robot, or to switch it between absolute and relative actions. A dataset
prepared on the robot stays prepared through the Hub and `rsync`. It is safe to repeat.

- **Copied from the robot (2.2):**

  **Absolute actions**
  ```bash
  stretch4_fix_stats <user>/<folder_name>
  ```

  **Relative actions** (`--policy.use_relative_actions=true`)
  ```bash
  stretch4_fix_stats <user>/<folder_name> --relative
  ```

  If the copy is not in `~/.cache/huggingface/lerobot`, give its folder instead of the name
  (`stretch4_fix_stats /path/to/dataset`), and add `--dataset.root=/path/to/dataset` to
  `lerobot-train` in Step 3.

- **From the Hub:** if you pushed the dataset without preparing it, you can prepare it on the desktop. LeRobot
  keeps a dataset downloaded from the Hub in its own cache, under
  `~/.cache/huggingface/lerobot/hub/datasets--<user>--<folder_name>/snapshots/<version>/`, so give
  `stretch4_fix_stats` that folder instead of the name. The `*` in the command fills in `<version>`.
  Download the dataset first (without the videos, which it does not need):

  ```bash
  python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('<user>/<folder_name>', download_videos=False)"
  stretch4_fix_stats ~/.cache/huggingface/lerobot/hub/datasets--<user>--<folder_name>/snapshots/*/
  ```

  Add `--relative` for relative actions. This changes only the copy on this desktop, and
  `lerobot-train` uses it. Simpler: prepare it again on the robot and push it again.

`--chunk-size` and `--exclude-joints` are explained in
[Step 4 of file 1](1-collecting-and-preparing-data.md#step-4-prepare-the-dataset-recommended).
`stretch4_fix_stats` changes the dataset in place: do not train two policies from the same dataset
at the same time.

**Next:** [Step 3](#step-3-train).

---

## Step 3: Train

### 3.1 Run the training

The command is `lerobot-train` for every policy; what changes is the `--policy.*` flags. Values to
change are listed under each command.

Every command saves to `~/training/<run_name>`. Use **a new `<run_name>` for each training**:
LeRobot refuses to start in a folder that already exists. A name like `<task>_<policy>_<date>` tells
you later what each run was; each checkpoint's `pretrained_model/train_config.json` records the
dataset and flags it was trained with.

#### Diffusion

```bash
lerobot-train --dataset.repo_id=<user>/<folder_name> \
    --policy.type=diffusion --policy.push_to_hub=false \
    --policy.resize_shape="[240,384]" --policy.crop_ratio=0.9 \
    --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 --policy.n_action_steps=48 \
    --batch_size=32 --steps=100000 --output_dir=$HOME/training/<run_name>
```

**CHANGE THIS:**

- `<user>/<folder_name>`, `<run_name>`: your dataset (Step 2) and a new name for this run.
- `--policy.resize_shape`: the size you recorded at, `[height,width]`. Default `[240,384]`; change it
  if you used `--size` in `stretch4_pick_cameras`.
- `--policy.crop_ratio`: how much of each image the random training crop keeps (`0.9` = 90%, `1.0`
  = no crop). LeRobot works out the crop size from `resize_shape`.

**Keep as is:**

- `--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 --policy.n_action_steps=48`:
  make the policy fast enough to run on the Jetson. They are saved in
  the checkpoint's `config.json`, where you can change them later without retraining.
- `--policy.push_to_hub=false`: without it, training stops if no Hub repo is set. To push the model,
  drop it and add `--policy.repo_id=<user>/<model>`.

#### pi0.5

pi0.5 is trained with LoRA (a full fine-tune of the whole model needs about 70 GB of GPU memory): the pretrained model (`lerobot/pi05_base`) stays frozen and small
adapters are trained on top of it. This command puts adapters on every layer, the vision encoder
included, so the policy learns to read your cameras.

**Before you train:** pi0.5 uses the text tokenizer of
[google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224), which needs access
approval. Request access on that page, wait until it is granted, and run `hf auth login` on the
desktop. Running the policy (file 3) needs the same.

The command:

```bash
lerobot-train \
    --dataset.repo_id=<user>/<folder_name> \
    --dataset.image_transforms.enable=true \
    --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["grip","base_v","base_omega"]' \
    --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.push_to_hub=false \
    --policy.optimizer_lr=1e-4 --policy.scheduler_decay_lr=1e-5 \
    --peft.method_type=LORA --peft.target_modules=all-linear --peft.r=32 --peft.lora_alpha=64 \
    --batch_size=16 --steps=30000 --save_freq=5000 \
    --output_dir=$HOME/training/<run_name> \
    --wandb.enable=true --wandb.mode=offline
```

**CHANGE THIS:**

- `<user>/<folder_name>`: your dataset (Step 2).
- `<run_name>`: a new name for this run.

**Recommended:** run it with the values shown, with a dataset prepared with
`stretch4_fix_stats --relative`
([Step 4 of file 1](1-collecting-and-preparing-data.md#step-4-prepare-the-dataset-recommended)).

**Adjust to your dataset and GPU:**

- `--steps` (30000): how long to train. A bigger dataset needs more steps. The training log shows
  `epch`: how many times training has gone through the whole dataset.
- `--batch_size` (16): lower it if you run out of GPU memory (16 → 8 → 4). A smaller batch goes
  through fewer frames per step, so raise `--steps` to reach the same `epch`.
- `--save_freq` (5000): how often to save a checkpoint.

**Our recommended choices** (change them only if you know why):

- `--policy.use_relative_actions=true` with
  `--policy.relative_exclude_joints='["grip","base_v","base_omega"]'`: the arm joints are learned
  relative to where they are, the gripper and the base stay absolute.
- `--peft.target_modules=all-linear`: LoRA trains every layer, so the policy learns to read your
  cameras. LeRobot's default trains only the action head.
- `--peft.r=32 --peft.lora_alpha=64`: the adapter's size and strength. Keep `lora_alpha` at 1-2×
  `r`.
- `--policy.optimizer_lr=1e-4`: a learning rate for LoRA. pi0.5's default (2.5e-5) is for a full
  fine-tune.

**Must match the dataset.** These must agree with how you ran `stretch4_fix_stats`. If you change
one, prepare the dataset again with the same value.

- `--policy.use_relative_actions=true` ↔ `--relative`.
- `--policy.relative_exclude_joints` ↔ `--exclude-joints` (default `grip base_v base_omega`). Do not
  leave it out: LeRobot's default (`gripper`) matches none of our columns.
- `--policy.chunk_size` (50, not in the command) ↔ `--chunk-size` (default 50).

`--wandb.mode=offline` saves the loss curve on disk; without it the loss is only printed.

Run the checkpoint with `stretch4_policy_server` (file 3): LeRobot's stock server runs a LoRA
checkpoint as a random policy.

#### Other policies (ACT, ...)

Start from this and add the policy's own flags from LeRobot's docs:

```bash
lerobot-train --dataset.repo_id=<user>/<folder_name> \
    --policy.type=act --policy.push_to_hub=false \
    --output_dir=$HOME/training/<run_name>
```

- **CHANGE** `--policy.type`: the policy, e.g. `act`, `vqbet`.

### 3.2 Checkpoints

Checkpoints are written to `~/training/<run_name>/checkpoints/<step>/pretrained_model` every
`--save_freq` steps, and `last`
points to the newest. Only `pretrained_model/` is needed to run the
policy.

Start with `last`. If the policy does not work well on the robot, it may have overfit: try an
earlier checkpoint.

---

**Next:** [3-deploy-policy.md](3-deploy-policy.md)
