# 3. Deploying a Policy

This file goes over the last step: running the policy you trained in
[2-training-policy.md](2-training-policy.md) on Stretch 4.

The robot streams its cameras and joints to a **policy server**, which runs the policy on a GPU and
sends the actions back. First choose where the server runs:

| policy | **Jetson** (inside the robot) | **Desktop GPU** |
|---|---|---|
| Diffusion | yes | yes |
| pi0.5 | no: the Jetson Orin NX is not powerful enough | yes |
| ACT | yes | yes |
| setup | install once, copy the checkpoint to it | none if it is the desktop you trained on; otherwise install and copy the checkpoint |

Only Step 1 depends on this choice; the other steps are the same for both.

To log in from the desktop (`<robot_ip>` is the robot's IP address; `exit` logs you out):

- robot: `ssh hello-robot@<robot_ip>`
- Jetson, which is reachable only through the robot: `ssh -J hello-robot@<robot_ip> jetson1@192.168.1.101`

*Optional:* to stop typing passwords, run once `ssh-copy-id hello-robot@<robot_ip>`, and for the
Jetson `ssh-copy-id -o ProxyJump=hello-robot@<robot_ip> jetson1@192.168.1.101`.

**Before you start:** a trained checkpoint (`~/training/<run_name>/checkpoints/last/pretrained_model`),
and the dataset it was trained on, on the robot (Step 2).

## Contents

- [Step 1: Set up the policy server](#step-1-set-up-the-policy-server)
  - [Option 1: Jetson](#option-1-jetson)
  - [Option 2: Desktop GPU](#option-2-desktop-gpu)
- [Step 2: The dataset on the robot](#step-2-the-dataset-on-the-robot)
- [Step 3: Get the robot ready](#step-3-get-the-robot-ready)
- [Step 4: Run the policy](#step-4-run-the-policy)
  - [4.1 Start the server](#41-start-the-server)
  - [4.2 Start the robot](#42-start-the-robot)
  - [4.3 Running several episodes](#43-running-several-episodes)
  - [4.4 What you should see](#44-what-you-should-see)
- [Safety and limits](#safety-and-limits)
- [Troubleshooting](#troubleshooting)

**Previous:** [2-training-policy.md](2-training-policy.md)

---

## Step 1: Set up the policy server

### Option 1: Jetson

On the Jetson, once: clone this repository and run the setup script. It needs internet access and
takes a while.

```bash
sudo apt update && sudo apt install python3.10-venv
mkdir -p ~/repos && cd ~/repos
git clone https://github.com/hello-robot/stretch4_lerobot.git
cd stretch4_lerobot
tools/setup_jetson.sh          # creates ~/repos/stretch4_lerobot/.venv
```

It ends by printing whether CUDA works (`cuda True`).

Then, on the desktop, copy the checkpoint to the Jetson:

```bash
rsync -aP --mkpath -e "ssh -J hello-robot@<robot_ip>" \
    ~/training/<run_name>/checkpoints/last/pretrained_model/ \
    jetson1@192.168.1.101:~/policies/<name>/pretrained_model/
```

**CHANGE THIS:**

- `<robot_ip>`: the robot's IP address.
- `<run_name>`: your training run (file 2).
- `<name>`: any name for the policy on the Jetson.


### Option 2: Desktop GPU

- **The desktop you trained on:** nothing to do. The server uses the training venv
  (`~/repos/stretch4_lerobot/.venv`) and the checkpoint where training saved it.
- **Another desktop:** install as in [Step 1 of file 2](2-training-policy.md#step-1-set-up-the-training-machine),
  then copy the checkpoint to it from the training desktop:

  ```bash
  rsync -aP --mkpath ~/training/<run_name>/checkpoints/last/pretrained_model/ \
      <user>@<desktop_ip>:~/policies/<name>/pretrained_model/
  ```

  `<user>@<desktop_ip>` is your login on that desktop; `<run_name>` and `<name>` as in Option 1.
  For pi0.5, also run `hf auth login` there, with an account that has access to the tokenizer
  ([file 2, pi0.5](2-training-policy.md#pi05)).

---

## Step 2: The dataset on the robot

The robot reads the cameras, joint ranges and held joints from the dataset's `meta/` folder.

- **You prepared the dataset on the robot, or did not prepare it:** nothing to copy. The dataset is
  at `~/.cache/huggingface/lerobot/<user>/<folder_name>_YYYYMMDD_HHMMSS` on the robot.
- **You prepared or changed it on the desktop**
  ([Step 2.3 of file 2](2-training-policy.md#23-prepare-the-dataset-on-the-desktop-optional)): on
  the desktop, copy its `meta/` to the robot. The dataset is then at `~/datasets/<folder_name>` on
  the robot.

  ```bash
  rsync -a --mkpath ~/.cache/huggingface/lerobot/<user>/<folder_name>/meta \
      hello-robot@<robot_ip>:~/datasets/<folder_name>/
  ```

---

## Step 3: Get the robot ready

On the robot, make sure:

- `stretch_body_server` is running and the robot is homed;
- no other program holds the gripper camera or commands the arm;
- the arm is at the start pose of the demos: `stretch4_pose goto <pose_name>.json`
  ([Step 3.2 of file 1](1-collecting-and-preparing-data.md#32-save-and-reset-to-a-start-pose-optional)).
  Not needed with `stretch4_rollout` (4.3): it moves the arm there itself.

---

## Step 4: Run the policy

Start the **server first**, then the **robot**, each in its own terminal. The robot moves: keep a
hand on the run-stop. Ctrl-C stops it.

### 4.1 Start the server

On the Jetson or the desktop (Step 1):

```bash
source ~/repos/stretch4_lerobot/.venv/bin/activate
stretch4_policy_server --host=0.0.0.0 --port=8080
```

It prints `PolicyServer started on 0.0.0.0:8080`.

Use `stretch4_policy_server`, not LeRobot's stock server: same flags and protocol, with fixes for
bugs in LeRobot 0.6's async inference (see [the README](../README.md#why-stretch4_policy_server-instead-of-stock-lerobot)).

### 4.2 Start the robot

On the robot:

> **Warning:** the policy drives the base as well as the arm. To keep the base still, for example on
> a first run, add `--robot.use_base=false`.

```bash
source ~/repos/stretch4_lerobot/.venv/bin/activate

python -m lerobot.async_inference.robot_client \
    --robot.type=stretch4 --robot.mode=control \
    --robot.preprocessing_from=<dataset_folder> --task="<Task Name>" --fps=30 \
    --server_address=<server_ip>:8080 --pretrained_name_or_path=<checkpoint> \
    --policy_type=<policy_type> --actions_per_chunk=<actions_per_chunk> \
    --policy_device=cuda --chunk_size_threshold=0.7
```

**CHANGE THIS:**

- **Your dataset:**
  - `<dataset_folder>`: the dataset's folder on the robot (Step 2).
  - `<Task Name>`: **the exact task text the dataset was recorded with** (`--dataset.single_task`).
  - `--fps`: the rate the dataset was recorded at (`--dataset.fps`, 30 in file 1).
- **Where the server runs** (Step 1):
  - Jetson: `<server_ip>` is `192.168.1.101`, `<checkpoint>` is
    `/home/jetson1/policies/<name>/pretrained_model`.
  - Desktop: `<server_ip>` is the desktop's IP address, `<checkpoint>` is the checkpoint's full path
    on the desktop, e.g. `/home/<you>/training/<run_name>/checkpoints/last/pretrained_model`, or
    `/home/<you>/policies/<name>/pretrained_model` if you copied it (Step 1). Not `~`: the command
    runs on the robot.
- **The policy:**
  - Diffusion: `<policy_type>` is `diffusion`, `<actions_per_chunk>` is `48` (`n_action_steps` from
    file 2).
  - pi0.5: `<policy_type>` is `pi05`, `<actions_per_chunk>` is `50` (its `chunk_size`).
  - ACT and others: `<policy_type>` is the `--policy.type` you trained with (e.g. `act`),
    `<actions_per_chunk>` is the checkpoint's `n_action_steps` from its `config.json` (100 for ACT).

Once this works and you plan to run the policy many times, use `stretch4_rollout` instead
([4.3](#43-running-several-episodes)).

**Diffusion settings you can change without retraining.** They are in the checkpoint's
`pretrained_model/config.json` on the server. Edit them there and restart the server:

```json
"noise_scheduler_type": "DDIM",
"num_inference_steps": 10,
"n_action_steps": 48,
```

- `noise_scheduler_type`, `num_inference_steps`: how each chunk of actions is computed. DDIM with 10
  steps is fast enough for the Jetson; LeRobot's default (DDPM, 100 steps) is about 10× slower.
- `n_action_steps`: how many actions the robot runs from each chunk, at most
  `horizon - n_obs_steps + 1` (63 by default). If you change it, pass the same number as
  `--actions_per_chunk` on the robot.

### 4.3 Running several episodes

Recommended once the policy runs (4.2) and you want to run it many times, e.g. to benchmark it. With
`robot_client`, every run starts the cameras and loads the policy on the server again, which takes
a while each time. `stretch4_rollout` does both once, then runs as many episodes as you ask for and
returns the arm to the same start pose before each one.

It is the command from 4.2 with `stretch4_rollout` in place of
`python -m lerobot.async_inference.robot_client`, plus a start pose and a number of episodes:

```bash
stretch4_rollout \
    --robot.type=stretch4 --robot.mode=control \
    --robot.preprocessing_from=<dataset_folder> --task="<Task Name>" --fps=30 \
    --server_address=<server_ip>:8080 --pretrained_name_or_path=<checkpoint> \
    --policy_type=<policy_type> --actions_per_chunk=<actions_per_chunk> \
    --policy_device=cuda --chunk_size_threshold=0.7 \
    --start_pose=<pose_name>.json --num_episodes=10
```

The values are the same as in 4.2. Its own flags:

| flag | default | what it sets |
|---|---|---|
| `--start_pose` | required | the pose file to return to, saved when you recorded ([Step 3.2 of file 1](1-collecting-and-preparing-data.md#32-save-and-reset-to-a-start-pose-optional)) |
| `--num_episodes` | 10 | how many episodes to run |
| `--episode_time_s` | 60 | the longest an episode runs, in seconds |
| `--reset_time_s` | 20 | the wait at the start pose before each episode, to reset the scene, in seconds |
| `--confirm_goto` | `true` | wait for **Enter** before each move to the start pose |

Each episode goes:

1. It waits for **Enter**, so you can drive the base back first.
2. It moves the arm, wrist and gripper slowly to the start pose. The base is not moved.
3. It waits `--reset_time_s` for you to reset the scene.
4. The policy runs for up to `--episode_time_s`.

**Rollout keys** (in the terminal running `stretch4_rollout`):

- **Enter:** move the arm to the start pose
- **→ (Right Arrow Key):** end the episode now (or end the reset early)
- **← (Left Arrow Key):** discard the episode and run it again
- **Esc key:** stop the session (at the **Enter** prompt: stop without moving the arm)

**Run-stop:** while the robot is run-stopped it does not move, neither to the start pose nor under
the policy, and the terminal says so in red. Release the run-stop and press **Enter** again.

### 4.4 What you should see

- **On the robot, at the start:** `held joints (pinned in the state, kept where they are): ...`
  (the joints `stretch4_fix_stats` held).
- **On the robot, with `stretch4_rollout`:** `Start pose: <file>` at the start, then for each
  episode `Episode 1/10 running (up to 60 s)` and `Episode 1/10 finished after ...`, and a
  `Rollout complete: ...` summary at the end.
- **On the server, diffusion:** `observation history: 2 frames, ...` once per chunk, and mostly
  `Running inference` lines with `must_go: False`. Mostly `True` means the robot runs out of actions
  before each new chunk arrives.
- **On the server, pi0.5:** when the robot connects, `Loading model from: lerobot/pi05_base` and
  `All keys loaded successfully!`.
- Both sides write logs to a `logs/` folder in the directory you ran the command from
  (`policy_server_*.log`, `robot_client_*.log`).

---

## Safety and limits

In control mode, the arm, wrist and gripper get rate-limited targets clamped to the dataset's range.
The base gets the commanded velocity, capped at `--robot.base_max_lin` (default 0.3 m/s) and
`--robot.base_max_ang` (default 0.6 rad/s), and stops if actions stop arriving for 0.3 s.

- **Joint range:** the clamp is the dataset's action range (from `meta/episodes/`), padded by 5% and
  at least a per-joint minimum. `--robot.joint_bounds` sets it by hand.
- **State guard** (`--robot.state_guard=false` turns it off): the state the policy sees stays
  inside what it was trained on. Joints that `stretch4_fix_stats` held are pinned to their training
  value and never commanded; every other value is clamped to the dataset's range, with a warning
  when the robot is well outside it.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| **Training fails: "`observation.images.X` does not match ..."** (diffusion) | The cameras were recorded at different sizes | Record every camera at the same size: `stretch4_pick_cameras --out cameras.json --size 240 384`, then record again. |
| **Robot moves to the wrong place / ignores the object** | Too few demonstrations, or the object is outside the area the demos covered | Collect more demonstrations (50–100+) that cover the workspace.|
| **Hesitates, then jumps (stop-and-go)** | New action chunks arrive after the robot has run out of actions | On the server, mostly `must_go: True` means inference is too slow: use DDIM with 10 steps ([4.2](#42-start-the-robot)), or run the server on a desktop GPU. |
| **Base drives too slowly** | Base velocity is capped | Raise `--robot.base_max_lin` (default 0.3 m/s) or `--robot.base_max_ang` (default 0.6 rad/s). |
| **Robot does not move at all** | Stock LeRobot server, `--robot.mode` not `control`, or run-stopped | Use `stretch4_policy_server` (the stock server logs an empty `Error in StreamActions:`), pass `--robot.mode=control`, and release the run-stop. |
| **Client exits: "Could not connect to the robot server" / "Robot is not homed"** | `stretch_body_server` not running, or robot not homed | Run `stretch_body_server --restart`, then `stretch_robot_home`. |
