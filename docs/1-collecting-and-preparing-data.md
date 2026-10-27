# 1. Collecting and Preparing Data

This file goes over the first step in training and running a policy on Stretch 4: collecting the
data and preparing it for training. Recording does not depend on which policy you train later
(diffusion, ACT, pi0.5...): the same dataset works for all of them.

**Where things run:** everything in this file runs on the **robot** (the Stretch 4's computer).

The three files use these names for the machines:

| machine | what it is |
|---|---|
| **robot** | the Stretch 4's computer (this file) |
| **desktop / PC** | your computer with an NVIDIA GPU, where you train (file 2) |
| **Jetson** | the GPU computer inside the robot, can run a policy (file 3) |

**Before you start:** your Stretch 4 is set up (`stretch4_body` is installed), `stretch_body_server`
is running and the robot is homed, a teleop device works, and you have a
[Hugging Face](https://huggingface.co) account if you want to push the dataset to the Hub.

## Contents

- [Step 1: Install](#step-1-install)
- [Step 2: Pick your cameras](#step-2-pick-your-cameras)
- [Step 3: Record](#step-3-record)
  - [3.1 Tips for good demonstrations](#31-tips-for-good-demonstrations)
  - [3.2 Save and reset to a start pose (optional)](#32-save-and-reset-to-a-start-pose-optional)
  - [3.3 Record](#33-record)
  - [3.4 Resume a stopped session](#34-resume-a-stopped-session)
- [Step 4: Prepare the dataset](#step-4-prepare-the-dataset-recommended)
- [Step 5: Push to the Hugging Face Hub](#step-5-push-to-the-hugging-face-hub-recommended)

**Next:** [2-training-policy.md](2-training-policy.md)

---

## Step 1: Install

LeRobot 0.6 needs Python ≥ 3.12. `--system-site-packages` lets the venv use the robot's
`stretch4_body`, `depthai` and tkinter.

Clone this repository into `~/repos` on the robot and create the venv inside it. Files 2 and 3 assume
`~/repos/stretch4_lerobot/.venv`:

```bash
# Run this on Stretch
cd ~/repos
git clone https://github.com/hello-robot/stretch4_lerobot.git
cd stretch4_lerobot

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install "lerobot[dataset,training,deepdiff-dep,async]==0.6.*"
pip install -e packages/lerobot_camera_oak -e packages/lerobot_robot_stretch4 \
            -e packages/lerobot_teleoperator_stretch4
python -c "import stretch4_body, depthai, lerobot, lerobot_robot_stretch4; print('ok')"
```

**In every new terminal**, run this before the commands in the next steps (they need the venv, and
`cameras.json` is saved in and read from this folder):

```bash
cd ~/repos/stretch4_lerobot && source .venv/bin/activate
```

**Turn off the lights around Stretch 4's eyes** before you collect data, so they do not show up in
the camera images. Open `~/stretch_user/stretch-se4-<robot_number>/stretch_user_params.yaml`. The
`<robot_number>` is at the end of the name in your terminal, `hello-robot@stretch-se4-<robot_number>`.

Set `enabled: 0` under `sentry_eye_animations` and keep the other lines of the file
as they are:

```yaml
sentry_eye_animations:
  enabled: 0
```

Then restart the robot's server so the change takes effect:

```bash
stretch_body_server --restart
```


---

## Step 2: Pick your cameras

Make sure you turned off the eye lights (Step 1), so they do not show up in the camera images.

Choose which cameras to record and how each one is cropped with `stretch4_pick_cameras`. It shows
each camera live with its crop box and what will be recorded. Draw a crop box on each camera with
the mouse: drag outside a box to draw one, inside it to move it, a corner or an edge to resize it;
the wheel grows or shrinks it.

```bash
stretch4_pick_cameras --out ~/repos/stretch4_lerobot/cameras.json
```

- `--out`: lets you specify the location where the file will be saved. If you don't add a full path,
it'll save it in the directory you run the command from.

**Optional flags:**

- `--cameras`: which cameras to record, any of `head_left`, `head_right`, `gripper` (the right
  gripper eye) and `gripper_left`. Default: `--cameras head_left head_right gripper`.
- `--size H W`: the recorded size of every camera, height then width. Default: `--size 240 384`. The size is applied to all cameras.

**Keys in the window:**

- **Enter** saves the boxes to the cameras file.
- **r** clears the boxes on every camera, gripper included.
- **m** (head cameras only) gives one head camera the same box as the other: the one you last
  drew, moved or resized. E.g. adjust `head_left`'s box, press **m**, and `head_right` gets the
  same box. The gripper camera is never changed.
- **q** quits without saving.

You pass the file to `lerobot-record` as `--robot.cameras_file=cameras.json` (Step 3).

---

## Step 3: Record

### 3.1 Tips for good demonstrations

- **Start in roughly the same place:** Start each episode near the same pose. Small differences help: the policy learns to handle a wider range of start poses. Avoid start poses far from the others.
- **Smooth motion:** Drive the task smoothly, without long pauses.
- **Repeat each variation:** If you vary the task (object position, orientation, or type), repeat each variation in many episodes. A variation shown only once is treated as noise.
- **Re-record unwanted moves:** Re-record (`←`) an episode if you did something you don't want the policy to learn, like bumping an object or moving the wrong joint. Policies imitate what they see.
- **Keep retries:** If you miss a grasp, try again in the same episode instead of re-recording. The policy then learns to retry when it misses, which makes it more robust.

### 3.2 Save and reset to a start pose (optional)

Before you record, you can drive the arm and gripper to a reference start pose and save it. During
each reset, run `goto` from a second terminal to move the arm back.

Pick any name for the pose file. A plain name like `pick_cup.json` is saved in the repo's
`start_poses/` folder, wherever you run the command from. You pass the same name to
`stretch4_rollout` later (file 3).

```bash
stretch4_pose save <pose_name>.json
```

```bash
stretch4_pose goto <pose_name>.json
```

`goto` asks before it moves the arm. Optional flags:

- `--yes`: move without asking.
- `--no-gripper`: leave the gripper as it is.

### 3.3 Record

Run your teleop (gamepad, Stretch Dex Teleop or Meta Quest 3) as its own process, as usual. `lerobot-record`
only observes:

```bash
lerobot-record --robot.type=stretch4 --robot.mode=observe --robot.cameras_file=cameras.json \
    --teleop.type=stretch4_observer \
    --dataset.repo_id=<user>/<folder_name> --dataset.single_task="<Task Name>" \
    --dataset.num_episodes=<num_episodes> --dataset.fps=30 --dataset.streaming_encoding=true \
    --dataset.push_to_hub=false
```

**CHANGE THIS:**

- `<user>`: your Hugging Face username. Without an account (or if you will not push), any name
  works, e.g. `hello-robot`.
- `<folder_name>`: a name for the dataset, e.g. `pick_cup`.
- `<num_episodes>`: how many demonstrations to record, e.g. `50`.
- `--dataset.single_task`: the task text. **Write it down: use the exact same text as `--task` when
  you run the policy** (file 3, Step 4); a different text makes the policy behave differently.
- `--dataset.push_to_hub=false` keeps the dataset local: you prepare it (Step 4) before pushing it
  (Step 5). Without the flag, `lerobot-record` pushes it to the Hub when it finishes.

**Optional flags** (add them to the command to change them):

- `--dataset.episode_time_s`: the longest an episode can last, in seconds. Default: `60`.
- `--dataset.reset_time_s`: the longest the reset between episodes can last, in seconds.
  Default: `60`.

The recorded action is where the robot actually went a few frames later (the lookahead), not what
the teleop device asked for, so every teleop device gives the same data. Set the lookahead with
`--robot.action_lookahead=N`, in frames. Default: `3`.

The dataset is saved under `~/.cache/huggingface/lerobot/<user>/<folder_name>_YYYYMMDD_HHMMSS`
(LeRobot's default; `$HF_LEROBOT_HOME` changes it). The timestamp is added at the start of each new
recording so a run never overwrites an earlier one. To find the exact name afterwards:
`ls ~/.cache/huggingface/lerobot/<user>`.

**Recorder keys:**

- **→ (Right Arrow Key):** keep the episode and end it now (or end the reset early)
- **← (Left Arrow Key):** throw it out and re-record it
- **Esc key:** stop (the current episode is saved)

### 3.4 Resume a stopped session

Use the same `lerobot-record` command as in Step 3.3, with four changes:

1. Add `--resume=true`.
2. Add `_YYYYMMDD_HHMMSS` to the end of `--dataset.repo_id`. To find your dataset's ending, run
   `ls ~/.cache/huggingface/lerobot/<user>`.
3. Add `--dataset.root=` with the dataset's folder: `$HOME/.cache/huggingface/lerobot/` followed
   by the same repo id.
4. Change `<num_episodes>` to how many episodes are left to record.

The command will look like this:

```bash
lerobot-record --robot.type=stretch4 --robot.mode=observe --robot.cameras_file=cameras.json \
    --teleop.type=stretch4_observer \
    --dataset.repo_id=<user>/<folder_name>_YYYYMMDD_HHMMSS --dataset.single_task="<Task Name>" \
    --dataset.num_episodes=<num_episodes> --dataset.fps=30 --dataset.streaming_encoding=true \
    --dataset.push_to_hub=false --resume=true \
    --dataset.root=$HOME/.cache/huggingface/lerobot/<user>/<folder_name>_YYYYMMDD_HHMMSS
```

**CHANGE THIS:**

- `<user>/<folder_name>_YYYYMMDD_HHMMSS` (in both places): your dataset, e.g.
  `hello-robot/pick_cup_20261020_143015`.
- `<num_episodes>`: how many episodes to **add**, not the total.

---

## Step 4: Prepare the dataset (Recommended)

Run `stretch4_fix_stats` on the recorded dataset, with absolute or relative actions to match how you
will train. Check what your policy takes: if you will train it with
`--policy.use_relative_actions=true`, use relative actions; otherwise use absolute actions. We have
tested **diffusion and ACT with absolute actions** and **pi0.5 with relative actions**.

`<folder_name>_YYYYMMDD_HHMMSS` is the folder `lerobot-record` created (Step 3.3). **Run only one of
these two commands**:

- **Absolute actions**:

  ```bash
  stretch4_fix_stats <user>/<folder_name>_YYYYMMDD_HHMMSS
  ```

- **Relative actions** (you will train with `--policy.use_relative_actions=true` in [2-training-policy.md](2-training-policy.md)):

  ```bash
  stretch4_fix_stats <user>/<folder_name>_YYYYMMDD_HHMMSS --relative
  ```

  Decide these before you run it, they must match the policy you train:

  - `--chunk-size N`: how many actions the policy predicts at once. Set it to the same value as
    the policy's `chunk_size`. Default: `50`, the same as pi0.5's.
  - `--exclude-joints`: the joints kept absolute, must match the policy's `relative_exclude_joints`.
    Default `grip base_v base_omega` (`base_v` matches both `base_vx` and `base_vy`).

If the dataset is not in `~/.cache/huggingface/lerobot`, give its folder instead of the name, e.g.
`stretch4_fix_stats /path/to/dataset`.

`stretch4_push_dataset` (Step 5) runs this for you on a dataset that was never prepared, **with
absolute actions**. For relative actions, run the `--relative` command yourself before pushing.

To undo it and get the dataset back as recorded:

```bash
stretch4_fix_stats <user>/<folder_name>_YYYYMMDD_HHMMSS --undo
```

**What it does.** LeRobot scales every joint by how much it moved in the dataset, so a joint the
demos barely moved gets its sensor noise blown up to full scale. `stretch4_fix_stats`:

- **holds** each joint the demos moved in fewer than 10% of the episodes: it is recorded as a
  constant, and the robot keeps it still when it runs the policy;
- **floors** the stats of every joint, so a joint that barely moved stays near 0 after scaling;
- with `--relative`, computes the action stats for relative actions;
- changes the dataset in place. The first run saves the recording to
  `meta/stretch4_original_columns.npz`, so you can run it again with other options, or `--undo` it.

**If you ran `stretch4_fix_stats` and then recorded more episodes** (Step 3.4), run
`stretch4_fix_stats` again with the same options: the new episodes are not prepared yet. `stretch4_push_dataset` does this for you.
The robot warns when it connects if the dataset was never prepared, or got new episodes after it
was prepared.

---

## Step 5: Push to the Hugging Face Hub (Recommended)

Log in first with `hf auth login`. Paste a token from
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), and answer `N` to
"Add token as git credential?".

You might see an error (a `Traceback`) at the end. If it printed `Login successful.` before the
error, you are logged in and can ignore it.

```bash
stretch4_push_dataset [<dataset>]
```

- `<dataset>`: the local dataset to push. Leave it out to pick from a list.

**Optional flags:**

- `--repo-id <user>/<name>`: the name on the Hub. Default: asked.
- `--public`: make the dataset public. Default: asked (private).
- `--yes`: ask nothing and push straight away.
- `--list`: list the local datasets and exit.
- `--no-fix-stats`: push without preparing the dataset first (Step 4).

---

**Next:** [2-training-policy.md](2-training-policy.md)
