"""Column names shared by every Stretch 4 dataset, the robot and the policies."""

# The joints a policy commands, in action order.
POSE_JOINTS = ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "grip_mm")

# The action: the base's body-frame velocity, then the joints (absolute). A base-pose layout
# (odometry x, y, theta) is only recognised, never fed.
VELOCITY_ACTION_COLUMNS = ["base_vx", "base_vy", "base_omega", *POSE_JOINTS]
POSE_ACTION_COLUMNS = ["base_x", "base_y", "base_theta", *POSE_JOINTS]

# What a policy sees, in every dataset and on the robot: the velocity action's columns, so
# state[i] and action[i] are the same quantity at every index.
POLICY_STATE_COLUMNS = list(VELOCITY_ACTION_COLUMNS)

# Optional last state column. Only ever in the state: effort is never commanded, so no action
# carries it, and state[i] and action[i] still match for every action column.
GRIP_EFFORT = "grip_effort"


def policy_state_columns(grip_effort=False):
    """The policy's state names: POLICY_STATE_COLUMNS, plus grip_effort last when asked."""
    return [*POLICY_STATE_COLUMNS, GRIP_EFFORT] if grip_effort else list(POLICY_STATE_COLUMNS)


def check_state_names(state_names, action_names):
    """None if state and action line up as every Stretch 4 dataset does, else why not.

    The state is POLICY_STATE_COLUMNS, optionally followed by grip_effort. The action is one of
    the two action layouts. Index for index they then hold the same joint (3..8) and, with a
    velocity action, the same base velocity (0..2)."""
    state, action = list(state_names), list(action_names)
    if action not in (VELOCITY_ACTION_COLUMNS, POSE_ACTION_COLUMNS):
        return (f"action {action} is neither {VELOCITY_ACTION_COLUMNS} nor {POSE_ACTION_COLUMNS}")
    if state not in (POLICY_STATE_COLUMNS, [*POLICY_STATE_COLUMNS, GRIP_EFFORT]):
        return (f"observation.state {state} is not {POLICY_STATE_COLUMNS}, optionally followed by "
                f"{GRIP_EFFORT}")
    return None


# The gripper a dataset or sidecar without a gripper_tool tag was recorded with.
DEFAULT_GRIPPER_TOOL = "stretch_gripper"


def gripper_mismatch(trained, live):
    """Compare the gripper a policy was trained with to the gripper on the robot.

    Returns None if they are the same gripper, else an error message explaining the mismatch.
    """
    trained, live = trained or DEFAULT_GRIPPER_TOOL, live or DEFAULT_GRIPPER_TOOL
    if trained == live:
        return None
    return (f"the policy was trained with the {trained}, the robot has the {live}: the gripper "
            "camera's view, the fingers and grip_effort differ, so the policy sees inputs it "
            "never trained on")
