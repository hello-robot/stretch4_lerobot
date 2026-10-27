"""Push a locally recorded dataset to the Hugging Face Hub (recorded with --dataset.push_to_hub=false).

    stretch4_push_dataset                       # interactive: pick a dataset, name it, push
    stretch4_push_dataset --list                # just list the local datasets
"""

import argparse
import json
import re
import sys
from pathlib import Path


def lerobot_home() -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME
    return Path(HF_LEROBOT_HOME)


def is_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def resolve(dataset: str, home: Path) -> Path:
    """A folder, or a repo id under the LeRobot home."""
    for path in (Path(dataset).expanduser(), home / dataset):
        if is_dataset(path):
            return path.resolve()
    raise SystemExit(f"no LeRobot dataset at {dataset} or {home / dataset}; "
                     "run stretch4_push_dataset with no arguments to list them")


def local_repo_id(root: Path) -> str:
    """<user>/<name> from the folder layout <home>/<user>/<name>."""
    return f"{root.parent.name}/{root.name}"


def summary(root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    cameras = [k.removeprefix("observation.images.") for k in info["features"]
               if k.startswith("observation.images.")]
    return {"episodes": info.get("total_episodes", 0), "frames": info.get("total_frames", 0),
            "fps": info.get("fps"), "cameras": cameras, "size_mb": size / 1e6,
            "version": info.get("codebase_version")}


STAMP = re.compile(r"_\d{8}_\d{6}$")    # lerobot-record's stamp_repo_id suffix


def find_datasets(home: Path) -> list[Path]:
    return sorted(p.parent.parent for p in home.glob("*/*/meta/info.json"))


def list_datasets(home: Path, found: list[Path]) -> None:
    if not found:
        print(f"no datasets in {home}")
        return
    print(f"datasets in {home}:")
    for i, root in enumerate(found, 1):
        s = summary(root)
        print(f"  {i:2d}. {local_repo_id(root):50s} {s['episodes']:4d} episodes  "
              f"{s['size_mb']:7.0f} MB  {', '.join(s['cameras'])}")


def suggested_name(root: Path) -> str:
    return f"{root.parent.name}/{STAMP.sub('', root.name)}"


def valid_repo_id(repo_id: str) -> bool:
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id))


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("not pushed")


def pick(home: Path, found: list[Path]) -> Path:
    list_datasets(home, found)
    if not found:
        raise SystemExit(1)
    while True:
        choice = ask(f"dataset to push [1-{len(found)}, or its name]: ")
        if choice.isdigit() and 1 <= int(choice) <= len(found):
            return found[int(choice) - 1]
        try:
            return resolve(choice, home)
        except SystemExit:
            print(f"  not one of the datasets above: {choice!r}")


def ask_name(root: Path) -> str:
    default = suggested_name(root)
    while True:
        name = ask(f"name on the Hub [{default}]: ") or default
        if "/" not in name:
            name = f"{root.parent.name}/{name}"     # just the name: same user
        if valid_repo_id(name):
            return name
        print(f"  must be USER/NAME (letters, digits, - _ .), got {name!r}")


def hub_exists(repo_id: str) -> bool:
    from huggingface_hub import repo_exists
    try:
        return repo_exists(repo_id, repo_type="dataset")
    except Exception:
        return False


def whoami():
    from huggingface_hub import whoami as hf_whoami
    try:
        return hf_whoami()["name"]
    except Exception:
        raise SystemExit("not logged in to the Hugging Face Hub: run  hf auth login")


def push(repo_id: str, root: Path, private: bool):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LeRobotDataset(repo_id, root=root).push_to_hub(private=private)


def ensure_prepared(root: Path, interactive: bool) -> None:
    """Run stretch4_fix_stats if the dataset was never prepared or changed since; a changed one
    with the options it was prepared with (--relative stays relative)."""
    from . import sidecar
    from .prepare_dataset import is_stale, options_from, prepare

    side = sidecar.read(root) or {}
    if side.get("schema") != sidecar.SCHEMA:
        raise SystemExit(f"{root}: {sidecar.SIDECAR} is {side.get('schema')!r}, not "
                         f"{sidecar.SCHEMA!r}; re-record it")
    why = ("it was never prepared" if not side.get("prepared") else is_stale(root, side))
    if not why:
        return
    print(f"{root}: {why}. stretch4_fix_stats holds the joints the demos rarely moved and floors "
          "the stats, so the policy does not train on amplified joint noise.")
    opts = options_from(side.get("prepared"))
    command = "stretch4_fix_stats" + (" --relative" if opts.get("relative") else "")
    if interactive and ask(f"Run {command} now? [Y/n]: ").lower() in ("n", "no"):
        print(f"not prepared: run {command} before training on it")
        return
    prepare(root, **opts)


def main(argv=None):
    summary_line, _, epilog = __doc__.partition("\n\n")
    p = argparse.ArgumentParser(prog="stretch4_push_dataset", description=summary_line,
                                epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", metavar="DATASET",
                   help="local repo id or folder (omit to pick from a list)")
    p.add_argument("--repo-id", metavar="USER/NAME",
                   help="name on the Hub (default: asked; with --yes, DATASET's repo id)")
    p.add_argument("--public", action="store_true", help="make it public (default: asked, private)")
    p.add_argument("--yes", action="store_true", help="ask nothing, push straight away")
    p.add_argument("--list", action="store_true", help="list the local datasets and exit")
    p.add_argument("--no-fix-stats", action="store_true",
                   help="push without running stretch4_fix_stats on an unprepared dataset")
    args = p.parse_args(argv)

    home = lerobot_home()
    found = find_datasets(home)
    if args.list:
        list_datasets(home, found)
        return 0
    interactive = not args.yes

    root = resolve(args.dataset, home) if args.dataset else (
        pick(home, found) if interactive else p.error("--yes needs DATASET"))
    s = summary(root)
    if not s["episodes"]:
        raise SystemExit(f"{root} has no episodes")
    user = whoami()

    if args.repo_id:
        repo_id = args.repo_id
    elif interactive:
        repo_id = ask_name(root)
    else:
        repo_id = local_repo_id(root)
    if not valid_repo_id(repo_id):
        raise SystemExit(f"--repo-id must be USER/NAME, got {repo_id!r}")

    if not args.no_fix_stats:
        ensure_prepared(root, interactive)

    public = args.public
    if interactive and not args.public:
        public = ask("private? [Y/n]: ").lower() in ("n", "no")

    print(f"\nlocal:    {root}")
    print(f"          {s['episodes']} episodes, {s['frames']} frames at {s['fps']} fps, "
          f"cameras {', '.join(s['cameras'])}, {s['size_mb']:.0f} MB ({s['version']})")
    print(f"push to:  https://huggingface.co/datasets/{repo_id}  "
          f"({'PUBLIC' if public else 'private'}, as {user})")
    if hub_exists(repo_id):
        print(f"WARNING:  {repo_id} already exists on the Hub; pushing adds to / overwrites it")
    if interactive and ask("Push? [y/N]: ").lower() not in ("y", "yes"):
        print("not pushed")
        return 1

    push(repo_id, root, private=not public)
    print(f"pushed https://huggingface.co/datasets/{repo_id}\n"
          f"  train with: lerobot-train --dataset.repo_id={repo_id} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
