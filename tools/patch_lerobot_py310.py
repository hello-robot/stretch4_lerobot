"""Make lerobot 0.6.1 importable on Python 3.10 (Jetson JetPack 6 only has cp310 CUDA torch).

Rewrites the PEP 695 generics / `type` aliases and takes the 3.11+ typing names from
typing_extensions. Idempotent.
"""
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])  # .../site-packages/lerobot


def edit(rel, *subs):
    p = root / rel
    s = p.read_text()
    for old, new in subs:
        if old not in s:
            if new in s:
                continue
            raise SystemExit(f"{rel}: pattern not found: {old!r}")
        s = s.replace(old, new, 1)
    p.write_text(s)
    print("patched", rel)


edit("utils/io_utils.py",
     ("def deserialize_json_into_object[T: JsonLike](",
      "T = TypeVar(\"T\", bound=JsonLike)\n\n\ndef deserialize_json_into_object("),
     ("from typing import Any\n", "from typing import Any, TypeVar\n"))
edit("motors/motors_bus.py",
     ("type NameOrID = str | int", "NameOrID = str | int"),
     ("type Value = int | float", "Value = int | float"))
edit("datasets/aggregate.py",
     ("type FeatureDict =", "FeatureDict ="),
     ("type ChunkFile =", "ChunkFile ="),
     ("type VideoIndexState =", "VideoIndexState ="),
     ("from typing import Any, NotRequired, TypedDict",
      "from typing import Any, TypedDict\n\nfrom typing_extensions import NotRequired"))
edit("datasets/streaming_dataset.py",
     ("class Backtrackable[T]:", "T = TypeVar(\"T\")\n\n\nclass Backtrackable(Generic[T]):"),
     ("from typing import Literal\n", "from typing import Generic, Literal, TypeVar\n"))
edit("processor/pipeline.py",
     ("class DataProcessorPipeline[TInput, TOutput](HubMixin):",
      "class DataProcessorPipeline(HubMixin, Generic[TInput, TOutput]):"),
     ("from typing import Any, TypedDict, TypeVar, cast",
      "from typing import Any, Generic, TypedDict, TypeVar, cast"))
edit("configs/video.py",
     ("from typing import Any, ClassVar, Self",
      "from typing import Any, ClassVar\n\nfrom typing_extensions import Self"))
edit("utils/transition.py",
     ("from typing import NotRequired, TypedDict",
      "from typing import TypedDict\n\nfrom typing_extensions import NotRequired"))
for rel in ["policies/factory.py", "policies/pretrained.py", "policies/evo1/modeling_evo1.py",
            "policies/pi0/modeling_pi0.py", "policies/pi0_fast/modeling_pi0_fast.py",
            "policies/pi05/modeling_pi05.py", "policies/smolvla/modeling_smolvla.py"]:
    p = root / rel
    s = p.read_text()
    m = re.search(r"^from typing import (.*\bUnpack\b.*)$", s, re.M)
    if m:
        names = [n.strip() for n in m.group(1).split(",") if n.strip() != "Unpack"]
        s = s.replace(m.group(0), f"from typing import {', '.join(names)}\n\nfrom typing_extensions import Unpack", 1)
        p.write_text(s)
        print("patched", rel)
