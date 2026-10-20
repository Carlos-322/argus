"""Write tests/fixtures/name_readings_c916262.json: main's reading of every name in tests/test_formats.py NAME_CORPUS
by every name matcher that test_on_real_names_the_matchers_differ_from_main_only_where_main_was_wrong compares, read
from main's prepare/formats.py at MAIN_BEFORE_WHOLE_WORDS (c916262, main before the whole-word matchers of PR 60).

Run it from a checkout that holds that commit, after changing NAME_CORPUS or the matchers:
    python tests/fixtures/make_name_readings.py
"""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import test_formats as T  # noqa: E402


def main_formats():
    """main's prepare/formats.py at MAIN_BEFORE_WHOLE_WORDS, loaded as a module from git history."""
    commit = T.MAIN_BEFORE_WHOLE_WORDS
    source = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:prepare/formats.py"],
                            capture_output=True, check=True, timeout=60).stdout
    spec = importlib.util.spec_from_loader(f"formats_main_{commit}", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, f"{commit}:prepare/formats.py", "exec"), module.__dict__)
    return module


def main_matchers(old):
    """{matcher: main's reading}, by the names of test_formats.name_matchers. Main matched some of them with its own
    patterns (ACTION_TOPIC, DEPTH_TOPIC, the joint and gripper patterns, TEXT_TOPIC and the patterns written inline at
    their call sites)."""
    return {
        "side_of": old.side_of,
        "recorded_side": lambda n: sorted(filter(None, [old.recorded_state_identity(n)["side"]])),
        "is_mount_named": old.is_mount_named,
        "mounted_side": old.mounted_side,
        "scene_rank": old.scene_rank,
        "action": lambda s: bool(old.ACTION_TOPIC.search(s)),
        "depth": lambda s: bool(old.DEPTH_TOPIC.search(s)),
        "joint": lambda s: bool(old.STATE_JOINT_NAME.search(s)),
        "gripper": lambda s: bool(old.STATE_GRIPPER_NAME.search(s)),
        "text": lambda s: bool(old.TEXT_TOPIC.search(s)),
        "instruction_task": lambda s: bool(re.search("instruction|task", s, re.I)),
        "disparity": lambda s: bool(re.search(r"disparit|inverse", s, re.I)),
        "has_side": lambda s: bool(re.search(r"left|right", s, re.I)),
        "arm_topic": lambda s: bool(re.search(r"(^|[/_.-])arm([/_.-]|$)", s, re.I)),
        "ee_topic": lambda s: bool(re.search(r"(^|[/_.-])(ee|gripper)([/_.-]|$)", s, re.I)),
        "frame_times": lambda s: any(t.startswith(("time", "stamp")) for t in old.tokens(s)),
        "command": lambda s: old._names_word(s, old.COMMAND_WORDS),
        "touch": old.names_touch,
        "sensing": old.is_sensing,
        "not_rgb": old.not_rgb,
        "camera_named": old.camera_named,
        "state_words": old.state_words,
    }


def main():
    readers = main_matchers(main_formats())
    if sorted(readers) != sorted(T.name_matchers(T.f)):
        sys.exit(f"main_matchers and test_formats.name_matchers name different matchers: "
                 f"{sorted(set(readers) ^ set(T.name_matchers(T.f)))}")
    names = sorted(set(T.NAME_CORPUS))
    out = {"about": f"main's ({T.MAIN_BEFORE_WHOLE_WORDS}) reading of every tests/test_formats.py NAME_CORPUS name by "
                    "every name matcher; written by tests/fixtures/make_name_readings.py",
           "commit": T.MAIN_BEFORE_WHOLE_WORDS,
           "readings": {m: {n: read(n) for n in names} for m, read in sorted(readers.items())}}
    T.MAIN_READINGS.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {T.MAIN_READINGS.relative_to(ROOT)}: {len(readers)} matchers x {len(names)} names")


if __name__ == "__main__":
    main()
