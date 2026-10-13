"""Minimal stand-in for the BIRD script's ``tool.utils`` dependency.

The official ``bird_evaluation_raw.py`` does ``from tool.utils import
round_floats_in_structure``. In this workspace that module also imports
``chromadb`` (unrelated to evaluation) which is not installed. We add this tiny
package to ``sys.path`` ahead of it so the official execution-accuracy logic runs
unchanged, without pulling in retrieval dependencies.

``round_floats_in_structure`` is copied verbatim from the upstream implementation.
"""


def round_floats_in_structure(data, precision=6):
    if isinstance(data, float):
        return round(data, precision)
    elif isinstance(data, list):
        return [round_floats_in_structure(item, precision) for item in data]
    elif isinstance(data, tuple):
        return tuple(round_floats_in_structure(item, precision) for item in data)
    else:
        return data
