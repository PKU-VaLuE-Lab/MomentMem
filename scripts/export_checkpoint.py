#!/usr/bin/env python3
"""Export inference files without training state, local paths, or device topology."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

VLM_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "chat_template.json", "tokenizer.model",
)
PROMPT_KEYS = (
    "add_memory_instruction", "buffer_size", "max_history", "camera_mode",
    "use_wrist", "camera_keys", "num_cameras", "video_fps", "sample_fps",
    "retrieval_stride", "image_preprocess", "image_size", "max_length", "prompt_layout",
)
NORMALIZATION_KEYS = (
    "schema_version", "normalization_type", "policy_contract_sha256", "norm_stats_sha256",
)


def strip_source_fields(value):
    if isinstance(value, dict):
        return {key: strip_source_fields(item) for key, item in value.items()
                if key not in {"_name_or_path", "name_or_path", "_commit_hash"}}
    if isinstance(value, list):
        return [strip_source_fields(item) for item in value]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def export_vlm(source, destination):
    for name in ("config.json", "tokenizer_config.json", "keyframe_prompt_config.json"):
        if not (source / name).is_file():
            raise ValueError(f"Missing required VLM file: {name}")
    index = source / "model.safetensors.index.json"
    if index.is_file():
        index_data = json.loads(index.read_text())
        weights = sorted(set(index_data["weight_map"].values()))
        shutil.copyfile(index, destination / index.name)
    else:
        weights = ["model.safetensors"]
    for name in weights:
        shutil.copyfile(source / name, destination / name)
    for name in VLM_FILES:
        path = source / name
        if not path.is_file():
            continue
        # Vocabulary/tokenizer JSON is model data, not provenance.
        if name in {"config.json", "tokenizer_config.json", "preprocessor_config.json",
                    "video_preprocessor_config.json", "generation_config.json"}:
            write_json(destination / name, strip_source_fields(json.loads(path.read_text())))
        else:
            shutil.copyfile(path, destination / name)
    prompt = json.loads((source / "keyframe_prompt_config.json").read_text())
    write_json(destination / "keyframe_prompt_config.json",
               {key: prompt[key] for key in PROMPT_KEYS if key in prompt})


def export_vla(source, destination):
    params = source / "params"
    if not (params / "_METADATA").is_file():
        raise ValueError("Missing Orbax parameter metadata")
    assets = source / "assets"
    contract = copy.deepcopy(json.loads((assets / "policy_contract.json").read_text()))
    normalization = contract["normalization"]
    asset_id = normalization["asset_id"]
    stats = assets / asset_id / "norm_stats.json"
    # Device sharding is supplied by the inference loader on the target machine.
    shutil.copytree(params, destination / "params", ignore=shutil.ignore_patterns("_sharding"))
    contract["normalization"] = {key: normalization[key] for key in NORMALIZATION_KEYS}
    contract["normalization"]["asset_id"] = "policy"
    write_json(destination / "assets/policy_contract.json", contract)
    (destination / "assets/policy").mkdir()
    shutil.copyfile(stats, destination / "assets/policy/norm_stats.json")
    shutil.copyfile(assets / "memory_layout_contract.json",
                    destination / "assets/memory_layout_contract.json")


def export_checkpoint(kind, source, output):
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Export destination already exists")
    if output == source or source in output.parents:
        raise ValueError("Export destination must be outside the source checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".inference-export-", dir=output.parent))
    try:
        if kind == "vlm":
            export_vlm(source, staging)
        elif kind == "vla":
            export_vla(source, staging)
        else:
            raise ValueError("Model kind must be vlm or vla")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("vlm", "vla"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_checkpoint(args.kind, args.source, args.output)
    print("Inference checkpoint exported.")
