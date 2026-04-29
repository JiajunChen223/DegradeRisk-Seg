from __future__ import annotations

import _bootstrap  # noqa: F401

from scripts.common import load_experiment, maybe_prepare_data, parse_args


def main() -> None:
    args = parse_args("Build canonical Plot-Rice dataset")
    config, run_dir = load_experiment(args)
    maybe_prepare_data(config, prepare_normalization=not args.skip_normalization)
    source_type = config["data"].get("source", {}).get("type", "canonical_npz")
    message = f"Data preparation finished. source_type={source_type}, normalization_prepared={not args.skip_normalization}\n"
    (run_dir / "build.log").write_text(message, encoding="utf-8")


if __name__ == "__main__":
    main()
