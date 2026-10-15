#!/usr/bin/env python3
"""API-key-only bootstrap. This entry point needs only Python's standard library."""
import argparse
import getpass
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", help="DeepSeek key (DEEPSEEK_API_KEY is also supported)")
    parser.add_argument("--dataset", choices=["both", "bird", "spider200", "spider1"], default="both")
    parser.add_argument("--suite", choices=["all", "main", "relations", "policies", "mechanisms",
                        "commit", "perturbations", "tokens", "aligned", "thresholds"], default="all")
    parser.add_argument("--limit", type=int, help="Optional small validation sample per benchmark")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--model", default=os.environ.get("GWV2_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("GWV2_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--data-dir", default=os.environ.get("GWV2_DATA_DIR"))
    parser.add_argument("--outdir")
    parser.add_argument("--prepare-only", action="store_true", help="Install and validate data; no API calls")
    parser.add_argument("--experiments", nargs="+", help="Run named configurations from the selected suite")
    args = parser.parse_args()
    if sys.version_info < (3, 9):
        parser.error("Python 3.9 or newer is required")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir or root/".data").expanduser().resolve()
    env = os.environ.copy()
    env["GWV2_DATA_DIR"] = str(data_dir)
    env["GWV2_BASE_URL"] = args.base_url
    env["PYTHONUNBUFFERED"] = "1"
    if not args.prepare_only:
        key = args.api_key or env.get("DEEPSEEK_API_KEY") or env.get("DS_API_KEY") or env.get("OPENAI_API_KEY")
        if not key and sys.stdin.isatty():
            key = getpass.getpass("DeepSeek API key: ")
        if not key:
            parser.error("Supply --api-key or DEEPSEEK_API_KEY")
        env["DEEPSEEK_API_KEY"] = key
        # The explicit key always wins over a stale provider variable.
        env["DS_API_KEY"] = key
    environment = root/".venv"
    python = environment/("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        print("Creating the isolated Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    requirements = root/"requirements.txt"
    signature = hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp = environment/".requirements.sha256"
    if not stamp.is_file() or stamp.read_text().strip() != signature:
        print("Installing pinned dependencies...", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                        "-r", str(requirements)], cwd=root, env=env, check=True)
        stamp.write_text(signature+"\n")
    subprocess.run([str(python), "-m", "gws.setup", "--dataset", args.dataset],
                   cwd=root, env=env, check=True)
    if args.prepare_only:
        print("Preparation complete. No API calls were made.", flush=True)
        return
    command = [str(python), str(root/"run_experiments.py"), "--dataset", args.dataset,
               "--suite", args.suite, "--sample-seed", str(args.sample_seed), "--model", args.model]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    if args.outdir:
        command += ["--outdir", str(Path(args.outdir).expanduser().resolve())]
    if args.experiments:
        command += ["--experiments", *args.experiments]
    print("Starting the requested experiment matrix; progress and results are saved locally.", flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        # Do not print child arguments or the API-key-bearing parent arguments.
        print(f"Preparation/run failed (child exit {exc.returncode}).", file=sys.stderr)
        sys.exit(exc.returncode)
