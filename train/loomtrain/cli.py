"""loomtrain — one entry point over loom's training-plane modules.

Each subcommand is a module that still runs standalone
(`python3 -u -m loomtrain.recover_lora ...`); this dispatcher only rewrites
argv and hands over, so the battle-tested scripts keep their exact behavior.
The inference twin is the `loom` Zig binary; the two meet only through
content-addressed artifact files (LPG1 heads, LRA1 adapters, GGUF).
"""

import runpy
import sys

SUBCOMMANDS = {
    "recover-lora": "loomtrain.recover_lora",
    "pregate-probe": "loomtrain.pregate_probe",
    "pregate-dump": "loomtrain.pregate_dump",
    "pregate-export": "loomtrain.pregate_export",
    "router-retrain": "loomtrain.router_retrain",
    "btx": "loomtrain.btx",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in SUBCOMMANDS:
        print("loomtrain — loom's training plane\n\nsubcommands:")
        for name, mod in SUBCOMMANDS.items():
            print(f"  {name:16s} ({mod})")
        print("\neach forwards its remaining arguments to the module unchanged;")
        print("run any of them with --help for details.")
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 2)
    sub = sys.argv[1]
    mod = SUBCOMMANDS[sub]
    sys.argv = [f"loomtrain {sub}"] + sys.argv[2:]
    runpy.run_module(mod, run_name="__main__")


if __name__ == "__main__":
    main()
