"""Full pipeline: closed training, GAN with validation-based extension,
open-head training, OpenMax calibration, final evaluation.

Usage: python scripts/run_pipeline.py [config.yaml]

GAN convergence: train in 10-epoch increments (resumable checkpoint). After each
increment, run the open stage and record the validation harmonic accuracy from
openmax_selection.json. Stop when an increment fails to improve it, and restore
the best increment's artifacts for final evaluation. Progress is checkpointed in
<run_dir>/pipeline_state.json, so the script can be re-run after interruption.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_rfnet.cli import load_config
from open_rfnet.training import (
    evaluate_closed,
    evaluate_open,
    train_closed,
    train_gan_stage,
    train_open_stage,
)

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "/shared/Open-RFNet/configs/paper_denoise.yaml"
GAN_INCREMENT = 10
MAX_GAN_EPOCHS = 40

config = load_config(CONFIG_PATH)
run_dir = Path(config["run_dir"])
run_dir.mkdir(parents=True, exist_ok=True)
manifest = Path(config["cache_dir"]) / "manifest.json"
assert manifest.exists(), "manifest missing"

state_path = run_dir / "pipeline_state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {}

def save_state():
    state_path.write_text(json.dumps(state, indent=2))

closed_path = run_dir / "closed.pt"
if not state.get("closed_done") or not closed_path.exists():
    print("=== stage: train-closed ===", flush=True)
    train_closed(config, manifest)
    state["closed_done"] = True
    save_state()
    closed_metrics = evaluate_closed(config, manifest, closed_path)
    (run_dir / "closed_eval.json").write_text(json.dumps(closed_metrics, indent=2))
    print("closed accuracy:", closed_metrics["accuracy"], flush=True)

best = state.get("best", {"harmonic": -1.0, "epochs": 0})
gan_history = state.get("gan_history", [])
epochs_done = state.get("gan_epochs_done", 0)
while epochs_done < MAX_GAN_EPOCHS:
    epochs_next = epochs_done + GAN_INCREMENT
    print(f"=== stage: train-gan to {epochs_next} epochs ===", flush=True)
    config["gan"]["epochs"] = epochs_next
    train_gan_stage(config, manifest, closed_path)
    print(f"=== stage: train-open (gan {epochs_next}) ===", flush=True)
    train_open_stage(config, manifest, closed_path, run_dir / "generator.pt")
    selection = json.loads((run_dir / "openmax_selection.json").read_text())
    harmonic = float(selection["selected"]["harmonic_accuracy"])
    gan_history.append({"gan_epochs": epochs_next, "val_harmonic": harmonic,
                        "selected": selection["selected"]})
    print(f"gan {epochs_next} epochs -> val harmonic {harmonic:.4f}", flush=True)
    for name in ("open.pt", "openmax.json", "openmax_selection.json", "synthetic_unknown.pt"):
        shutil.copy2(run_dir / name, run_dir / f"gan{epochs_next}_{name}")
    epochs_done = epochs_next
    state["gan_epochs_done"] = epochs_done
    state["gan_history"] = gan_history
    if harmonic > best["harmonic"] + 1e-4:
        best = {"harmonic": harmonic, "epochs": epochs_next}
        state["best"] = best
        save_state()
    else:
        state["best"] = best
        save_state()
        print(f"no improvement over {best}; stopping GAN extension", flush=True)
        break

# restore best increment's artifacts
if best["epochs"] and best["epochs"] != epochs_done:
    for name in ("open.pt", "openmax.json", "openmax_selection.json", "synthetic_unknown.pt"):
        shutil.copy2(run_dir / f"gan{best['epochs']}_{name}", run_dir / name)
    print(f"restored artifacts from gan epoch {best['epochs']}", flush=True)

print("=== stage: evaluate ===", flush=True)
metrics = evaluate_open(config, manifest, run_dir / "open.pt", run_dir / "openmax.json")
print(json.dumps({k: v for k, v in metrics.items() if k != "per_class_accuracy"}, indent=2), flush=True)
print("PIPELINE_COMPLETE", flush=True)
