
from pathlib import Path
from lab.engine.plot_results import plot_model_result

def main():
    # Hardcoded to the run ID found in the user's output
    # run_20251128_221544
    # But to be safe/general, let's find the latest run in manual_test again
    root = Path("runs/sweeps/manual_test")
    if not root.exists():
        print(f"Root {root} does not exist.")
        return

    # Find latest run
    runs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    if not runs:
        print("No runs found.")
        return
        
    latest_run = runs[-1]
    print(f"Charting latest run: {latest_run}")
    
    plot_model_result(latest_run)

if __name__ == "__main__":
    main()
