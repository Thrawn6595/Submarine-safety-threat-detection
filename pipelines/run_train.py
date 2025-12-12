from pathlib import Path
import sys

# Ensure local src/ is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    print("TODO: wire train pipeline")
    # Later:
    # - load data
    # - split
    # - preprocess fit on train
    # - train
    # - evaluate
    # - save artifacts

if __name__ == "__main__":
    main()
