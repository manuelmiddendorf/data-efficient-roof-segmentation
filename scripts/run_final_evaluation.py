"""Run the frozen final test evaluation without training or tuning."""
from pathlib import Path
from roofseg.final_evaluation import run_final_evaluation

if __name__ == '__main__':
    run_final_evaluation(Path(__file__).resolve().parents[1])
