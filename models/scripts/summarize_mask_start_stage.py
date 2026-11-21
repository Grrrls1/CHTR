"""CPU-only acceptance and reporting for the frozen mask/start experiments."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.experiments.mask_start_summary import aggregate
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['base','val_noise','final'],required=True);a=p.parse_args()
    aggregate(Path(a.output).resolve(),a.phase)
