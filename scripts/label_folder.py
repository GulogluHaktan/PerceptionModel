"""Usage: python scripts/label_folder.py <images_dir> <labels_out_dir> [size]"""
import sys
sys.path.insert(0, "src")
from pfm.labeler import run_folder
run_folder(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 448)
