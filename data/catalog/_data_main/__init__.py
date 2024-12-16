import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "raw_data"))
sys.path.append(os.path.join(current_dir, "process_data"))
