# AMENDMENT OF FRONTEND VISUALIZATION TOOL

This document aims to itemize the necessary steps required to add new results to the frontend visualization tool and or add new recourse methods.

1. The root directory contains a `results.csv` file that contains benchmarking results from running the `run_experiment.py` file. This is the result displayed on the frontend.

2. New recourse methods are added in the `server.py` file in the root directory. Specifically, the newly added method may be added chronologically to the `methods_list` list at the top of the file. Ensure newly added recourse methods have a corrsponding result in the `results.csv` file to ensure the corresponding data is displayed on the site.
