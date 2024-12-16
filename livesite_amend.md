# AMENDMENT OF FRONTEND VISUALIZATION TOOL

This document aims to itemize the necessary steps required to add new results to the frontend visualization tool and or add new recourse methods.

1. The experiments directory contains a `results.csv` file that contains benchmarking results from running the `run_experiment.py` file. This is the result displayed on the frontend.

2. New recourse methods (added to the repository) that we may want to display on the site are also added to the `server.py` file in the `livesite` directory. Specifically, the newly added method should be added chronologically to the `methods_list` list at the top of the file.

   **NOTE:** Ensure all newly added recourse methods have a corresponding result in the `results.csv` file to ensure the result is displayed on the site.
