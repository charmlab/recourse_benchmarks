import os
import shutil

cache_dir = ".pytest_cache"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Deleted {cache_dir}")
else:
    print(f"{cache_dir} not found")
