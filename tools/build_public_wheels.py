"""Build the two public wheels; never builds or packages OptimizerPlus."""
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
out=root/'dist'
for package in ('trainerplus','learnbywire'):
    subprocess.run([sys.executable,'-m','build','--wheel','--outdir',str(out),str(root/'packages'/package)],check=True)
