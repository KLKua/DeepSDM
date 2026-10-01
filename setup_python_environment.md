# Python package and environment management for SEAM-SDM

This document describes how to create, isolate, verify, register, and use the Python environment for running SEAM-SDM analyses. It is intended to make the Python runtime reproducible and to avoid two common problems:

1. the Jupyter kernel uses a different Python interpreter than the terminal;
2. Python imports packages from user-level or system-level locations instead of the intended conda environment.

The examples below use the conda environment name:

```bash
SEAM-SDM-PYTHON
```

Throughout this document, replace:

```bash
<PROJECT_DIR>
```

with the path to your local SEAM-SDM project directory.

---

## 1. Requirements file

Place the following package list in `requirements.txt` in the project root.

```text
mlflow==2.22.0
boto3==1.39.17
click==8.3.3
cloudpickle==3.1.2
defusedxml==0.7.1
gitpython==3.1.31
ipython==9.13.0
matplotlib==3.10.9
numpy==1.26.4
packaging==24.0
prometheus-client==0.25.0
protobuf==4.24.4
pytorch-lightning==2.5.1.post0
pyyaml==6.0.3
rasterio==1.4.3
regex==2026.4.4
requests==2.33.1
torch==2.10.0
torchvision==0.25.0
cryptography==41.0.7
opencv-python-headless==4.11.0.86
h5py==3.13.0
umap-learn==0.5.12
feather-format==0.4.1
statsmodels==0.14.6
ipywidgets==8.1.8
jupyterlab_widgets==3.0.16
widgetsnbextension==4.0.15
```
---

## 2. Create a clean conda environment

Create a new conda environment with Python 3.12 and pip:

```bash
conda create -n SEAM-SDM-PYTHON -c conda-forge python=3.12 pip ipykernel -y
conda activate SEAM-SDM-PYTHON
```

Confirm that the shell is using the intended Python interpreter:

```bash
which python
python -c "import sys; print(sys.executable)"
```

The output should point to the Python executable inside the `SEAM-SDM-PYTHON` conda environment.

---

## 3. Prevent user-site package contamination

Python may import packages from user-level or system-level locations if they are visible on `sys.path`. This can make runtime package versions different from those installed in the conda environment.

To prevent user-site contamination, set:

```bash
export PYTHONNOUSERSITE=1
```

For a permanent fix, create activation and deactivation scripts inside the conda environment:

```bash
conda activate SEAM-SDM-PYTHON

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/disable_user_site.sh" <<'EOS'
export PYTHONNOUSERSITE=1
EOS

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/disable_user_site.sh" <<'EOS'
unset PYTHONNOUSERSITE
EOS

conda deactivate
conda activate SEAM-SDM-PYTHON
```

Check that user-site packages are disabled:

```bash
python - <<'PY'
import sys, site
print("Python executable:", sys.executable)
print("Conda prefix:", sys.prefix)
print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
print("USER_SITE:", site.getusersitepackages())
print("Paths containing .local:", [p for p in sys.path if ".local" in p])
PY
```

Expected result:

```text
ENABLE_USER_SITE: False
Paths containing .local: []
```

If `PYTHONPATH` has been manually set, it may also contaminate the environment. Check and clear it if necessary:

```bash
echo "$PYTHONPATH"
unset PYTHONPATH
```

---

## 4. Install Python packages

From the project root:

```bash
cd <PROJECT_DIR>
conda activate SEAM-SDM-PYTHON
export PYTHONNOUSERSITE=1
unset PYTHONPATH

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-user --no-cache-dir -r requirements.txt
```

Always install packages using the Python executable from the active conda environment:

```bash
python -m pip install package_name
```

Do not use:

```bash
pip install --user package_name
sudo pip install package_name
```

After installation, check dependency consistency:

```bash
python -m pip check
```

---

## 5. Register the conda environment as a Jupyter kernel

Install and register the kernel:

```bash
conda activate SEAM-SDM-PYTHON
export PYTHONNOUSERSITE=1
unset PYTHONPATH

python -m pip install --no-user ipykernel
python -m ipykernel install --user \
  --name SEAM-SDM-PYTHON \
  --display-name "Python (SEAM-SDM-PYTHON)"
```

Set the kernel itself to disable user-site packages. This is important because Jupyter may not inherit your terminal environment variables.

```bash
python - <<'PY'
import json
from pathlib import Path

kernel_base = Path.home() / ".local/share/jupyter/kernels"
target_display = "Python (SEAM-SDM-PYTHON)"

matches = []
for p in kernel_base.glob("*"):
    kj = p / "kernel.json"
    if not kj.exists():
        continue
    try:
        d = json.loads(kj.read_text())
    except Exception:
        continue
    if d.get("display_name") == target_display or p.name.lower() == "seam-sdm-runversion":
        matches.append((p, kj, d))

if not matches:
    raise SystemExit("Could not find the SEAM-SDM-PYTHON kernel. Run `jupyter kernelspec list` to inspect kernel names.")

for p, kj, d in matches:
    d["env"] = d.get("env", {})
    d["env"]["PYTHONNOUSERSITE"] = "1"
    d["env"]["PYTHONPATH"] = ""
    kj.write_text(json.dumps(d, indent=2))
    print("Updated:", kj)
    print(json.dumps(d, indent=2))
PY
```

List available kernels:

```bash
jupyter kernelspec list
```

In Jupyter Notebook or JupyterLab, select:

```text
Kernel -> Change Kernel -> Python (SEAM-SDM-PYTHON)
```

Verify inside a notebook cell:

```bash
python - <<'PY'
import sys, site
print(sys.executable)
print("sys.prefix:", sys.prefix)
print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
print("Paths containing .local:", [p for p in sys.path if ".local" in p])
PY
```

Expected result:

```text
ENABLE_USER_SITE: False
Paths containing .local: []
```

The executable path should point to the `SEAM-SDM-PYTHON` conda environment.

---

## 6. Check actual runtime package versions

The version shown by `pip show` is the installed distribution metadata. The version that matters during execution is the package that Python actually imports at runtime. Therefore, always check both:

- `metadata`: the installed distribution version known to Python packaging metadata;
- `runtime`: the version exposed by the module that Python actually imports;
- `runtime_file`: the physical file path loaded by Python;
- `source`: whether the module came from the active conda environment.

Run the following command directly in the terminal. It does not create a separate `.py` file.

```bash
cd <PROJECT_DIR>
conda activate SEAM-SDM-PYTHON
export PYTHONNOUSERSITE=1
unset PYTHONPATH

python - <<'PYCODE'
import sys
import site
import importlib
import importlib.metadata as md

ENV_PREFIX = sys.prefix

targets = [
    # display_name, distribution_name, expected_distribution_version, import_module
    ('mlflow', 'mlflow', '2.22.0', 'mlflow'),
    ('boto3', 'boto3', '1.39.17', 'boto3'),
    ('click', 'click', '8.3.3', 'click'),
    ('cloudpickle', 'cloudpickle', '3.1.2', 'cloudpickle'),
    ('defusedxml', 'defusedxml', '0.7.1', 'defusedxml'),
    ('gitpython', 'gitpython', '3.1.31', 'git'),
    ('ipython', 'ipython', '9.13.0', 'IPython'),
    ('matplotlib', 'matplotlib', '3.10.9', 'matplotlib'),
    ('numpy', 'numpy', '1.26.4', 'numpy'),
    ('packaging', 'packaging', '24.0', 'packaging'),
    ('prometheus-client', 'prometheus-client', '0.25.0', 'prometheus_client'),
    ('protobuf', 'protobuf', '4.24.4', 'google.protobuf'),
    ('pytorch-lightning', 'pytorch-lightning', '2.5.1.post0', 'pytorch_lightning'),
    ('pyyaml', 'pyyaml', '6.0.3', 'yaml'),
    ('rasterio', 'rasterio', '1.4.3', 'rasterio'),
    ('regex', 'regex', '2026.4.4', 'regex'),
    ('requests', 'requests', '2.33.1', 'requests'),
    ('torch', 'torch', '2.10.0', 'torch'),
    ('torchvision', 'torchvision', '0.25.0', 'torchvision'),
    ('cryptography', 'cryptography', '41.0.7', 'cryptography'),
    ('opencv-python-headless', 'opencv-python-headless', '4.11.0.86', 'cv2'),
    ('h5py', 'h5py', '3.13.0', 'h5py'),
    ('umap-learn', 'umap-learn', '0.5.12', 'umap'),
    ('feather-format', 'feather-format', '0.4.1', 'feather'),
    ('statsmodels', 'statsmodels', '0.14.6', 'statsmodels'),
    ('ipywidgets', 'ipywidgets', '8.1.8', 'ipywidgets'),
    ('jupyterlab_widgets', 'jupyterlab_widgets', '3.0.16', 'jupyterlab_widgets'),
    ('widgetsnbextension', 'widgetsnbextension', '4.0.15', 'widgetsnbextension'),
]

print("Python executable:", sys.executable)
print("Python version:", sys.version.replace("\n", " "))
print("sys.prefix:", sys.prefix)
print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
print("USER_SITE:", site.getusersitepackages())
print("Paths containing .local:", [p for p in sys.path if ".local" in p])
print()

header = f"{'package':<24} {'expected':<14} {'metadata':<14} {'runtime':<14} {'metadata_status':<16} {'source':<10} runtime_file"
print(header)
print("-" * len(header))

for display_name, dist_name, expected, module_name in targets:
    try:
        metadata_version = md.version(dist_name)
    except md.PackageNotFoundError:
        metadata_version = "NOT FOUND"

    try:
        module = importlib.import_module(module_name)
        runtime_version = getattr(module, "__version__", None)
        if runtime_version is None:
            runtime_version = "NO __version__"
        runtime_file = getattr(module, "__file__", "built-in/namespace")
    except Exception as e:
        runtime_version = "IMPORT ERROR"
        runtime_file = f"{type(e).__name__}: {e}"

    if expected == "UNPINNED":
        metadata_status = "UNPINNED"
    elif metadata_version == expected:
        metadata_status = "EXPECTED_OK"
    elif metadata_version == "NOT FOUND":
        metadata_status = "MISSING"
    else:
        metadata_status = "EXPECTED_DIFF"

    if isinstance(runtime_file, str) and runtime_file.startswith(ENV_PREFIX):
        source = "FROM_ENV"
    elif runtime_version == "IMPORT ERROR":
        source = "IMPORT_ERR"
    else:
        source = "FROM_OTHER"

    print(
        f"{display_name:<24} {expected:<14} {metadata_version:<14} "
        f"{str(runtime_version):<14} {metadata_status:<16} {source:<10} {runtime_file}"
    )

try:
    import torch
    print("\nTorch CUDA check:")
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU capability:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("\nTorch CUDA check failed:", repr(e))
PYCODE
```

Interpretation:

- `runtime` is the version reported by the module that Python actually imports.
- `runtime_file` shows the exact file loaded by Python.
- `source` should be `FROM_ENV`. If it is `FROM_OTHER`, Python is importing from outside the intended conda environment.
- `metadata_status` should usually be `EXPECTED_OK`.
- For some packages, the distribution version and runtime module version may not be identical. For example, `opencv-python-headless==4.11.0.86` may report `cv2.__version__` as `4.11.0`. In that case, use both `metadata` and `runtime_file` to verify the installed package and actual import source.

---

## 7. Run SEAM-SDM Python scripts from terminal

All SEAM-SDM `.py` scripts should be executed from the project root using the same environment setup.

```bash
cd <PROJECT_DIR>
conda activate SEAM-SDM-PYTHON
export PYTHONNOUSERSITE=1
unset PYTHONPATH

python <script_name>.py
```

Examples:

```bash
python 02_train_deepsdm.py
python some_other_script.py
```

For maximum safety, call the environment through `conda run`:

```bash
cd <PROJECT_DIR>
conda run -n SEAM-SDM-PYTHON env PYTHONNOUSERSITE=1 PYTHONPATH= python <script_name>.py
```

You can also run a script from any working directory by giving the full script path:

```bash
conda run -n SEAM-SDM-PYTHON env PYTHONNOUSERSITE=1 PYTHONPATH= python <PROJECT_DIR>/<script_name>.py
```

---

## 8. Troubleshooting

### 8.1 Python imports packages from user-level locations

Symptom:

```text
FROM_OTHER <user-level-or-system-level-package-path>
```

Fix:

```bash
conda activate SEAM-SDM-PYTHON
export PYTHONNOUSERSITE=1
unset PYTHONPATH
python -c "import sys, site; print(site.ENABLE_USER_SITE); print([p for p in sys.path if '.local' in p])"
```

If using Jupyter, update the kernel `kernel.json` as shown in Section 5.

---

### 8.2 Jupyter cannot save notebook: readonly database

Symptom:

```text
attempt to write a readonly database
```

Usually this is caused by a read-only Jupyter notebook signature database.

Fix:

```bash
mkdir -p ~/.local/share/jupyter
mv ~/.local/share/jupyter/nbsignatures.db \
   ~/.local/share/jupyter/nbsignatures.db.bak_$(date +%Y%m%d_%H%M%S)
```

Restart Jupyter after this.

Also check that the notebook directory is writable:

```bash
cd <PROJECT_DIR>
touch test_write_permission.tmp && rm test_write_permission.tmp
```
