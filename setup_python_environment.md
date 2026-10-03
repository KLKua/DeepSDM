# Python package and environment management for SEAM-SDM

This document describes how to create, isolate, verify, register, and use the Python environment for running SEAM-SDM analyses. 

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

Use the [requirements.txt](requirements.txt) included in the project root. It is
the single source for the package versions installed in Section 4 and checked in
Section 6. Keep it alongside this guide; there is no need to recreate the file.

This list pins the packages it names, but does not yet lock every transitive
dependency or the complete Conda environment.

---

## 2. Create a clean conda environment

Create a new conda environment with Python 3.12.13 and pip:

```bash
conda create -n SEAM-SDM-PYTHON -c conda-forge python=3.12.13 pip ipykernel -y
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

The reference environment uses the CUDA 12.8 (`cu128`) build of PyTorch on
Linux x86_64 with NVIDIA GPUs. The commands below select that build and use
`requirements.txt` to specify the package versions. A supported NVIDIA GPU and
compatible driver are required. Other hardware may need a different build of
the same PyTorch version; see the [official installation options](https://pytorch.org/get-started/previous-versions/#v2100).

From the project root:

```bash
cd <PROJECT_DIR>
conda activate SEAM-SDM-PYTHON
export PYTHONNOUSERSITE=1
unset PYTHONPATH

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-user --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 --constraint requirements.txt torch torchvision
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

This section requires an existing JupyterLab or Jupyter Notebook installation.
If needed, follow the [official installation and launch instructions](https://jupyter.org/install)
in a separate environment. 

Install and register the kernel:

```bash
conda activate SEAM-SDM-PYTHON
export PYTHONNOUSERSITE=1
unset PYTHONPATH

python -m pip install --no-user ipykernel
python -m ipykernel install --user \
  --name SEAM-SDM-PYTHON \
  --display-name "Python (SEAM-SDM-PYTHON)" \
  --env PYTHONNOUSERSITE 1 \
  --env PYTHONPATH ""
```

The `--env` options save these settings with the kernel, so user-site packages
are disabled and `PYTHONPATH` is cleared whenever it starts, regardless of the
terminal environment used to launch Jupyter.

List available kernels:

```bash
jupyter kernelspec list
```

Open a project notebook in Jupyter Notebook or JupyterLab, then select:

```text
Kernel -> Change Kernel -> Python (SEAM-SDM-PYTHON)
```

Verify inside a notebook cell:

```python
import sys
import site

print(sys.executable)
print("sys.prefix:", sys.prefix)
print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
print("Paths containing .local:", [p for p in sys.path if ".local" in p])
```

Expected result:

```text
ENABLE_USER_SITE: False
Paths containing .local: []
```

The executable path should point to the `SEAM-SDM-PYTHON` conda environment.

---

## 6. Run SEAM-SDM Python scripts from terminal

All SEAM-SDM `.py` scripts should be executed from the project root using the same environment setup.

Before training, set `trainer_conf.devices` in `DeepSDM_conf.yaml` to the number
of GPUs you intend to use. The default is `4`; use `1` for a single GPU. This
number must not exceed the GPUs available to the current environment.

```bash
cd <PROJECT_DIR>
conda activate SEAM-SDM-PYTHON
export PYTHONNOUSERSITE=1
unset PYTHONPATH

python <script_name>.py
```

Example:

```bash
python 02_train_deepsdm.py
```