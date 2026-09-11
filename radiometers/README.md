# Libera Ground Calibration

This repository walks through the analysis of the Libera ground calibration data, going from the raw data files to the absolute spectral response functions. The ground calibration included several key measurements:
- Science radiometer calibration in ERF
  - Measurements from July 2024 - May 2025
- Integrated instrument calibrations
  - August 2025, absolute blackbodies
  - November 2025, absolute blackbodies, SW tie against the LST

The notebooks checks for the raw data (see config.toml below) and if it is present it allows the data analysis to run from the raw data. If not then the analysis can still be run from the high-level files.

# Learning Notebooks

erf_analysis.ipynb - Walks through the analysis of the ERF data to generation of the absolute spectral response in the SW.

erf_scirad_scattered_light.ipynb - Walks through analysis and fitting of the out of field response of the science radiometers. This generates a correction used in generation of the absolute spectral response.

# Known Limitations

- The OSA spatial-spectral non-uniformity measurement is still needed; the IR wavelength-scale term is a conservative Type-B bound until then.
- Current per-point uncertainties appear to be understated and the SRF fit uncertainties are correspondingly optimistic.
- The SSW AMTIR-1 edge temperature correction rests on a single ΔT of 5.04 °C and one measured node above 0.80 µm; linearity in temperature is assumed. We have more measurements from IOV that we will include to help bound this.
- The Feb 2025 LW wavelength offset (due to the OSA spatial-spectral non-uniformity) rests on SW and SSW; Total's rests entirely on the other channels.
- We are planning to generate SRFs that correct for the laser linewidth. This will sharpen some features. The laser-line convolution is implemented but not enabled.
- The LST has no level_03-equivalent uncertainty budget yet.
- The SW +yaw feature's origin is unestablished and is carried as an uncertainty rather than understood.

# Repository Layout

This directory (`radiometers/`) is a self-contained project inside the
`libera_ground_cal` repository. Sibling directories hold other Libera ground
calibrations; they are grouped here because they were all part of the same
ground calibration campaign from a system perspective, but they are otherwise
independent, each with its own Python environment and configuration.

**Run everything from inside this directory.** After cloning:

```bash
cd radiometers
```

then follow the setup below. Two things depend on it:

- The analysis modules import each other by plain module name (`import
  erf_analysis`), so they only resolve with this directory on `sys.path`.
  Jupyter and VS Code put a notebook's own directory there automatically, so
  opening either notebook from here works; running a script from the
  repository root does not.
- `pytest.ini` lives here, and it is what deselects the slow diffraction FFT
  test. A bare `pytest` from the repository root does not find it, silently
  runs the slow test, and warns about an unregistered `slow` marker. Run
  `pytest` from inside this directory, or `pytest radiometers/` from the root -
  both pick up the config.

Paths in `config.toml` are resolved relative to that file rather than to the
working directory, so `data/` and `figures/` always mean the ones in this
directory no matter where a process is started from.

# Local Project Setup

Using a virtual environment is strongly recommended for ensuring proper dependency management.

## Software Installation Requirements

Python Version Management

- Conda (we recommend using Miniconda)
  Python Dependency Management
- Poetry (we recommend installing from the official script not with pipx)

_Note: This setup assumes that both conda and poetry are installed and available in your PATH. We recommend installing miniconda and poetry from the install script if you haven't_

## One Time Setup steps

### Python Version Management with Conda

To begin we want to set the correct version of python (>3.11) to build our environment from.

```bash
conda create -n conda-python3.11 python=3.11
```

This creates a conda environment named `conda-python3.11` with Python 3.11 installed with
no additional packages. This environment will serve as the base interpreter for all
subsequent virtual environments.

### Poetry Configuration

Poetry is a dependency management tool for Python that simplifies package management
and virtual environment handling. Our team prefers using Poetry for managing Python projects
and configures our poetry system to create virtual environments in the project directory.

```bash
poetry config virtualenvs.in-project true
```

## Virtual Environment Setup

To set up a virtual environment for your project, follow these steps: 1. Save the path to the base conda environment's Python interpreter 2. Create a new poetry virtual environment using the base conda environment 3. Install the project dependencies using Poetry 4. Activate the virtual environment

### Linux and MacOS

Steps 1 + 2

```bash
export PATH_TO_PYTHON=$(conda run -n conda-python3.11 python -c "import sys; print(sys.executable)")
poetry env use $PATH_TO_PYTHON
```

Here you can ensure that the virtual environment was created successfully by running:

```bash
poetry env info
```

This should result in output similar to:

```
Virtualenv
   Python:         3.11.7
   Implementation: CPython
   Path:           /Users/myuser/path/to/libera_ground_cal/radiometer/.venv
   Valid:          True
```

To install the packages listed in the pyproject.toml

```bash
poetry install
```

To manually activate this virtual environment

```bash
source .venv/bin/activate
```

When a new package is added to pyproject.toml to update the dependencies run this:

```bash
poetry lock
poetry install
```

### Path Setup

Copy this example configuration file then edit to setup the correct paths for your computer. If the raw data is present then these paths should point to that data location which will allow the analysis from the raw data will be possible.

```bash
cp config.example.toml config.toml
```

