"""
Shared configuration loading for the Libera ground-calibration analysis.

This exists only to hold load_config, which every other module in the repo
needs. It was previously duplicated verbatim in four modules; the notebooks
reach it through different ones (erf.load_config and pst.load_config), so a
change had to be made in four places to take effect everywhere. Kept as its
own module rather than living in one of the analysis modules because the
import graph has two independent roots - libera_telescope_pst and
pbrr_conversions import nothing else local - so there was no existing module
all four could share without inventing a dependency that made no sense.
"""

import tomllib
from pathlib import Path
from types import SimpleNamespace


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


def load_config(path: Path = None) -> SimpleNamespace:
    """
    Reads config.toml into a namespace of Paths. Copy config.example.toml to
    config.toml and edit the paths for your machine.

    Nothing here depends on the process working directory, so the notebooks and
    modules work whatever directory they are run from:

      - The config file itself defaults to the one sitting beside this module,
        i.e. at the repository root, rather than to "config.toml" relative to
        the working directory. Pass path= to use a different one.
      - A RELATIVE path inside config.toml resolves against the directory
        holding that config file, so the shipped defaults of figures/, data/
        and conversions_and_calibrations/ mean those directories in the
        repository, not wherever the process happens to be.
      - An ABSOLUTE path is used unchanged (pathlib's "/" discards the left
        operand when the right side is absolute), and a leading ~ expands, so
        raw-data directories can point anywhere.

    figure_dir is created here if it does not exist, because it defaults to
    figures/ inside the repository and git cannot track an empty directory - a
    fresh clone therefore has no such directory, and the first savefig would
    fail. Doing it on load also covers a reader who points figure_dir somewhere
    else entirely. The other paths are deliberately NOT created: analysis_dir
    ships populated, and the raw-data directories are meant to be absent when
    the raw data is, so silently creating an empty one would turn a
    misconfigured path into a confusing downstream failure instead of an
    obvious one, and would defeat raw_data_available in erf_analysis, which
    decides by testing whether those directories exist.
    """

    path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    base = path.resolve().parent
    paths = SimpleNamespace(**{k: base/Path(v).expanduser()
                               for k, v in cfg["paths"].items()})

    if hasattr(paths, "figure_dir"):
        paths.figure_dir.mkdir(parents=True, exist_ok=True)

    return paths
