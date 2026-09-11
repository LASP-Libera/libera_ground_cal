# Libera Ground Calibration

Ground calibration analyses for the Libera instrument. Libera is the 
follow-on for the CERES instruments, and will continue broadband 
radiometric measurements of Earth outgoing radiation.

Each subdirectory is an **independent project** with its own Python
environment, configuration and documentation. They are collected in one
repository because from a system perspective they were all part of the same
ground calibration campaign, not because they share code.

| Directory | Contents |
|---|---|
| [`radiometers/`](radiometers/) | Science radiometer and LST calibration in the ERF: absolute spectral response functions with a propagated uncertainty budget, plus the out-of-field (scattered light) response model. |

## Working in a subproject

Everything is run from inside the subdirectory, not from this level:

```bash
cd radiometers
cp config.example.toml config.toml   # then edit the paths for your machine
poetry install
```

See that subdirectory's own README for setup, what the notebooks cover, and
its known limitations.

## Status

This is work in progress, published early so that interested groups can follow
the analysis rather than waiting for it to be finished. Each subproject's
README carries its own list of known limitations; read that before relying on
any numbers.

## License

BSD 3-Clause. See [LICENSE.txt](LICENSE.txt).
