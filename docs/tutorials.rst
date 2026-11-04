Tutorial
========

A self-contained tutorial is available as a Jupyter notebook in the repository.

* `demo.ipynb <https://github.com/VikVador/albus/blob/main/notebook/demo.ipynb>`_ loads a sample dataset, a 7-day, 4-member forecast of oxygen in the Black Sea at the surface and at 9 m depth along with the corresponding ground truth, then demonstrates every metric in the library: errors, skill, spread and their ratio, CRPS and power spectra in :mod:`albus.metrics`, and hypoxia detection scores with their ROC and PR curves in :mod:`albus.ocean.oxygen`.

The sample data live in `notebook/data <https://github.com/VikVador/albus/tree/main/notebook/data>`_, so the notebook runs from the root of a local clone of the repository.
