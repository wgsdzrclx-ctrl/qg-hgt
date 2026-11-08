"""TEP data loading and controlled corruption utilities.

Import concrete objects from :mod:`data.tep_dataset` or
:mod:`data.corruptions`; keeping package import light also lets corruption
self-checks run before PyG is installed.
"""

# Design note: data modules implement paper Sec. 4.1--4.4 and remain separate
# from model code so every baseline receives identical windows and protocols.
