"""Python implementation the Aperiodic-Fourier Modal Method. 
A fully vectorial method for solving Maxwell equations that combines a Fourier-based mode solver and a scattering matrix recursion algorithm to model full 3D structures.
This approach is well suited to calculate modes, transmission, reflection, scattering and absorption of multi-layered structures.
Moreover, support for Bloch modes of periodic structures allows for the simulation of photonic crystals or waveguide Bragg gratings.
"""
import os


from A_FMM.layer import *
from A_FMM.scattering import S_matrix
from A_FMM.creator import Creator
from A_FMM.stack import Stack
import A_FMM.inputs as inputs

try:
    from ._version import __version__
except ModuleNotFoundError:
    from hatch_vcs.version_source import VCSVersionSource
    import toml

    basedir = os.path.split(os.path.dirname(__file__))[0]
    config = toml.load(os.path.join(basedir, "pyproject.toml"))
    vcs_version = VCSVersionSource(basedir, config["tool"]["hatch"]["version"])
    __version__ = vcs_version.get_version_data()["version"]
