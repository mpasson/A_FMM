from pathlib import Path
from hatch_vcs.version_source import VCSVersionSource
import toml

basedir = Path(__file__).parent.parent
config = toml.load(basedir/"pyproject.toml")
vcs_version = VCSVersionSource(basedir, config["tool"]["hatch"]["version"])
__version__ = vcs_version.get_version_data()["version"]
