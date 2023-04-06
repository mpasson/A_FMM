import os
import setuptools_git_versioning

__version__ = setuptools_git_versioning.get_version(
    root=os.path.join(*os.path.split(os.path.dirname(__file__))[:-1])
)
__version__ = str(__version__)
