from setuptools import setup

setup(name='A_FMM',
      version='0.1',
      description="Aperiodic-Fourier Modal Method with Python",
      long_description="",
      author='Marco Passoni',
      author_email='mpasson91@egmail.com',
      license='TODO',
      packages=['A_FMM'],
      zip_safe=False,
      install_requires=[
            "numpy",
            "scipy",
            "matplotlib",
            "pandas",
          ],
      )
