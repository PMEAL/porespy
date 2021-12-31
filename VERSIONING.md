## Release Management and Versioning

`PoreSpy` uses [Semantic Versioning](http://semver.org) (i.e. X.Y.Z) to label releases.  All versions of `PoreSpy` since "0.3.4" are available on [PyPI](https://pypi.python.org/pypi).  Prior to this, only major and minor version were pushed.

All development occurs on `dev` via feature branches and the pull request functionality of Github. A new release is defined each time the `dev` branch is merged into the `release` branch. Several automations are setup so that upon each release, the code is automatically deployed to PyPi and Conda, and a release announcement is created on Github containing a summary of all the changes.

`PoreSpy` depends on other packages including [Scipy](https://scipy.org/) and its dependencies.  It is our policy to always support the latest version of all these packages and their dependencies.
