import porespy as ps
import scipy as sp
import git

class PackageTest():
    def setup_class(self):
        pass

    def test_randomize_colors(self):
        repo = git.Repo(search_parent_directories=True)
        assert pn.__version__ == repo.git.describe("--tags").strip('vV')
