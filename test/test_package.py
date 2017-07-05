import porespy as ps
import scipy as sp
import git

class TestPackage():
    def setup_class(self):
        pass

    def test_randomize_colors(self):
        repo = git.Repo(search_parent_directories=True)
        assert ps.__version__ == repo.git.describe("--tags").strip('vV')
