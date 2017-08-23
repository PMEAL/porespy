import porespy as ps
import scipy as sp
import git


class PackageTest():
    def setup_class(self):
        pass

    def test_version_number_and_git_tag_agree(self):
        repo = git.Repo(search_parent_directories=True)
        tag = repo.git.describe("--tags")
        tag = tag.strip('vV')  # Remove 'v' or 'V' from tag if present
        tag = tag.split('-')[0]  # Remove hash from tag number if present
        assert ps.__version__ == tag
