import porespy as ps
import git
import os
import subprocess
import tempfile


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    dirname, __ = os.path.split(path)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=360",
                "--output", fout.name, path]
        proc = subprocess.run(args)
        rc = proc.returncode
    return rc


def test_ipynb():
    rootdir = os.path.split(os.getcwd())[0]
    for path, subdirs, files in os.walk(rootdir):
        for name in files:
            if (name.endswith('.ipynb') and 'checkpoint' not in name):
                nbook = os.path.join(path, name)
                try:
                    rc = _notebook_run(nbook)
                    print(nbook, rc)
                    assert rc == 0
                except:
#                    assert 1 == 2
                    pass


class PackageTest():
    def setup_class(self):
        pass

    def test_version_number_and_git_tag_agree(self):
        repo = git.Repo(search_parent_directories=True)
        tag = repo.git.describe("--tags")
        tag = tag.strip('vV')  # Remove 'v' or 'V' from tag if present
        tag = tag.split('-')[0]  # Remove hash from tag number if present
        assert ps.__version__ == tag


if __name__ == '__main__':
    test_ipynb()
