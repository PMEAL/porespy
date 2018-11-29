import porespy as ps
import git
import os
import subprocess
import logging


def run_shell_command(command_line_args):
    try:
        command_line_process = subprocess.Popen(
            command_line_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        process_output, process_error = command_line_process.communicate()
        print(process_error)
    except (OSError, subprocess.CalledProcessError) as exception:
        logging.info('Exception occured: ' + str(exception))
        logging.info('Subprocess failed')
        return False
    else:
        # no exception was raised
        logging.info('Subprocess finished')
    return "Error" not in str(process_error)


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    dirname, __ = os.path.split(path)
    args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=360",
            "--output", "temp_output.ipynb", path]
    rc = run_shell_command(args)
    print(path, rc)
    print('-'*30)
    if rc:
        os.remove(os.path.join(dirname, "temp_output.ipynb"))
    return rc


def test_ipynb():
    rootdir = os.path.split(os.getcwd())[0]
    for path, subdirs, files in os.walk(rootdir):
        for name in files:
            if (name.endswith('.ipynb') and 'checkpoint' not in name):
                nbook = os.path.join(path, name)
                rc = _notebook_run(nbook)
                assert rc


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
