import sys
import pytest

args = ['-v']

if 'pep8' in sys.argv:
    args.append('--pep8')

if 'cov' in sys.argv:
    args.extend(['--cov-report=term-missing',
                 '--cov=porespy',
                 '--cov-config=.coveragerc'])

exit_code = pytest.main(args)
exit(exit_code)
