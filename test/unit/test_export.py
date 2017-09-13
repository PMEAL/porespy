import porespy as ps
import numpy as np
import pytest
import os
import sys


class ExportTest():

    def setup_class(self):
        self.path = os.path.dirname(os.path.abspath(sys.argv[0]))

    def test_export_to_palabos(self):
        X = 20
        Y = 20
        Z = 20
        S = X*Y*Z
        im = ps.generators.blobs(shape=[X, Y, Z], porosity=0.7, blobiness=1)
        tmp = os.path.join(self.path, 'palabos.dat')
        ps.export.to_palabos(im, tmp, solid=0)
        assert os.path.isfile(tmp)
        with open(tmp) as f:
            val = f.read().splitlines()
        val = np.asarray(val).astype(int)
        assert np.size(val) == S
        assert np.sum(val == 0) + np.sum(val == 1) + np.sum(val == 2) == S
        os.remove(tmp)
