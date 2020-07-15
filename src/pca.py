import numpy as np

class PCA(object):
    def __init__(self, nc=1, nr=1, l=1):
        self.l = l
        self.nc = nc
        self.nr = nr
        self.xm = np.zeros((nc, 1))
        self.usig = np.zeros((nc, l))
        self.data_matrix = None
        self.sig = None
        self.u = None

    def input_usig(self, usig=None, u=None, sig=None):
        if usig is not None:
            assert usig.shape == (self.nc, self.l)
            self.usig = usig
        elif u is not None and sig is not None:
            assert u.shape == (self.nc, self.l)
            assert sig.shape[0] == self.l
            if sig.shape == (self.l, ):
                sig = sig[:,None]
            self.u = u
            self.sig = sig
            self.usig = np.dot(self.u, np.diag(self.sig[:,0]))

    def input_xm(self, xm):
        assert xm.shape[0] == self.nc
        if xm.shape == (self.nc, ):
            xm = xm[:, None]
        self.xm = xm

    def construct_pca(self, x):
        assert x.shape == (self.nc, self.nr)
        self.data_matrix = x
        self.xm = np.mean(x, axis=1)[:, None]
        y = 1./(np.sqrt(float(self.nr - 1.))) * (x - self.xm)
        self.u, self.sig, _ = np.linalg.svd(y, full_matrices=False)
        self.u = self.u[:, :self.l]
        self.sig = self.sig[:self.l, None]
        self.usig = np.dot(self.u, np.diag(self.sig[:,0]))

    def generate_pca_realization(self, xi, dim=None):
        if dim is None:
            assert xi.shape[0] == self.l
            if xi.shape == (self.l, ):
                xi = xi[:,None]
            return self.usig.dot(xi) + self.xm
        else:
            assert xi.shape[0] == dim
            if xi.shape == (dim, ):
                xi = xi[:, None]
            return self.usig[:, :dim].dot(xi) + self.xm

    def get_xi(self, m, dim=None):
        assert self.u is not None, "Input or calculate U matrix to obtain reconstructed xi"
        assert m.shape[0] == self.nc
        
        if m.shape == (self.nc, ):
            m = m[:,None]
        if dim is None:
            xi = self.u.T.dot(m - self.xm)/self.sig
        else:
            xi = self.u[:, :dim].T.dot(m - self.xm) / self.sig[:dim]
        return xi


