import numpy as np

def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x))) / np.sqrt(np.prod(x.shape[-2:]))

def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x))) * np.sqrt(np.prod(x.shape[-2:]))


class CartesianSENSE:
    def __init__(self, coil_sensitivities, mask):
        
        self.C = coil_sensitivities
        self.mask = mask
        self.shape = coil_sensitivities.shape[:2]
        self.ncoils = coil_sensitivities.shape[2]

    # TO DO
    def mtimes(self, x, adjoint=False):
        if adjoint:
            # adjoint: E^H*y = C*F^-1(U*y_c)
            # should return [Nx, Ny] reconstructed images
            res = np.zeros(SHAPE, dtype=np.complex64) # --> 0. TO DO: put in right SHAPE
            for c in range(self.ncoils):
                # 1. TO DO: apply sampling mask (U * y_c)

                # 2. TO DO: apply inverse Fourier transform

                # 3. TO DO: apply conjugate coil sensitivity (conj(C) * img_c)

                pass
        
        else:
            # forward: E*x = U*F(C*x)
            # should return [Nx, Ny, Ncoils] masked k-space
            # Forward: E*x = U * F(C * x)
            res = np.zeros(SHAPE, dtype=np.complex64) # --> 0. TO DO: put in right SHAPE
            for c in range(self.ncoils):
                # 1. TO DO: apply coil sensitivity map (C * x)

                # 2. TO DO: apply Fourier transform

                # 3. TO DO: apply sampling mask (U * k-space)

                pass
            return res
        
    def __matmul__(self, x):
        # E @ x for forward operation.
        return self.mtimes(x, adjoint=False)

    def H(self, x):
        # E.H(x) for adjoint operation.
        return self.mtimes(x, adjoint=True)
