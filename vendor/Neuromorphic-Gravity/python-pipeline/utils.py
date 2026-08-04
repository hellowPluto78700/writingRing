import torch
import scipy.signal as signal

def convm(h, p):
    """
    Generates a convolution matrix
    
    Usage: H = convm(h, p)
    Given a vector h of length N, an N+p-1 by p convolution matrix is
    generated of the following form:
              |  h(0)  0      0     ...      0    |
              |  h(1) h(0)    0     ...      0    |
              |  h(2) h(1)   h(0)   ...      0    |
         H =  |   .    .      .              .    |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |  h(N) h(N-1) h(N-2) ...  h(N-p+1) |
              |   0   h(N)   h(N-1) ...  h(N-p+2) |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |   0    0      0     ...    h(N)   |
         
    That is, h is assumed to be causal, and zero-valued after N.
    """
    N = len(h) + 2 * p - 2
    hpad = torch.cat([torch.zeros(p - 1), h, torch.zeros(p - 1)])
    H = torch.zeros((len(h) + p - 1, p))
    # Construct H column by column
    for i in range(p):
        H[:,i] = hpad[p-i-1:N-i]
    
    return H

def prony(h, p, q):
    """
    Model a signal using Prony's method
 
    Usage: [b, a, err] = prony(h, p, q)
 
    The input sequence h is modeled as the unit sample response of
    a filter having a system function of the form
        H(z) = B(z) / A(z) 
    The polynomials B(z) and A(z) are formed from the vectors
        b = [b(0), b(1), ... b(q)]
        a = [1   , a(1), ... a(p)]
    The input q defines the number of zeros in the model
    and p defines the number of poles. The modeling error is 
    returned in err.
 
    This comes from Hayes, p. 149, 153, etc
    """
    N = len(h)
    assert p + q < N, 'Model order too large'
 
    # This formulation uses eq. 4.50, p. 153
    # Set up the convolution matrices
    H = convm(h, p + 1)
    Hq = H[q:N+p-1,0:p]
    hq1 = - H[q+1:N+p,0]
 
    # Solve for denominator coefficients
    if p > 0:
        a = torch.linalg.lstsq(Hq, hq1).solution
        a = torch.cat([torch.tensor([1]), a]) # a(0) is 1
    else:
        # all-zero model
        a = torch.tensor([1])
 
    # Solve for the model error
    err = H[q+1:N,0:p+1].T @ torch.conj(h[q+1:N]) @ a
 
    # Solve for numerator coefficients
    if q > 0:
        # (This is the same as for Pad?)
        b = H[0:q+1,0:p+1] @ a
    else:
        # all-pole model
        # b(0) is h(0), but a better solution is to match energy
        b = torch.sqrt(err)
 
    return b, a

def butter(N, Wn, **kwargs):
    b, a = signal.butter(N, Wn, **kwargs)
    b, a = torch.Tensor(b), torch.Tensor(a)
    return b, a