import torch

# Define power wavelet
def powerWavelet(M: int, s: float):
    '''Wavelet corresponding to power (dE/dt) in a primitive movement

    :param M: length (in samples) of the wavelet
    :param s: width (in samples) of the wavelet
    '''
    x = (torch.arange(M) - (M - 1) / 2) / s
    wavelet = (x > -0.5) * (x < 0.5) * 40 / 3 * x * (4 * x ** 2 - 1) ** 3
    output = torch.sqrt(1 / s) * wavelet
    return output

# Define acceleration wavelet
def accelerationWavelet(M: int, s: float):
    '''Wavelet corresponding to acceleration in a primitive movement

    :param M: length (in samples) of the wavelet
    :param s: width (in samples) of the wavelet
    '''
    x = (torch.arange(M) - (M - 1) / 2) / s
    wavelet = (x > -0.5) * (x < 0.5) * 29 / 4 * x * (4 * x ** 2 - 1)
    output = torch.sqrt(1 / s) * wavelet
    return output

# Define velocity wavelet
def velocityWavelet(M: int, s: float):
    '''Wavelet corresponding to velocity in a primitive movement

    :param M: length (in samples) of the wavelet
    :param s: width (in samples) of the wavelet
    '''
    x = (torch.arange(M) - (M - 1) / 2) / s
    wavelet = (x > -0.5) * (x < 0.5) * 11 / 7 * (4 * x ** 2 - 1) ** 2
    output = torch.sqrt(1 / s) * wavelet
    return output