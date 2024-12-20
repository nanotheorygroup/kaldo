"""
Dirac delta function implementations for different broadening shapes
"""
import torch


def triangular_delta(delta_omega, sigma):
    """
    Triangular delta function implementation for PyTorch tensors
    """
    delta_omega = torch.abs(delta_omega)
    deltaa = torch.abs(sigma)
    mask = delta_omega < deltaa
    out = torch.zeros_like(delta_omega)
    out[mask] = (1. / deltaa * (1 - delta_omega[mask] / deltaa))
    return out


def gaussian_delta(delta_omega, sigma):
    """
    Gaussian delta function implementation for PyTorch tensors
    """
    return 1. / (sigma * torch.sqrt(torch.tensor(2 * torch.pi))) * \
           torch.exp(-delta_omega ** 2 / (2 * sigma ** 2))


def lorentz_delta(delta_omega, sigma):
    """
    Lorentzian delta function implementation for PyTorch tensors
    """
    return 1. / torch.pi * sigma / (delta_omega ** 2 + sigma ** 2)
