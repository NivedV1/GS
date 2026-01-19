import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
N = 1500                  # Number of points
iterations = 500       # GS iterations
x = np.linspace(-1, 1, N)

# -------------------------------
# Ground-truth signal (unknown phase)
# -------------------------------
true_phase = np.sin(5 * np.pi * x)
true_amplitude = np.exp(-x**2 * 20)

u_true = true_amplitude * np.exp(1j * true_phase)

# Known magnitudes
spatial_mag = np.abs(u_true)
fourier_mag = np.abs(np.fft.fft(u_true))

# -------------------------------
# Initial guess (random phase)
# -------------------------------
phase_guess = np.exp(1j * 2 * np.pi * np.random.rand(N))
u = spatial_mag * phase_guess

# -------------------------------
# Gerchberg–Saxton loop
# -------------------------------
for _ in range(iterations):

    # Forward FFT
    U = np.fft.fft(u)

    # Enforce Fourier magnitude constraint
    U = fourier_mag * np.exp(1j * np.angle(U))

    # Inverse FFT
    u = np.fft.ifft(U)

    # Enforce spatial magnitude constraint
    u = spatial_mag * np.exp(1j * np.angle(u))

# -------------------------------
# Results
# -------------------------------
retrieved_phase = np.unwrap(np.angle(u))

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(x, true_phase, label="True phase")
plt.plot(x, retrieved_phase, "--", label="Retrieved phase")
plt.legend()
plt.title("Phase Retrieval using 1D Gerchberg–Saxton")

plt.subplot(2, 1, 2)
plt.plot(x, spatial_mag)
plt.title("Known Spatial Magnitude")

plt.text(
    0.02, 0.95,
    f"Iterations = {iterations}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top'
)

plt.tight_layout()
plt.show()
