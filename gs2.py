import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Grid & parameters (unchanged)
# ---------------------------------
N = 512
x = np.linspace(-1, 1, N)
iterations = 2
beta = 15.0        # strength of exponential phase guess

# ---------------------------------
# True phase (unknown in practice)
# ---------------------------------
true_phase = np.sin(5 * np.pi * x)

# Constant amplitude (phase-only object)
amplitude = np.ones_like(x)
u_true = amplitude * np.exp(1j * true_phase)

# Known magnitudes
amp_real = np.abs(u_true)
amp_fourier = np.abs(np.fft.fft(u_true))

# ---------------------------------
# Exponential initial phase guess
# ---------------------------------
initial_phase = beta * x**2
field = amp_real * np.exp(1j * initial_phase)

# ---------------------------------
# Gerchberg–Saxton loop (GS)
# ---------------------------------
for _ in range(iterations):
    # Forward FFT
    U = np.fft.fft(field)

    # Enforce Fourier magnitude (pure GS)
    U = amp_fourier * np.exp(1j * np.angle(U))

    # Back to real space
    field = np.fft.ifft(U)

    # Enforce real-space amplitude
    field = amp_real * np.exp(1j * np.angle(field))

# ---------------------------------
# Retrieved phase
# ---------------------------------
retrieved_phase = np.unwrap(np.angle(field))

# ---------------------------------
# Plot (phases only, same layout)
# ---------------------------------
plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
plt.plot(x, true_phase)
plt.title("True Phase")
plt.ylabel("rad")

plt.subplot(3, 1, 2)
plt.plot(x, initial_phase)
plt.title(f"Initial Exponential Phase Guess (β = {beta})")
plt.ylabel("rad")

plt.subplot(3, 1, 3)
plt.plot(x, retrieved_phase)
plt.title(f"Retrieved Phase (GS, {iterations} iterations)")
plt.xlabel("x")
plt.ylabel("rad")

plt.tight_layout()
plt.show()
