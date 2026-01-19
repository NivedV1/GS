import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Grid & parameters
# ---------------------------------
N = 512
x = np.linspace(-1, 1, N)
iterations = 1
alpha = 0.7          # WGS weight
beta = 15.0          # strength of exponential phase guess

# ---------------------------------
# True phase (unknown in practice)
# ---------------------------------
true_phase = np.sin(5 * np.pi * x)

# Constant amplitude (phase-only object)
amplitude = np.ones_like(x)
u_true = amplitude * np.exp(1j * true_phase)

# Known magnitudes
amp_real = np.abs(u_true)
amp_fourier_target = np.abs(np.fft.fft(u_true))

# ---------------------------------
# Exponential initial phase guess
# ---------------------------------
initial_phase = beta * x**2
field = amp_real * np.exp(1j * initial_phase)
# ---------------------------------
# Weighted Gerchberg–Saxton loop
# ---------------------------------
for _ in range(iterations):
    # Forward FFT
    U = np.fft.fft(field)

    # Weighted Fourier magnitude update (WGS)
    amp_fourier_current = np.abs(U)
    amp_fourier_new = (
        (1 - alpha) * amp_fourier_current
        + alpha * amp_fourier_target
    )

    U = amp_fourier_new * np.exp(1j * np.angle(U))

    # Back to real space
    field = np.fft.ifft(U)

    # Enforce real-space amplitude
    field = amp_real * np.exp(1j * np.angle(field))

# ---------------------------------
# Retrieved phase
# ---------------------------------
retrieved_phase = np.unwrap(np.angle(field))

# ---------------------------------
# Plot (phases only, separated)
# ---------------------------------
plt.figure(figsize=(10, 7))

# True phase
plt.subplot(3, 1, 1)
plt.plot(x, true_phase)
plt.title("True Phase")
plt.ylabel("rad")

# Initial guess
plt.subplot(3, 1, 2)
plt.plot(x, np.unwrap(initial_phase))
plt.title("Initial Random Phase Guess")
plt.ylabel("rad")

# Retrieved phase
plt.subplot(3, 1, 3)
plt.plot(x, retrieved_phase)
plt.title(f"Retrieved Phase (WGS, α = {alpha}, {iterations} iterations)")
plt.xlabel("x")
plt.ylabel("rad")

plt.tight_layout()
plt.show()
