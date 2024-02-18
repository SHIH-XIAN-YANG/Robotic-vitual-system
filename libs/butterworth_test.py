import numpy as np
import matplotlib.pyplot as plt

# Define the cutoff frequency (adjust as needed)
cutoff_frequency = 1000  # 1 kHz

# Create a function to simulate the 4th-order Butterworth filter
def butterworth_filter_4th_order(input_signal, sampling_frequency, cutoff_frequency):
    # Calculate the time constant (tau) based on the cutoff frequency
    tau = 1 / (2 * np.pi * cutoff_frequency)
    
    dt = 1 / sampling_frequency
    alpha = tau / (2 * tau + dt)
    
    # Initialize states for each of the four stages
    state1 = 0
    state2 = 0
    state3 = 0
    state4 = 0
    
    output_signal = []

    for x in input_signal:
        state1 += alpha * (x - state1)
        state2 += alpha * (state1 - state2)
        state3 += alpha * (state2 - state3)
        state4 += alpha * (state3 - state4)
        
        output_signal.append(state4)
    
    return output_signal

# Parameters
sampling_frequency = 10000  # Sampling frequency in Hz
duration = 0.2  # Duration of the signal in seconds
frequency_sine = 10  # Frequency of the sine wave in Hz
amplitude_sine = 1.0  # Amplitude of the sine wave
amplitude_noise = 0.5  # Amplitude of the noise
frequency_noise = 2000  # Frequency of the noise in Hz

# Time vector
t = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)

# Generate the sine wave component
sine_wave = amplitude_sine * np.sin(2 * np.pi * frequency_sine * t)

# Generate the noise component
noise = amplitude_noise * np.random.randn(len(t))
noise *= np.sin(2 * np.pi * frequency_noise * t)  # Add noise at 2 kHz

# Combine the sine wave and noise
signal = sine_wave + noise


# Apply the 4th-order Butterworth filter to the input signal
output_signal = butterworth_filter_4th_order(signal, sampling_frequency, cutoff_frequency)

# Plot the input and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Input Signal')
plt.plot(t, output_signal, label='Filtered Signal')
plt.legend()
plt.xlabel('Time (s)')
plt.title('4th-Order Butterworth Filter')
plt.grid()
plt.show()
