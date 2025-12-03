#!/usr/bin/env python3
"""Minimal JAXLEY neuron test"""

import jaxley as jx
from jaxley.channels import HH
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Create single compartment neuron
print("Creating neuron...")
comp = jx.Compartment()
cell = jx.Cell([comp], parents=[-1])

# Insert Hodgkin-Huxley channels
cell.insert(HH())
print(f"Neuron has {len(cell.nodes)} compartments")

# Create stimulus (step current)
print("Creating stimulus...")
current = jx.step_current(
    i_delay=1.0,    # Start at 1ms
    i_dur=5.0,      # Duration 5ms
    i_amp=0.1,      # Amplitude 0.1 nA
    delta_t=0.025,  # Time step
    t_max=20.0      # Total time
)

# Stimulate neuron
cell.stimulate(current)
cell.record("v")  # Record voltage

# Simulate
print("Running simulation...")
voltages = jx.integrate(cell, delta_t=0.025)

# Plot
print("Plotting...")
time = jnp.arange(len(voltages)) * 0.025  # Convert to ms

plt.figure(figsize=(10, 4))
plt.plot(time, voltages)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Single Compartment Neuron Response')
plt.grid(True, alpha=0.3)
plt.axhline(y=-70, color='k', linestyle='--', alpha=0.3, label='Resting potential')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Threshold (~0 mV)')
plt.legend()
plt.tight_layout()
plt.savefig('test_neuron_response.png', dpi=150)
print("Saved: test_neuron_response.png")
plt.show()

print(f"\n✓ Simulation complete!")
print(f"  Min voltage: {jnp.min(voltages):.1f} mV")
print(f"  Max voltage: {jnp.max(voltages):.1f} mV")
print(f"  Spike detected: {jnp.max(voltages) > 0}")