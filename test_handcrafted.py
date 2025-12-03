"""
Test if discrimination is POSSIBLE by hand-crafting extreme parameters
"""

import sys
sys.path.insert(0, '.')

import jax.numpy as jnp
from src.model import build_motion_detector
from src.stimulus import create_1d_motion
from src.simulate import simulate_sequence, count_spikes

# Build model
cell = build_motion_detector()

# Manually set EXTREME spatial gradient
print("Testing with EXTREME hand-crafted parameters...")
print("="*60)

# Test 1: Very strong left-right gradient
params_extreme = []
gNa_extreme = [0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12, 0.09, 0.06, 0.03]
gK_extreme = [0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06]

for i in range(10):
    params_extreme.append({'HH_gNa': jnp.array([gNa_extreme[i]])})
for i in range(10):
    params_extreme.append({'HH_gK': jnp.array([gK_extreme[i]])})

print(f"\nExtreme gNa gradient: {gNa_extreme[0]:.2f} → {gNa_extreme[-1]:.2f} ({gNa_extreme[0]/gNa_extreme[-1]:.1f}×)")

right_motion = create_1d_motion('right', n_frames=5)
left_motion = create_1d_motion('left', n_frames=5)

v_r, _ = simulate_sequence(cell, right_motion, params=params_extreme, verbose=False)
v_l, _ = simulate_sequence(cell, left_motion, params=params_extreme, verbose=False)

spikes_r = count_spikes(v_r[0])
spikes_l = count_spikes(v_l[0])

print(f"\nResults:")
print(f"  Right motion: {spikes_r} spikes")
print(f"  Left motion:  {spikes_l} spikes")
print(f"  Difference:   {spikes_r - spikes_l} spikes")

if abs(spikes_r - spikes_l) >= 5:
    print("\n✓ DISCRIMINATION IS POSSIBLE with extreme parameters!")
    print("  → The optimization just can't find this solution")
elif abs(spikes_r - spikes_l) >= 2:
    print("\n~ Weak discrimination is possible")
    print("  → But the signal might be too weak for gradient descent")
else:
    print("\n✗ NO DISCRIMINATION even with extreme parameters")
    print("  → The task might be impossible with this architecture")
    print("  → Consider: different stimulus, network instead of single cell, or additional mechanisms")