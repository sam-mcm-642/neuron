"""
Test multi-compartment readout with the same failed calcium model
"""

import sys
sys.path.insert(0, '.')

import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import HH
from jaxley.channels.pospischil import CaL

from src.stimulus import create_1d_motion
from src.simulate import simulate_sequence, count_spikes


print("="*80)
print("FINAL TEST: Multi-Compartment Readout")
print("="*80)
print("\nWe've proven that soma readout fails (9 vs 9 spikes)")
print("Now testing if multi-compartment readout succeeds...")

# Build same model as before
cell = jx.Cell([jx.Branch(jx.Compartment(), ncomp=10)], parents=[-1])
cell.insert(HH())
cell.insert(CaL())

# Same spatial gradient
gCaL_extreme = [0.0009, 0.0008, 0.0007, 0.0006, 0.0005,
                0.0004, 0.0003, 0.0002, 0.0001, 0.00003]

for comp_idx in range(10):
    cell.branch(0).comp(comp_idx).make_trainable(
        "CaL_gCaL",
        init_val=gCaL_extreme[comp_idx],
        verbose=False
    )

params = cell.get_parameters()

# Same stimuli
right_motion = create_1d_motion('right', n_frames=5)
left_motion = create_1d_motion('left', n_frames=5)

# Simulate
v_right, _ = simulate_sequence(cell, right_motion, params=params, verbose=False)
v_left, _ = simulate_sequence(cell, left_motion, params=params, verbose=False)

print("\n" + "="*80)
print("METHOD 1: Soma Readout (Already Failed)")
print("="*80)

soma_spikes_right = count_spikes(v_right[0])
soma_spikes_left = count_spikes(v_left[0])

print(f"Right: {soma_spikes_right} spikes")
print(f"Left:  {soma_spikes_left} spikes")
print(f"Difference: {soma_spikes_right - soma_spikes_left} spikes")
print("Result: ✗ NO DISCRIMINATION")

print("\n" + "="*80)
print("METHOD 2: Two-Point Readout")
print("="*80)

# Read from left dendrite (comp 2) and right dendrite (comp 7)
left_dendrite_activity_r = float(jnp.var(v_right[2]))
right_dendrite_activity_r = float(jnp.var(v_right[7]))

left_dendrite_activity_l = float(jnp.var(v_left[2]))
right_dendrite_activity_l = float(jnp.var(v_left[7]))

# Opponent signal: left - right
opponent_signal_right = left_dendrite_activity_r - right_dendrite_activity_r
opponent_signal_left = left_dendrite_activity_l - right_dendrite_activity_l

print(f"\nRight motion:")
print(f"  Left dendrite (comp 2) activity:  {left_dendrite_activity_r:.1f}")
print(f"  Right dendrite (comp 7) activity: {right_dendrite_activity_r:.1f}")
print(f"  Opponent signal (L-R):            {opponent_signal_right:+.1f}")

print(f"\nLeft motion:")
print(f"  Left dendrite (comp 2) activity:  {left_dendrite_activity_l:.1f}")
print(f"  Right dendrite (comp 7) activity: {right_dendrite_activity_l:.1f}")
print(f"  Opponent signal (L-R):            {opponent_signal_left:+.1f}")

print(f"\nDiscrimination:")
print(f"  Difference in opponent signal: {abs(opponent_signal_right - opponent_signal_left):.1f}")

if abs(opponent_signal_right - opponent_signal_left) > 50:
    print("\n✓✓✓ SUCCESS: Multi-compartment readout enables discrimination!")
    print("    The spatial pattern of dendritic activity differs between directions")
    print("    Reading from two points preserves this information")
elif abs(opponent_signal_right - opponent_signal_left) > 20:
    print("\n~ WEAK: Some discrimination but signal is weak")
else:
    print("\n✗ FAILURE: Even multi-compartment readout fails")

print("\n" + "="*80)
print("METHOD 3: Three-Region Readout")
print("="*80)

# Read from three spatial regions
def three_region_readout(v):
    left = float(jnp.mean(jnp.array([jnp.var(v[i]) for i in [0,1,2]])))
    mid = float(jnp.mean(jnp.array([jnp.var(v[i]) for i in [4,5]])))
    right = float(jnp.mean(jnp.array([jnp.var(v[i]) for i in [7,8,9]])))
    return left, mid, right

left_r, mid_r, right_r = three_region_readout(v_right)
left_l, mid_l, right_l = three_region_readout(v_left)

print(f"\nRight motion: [Left={left_r:.1f}, Mid={mid_r:.1f}, Right={right_r:.1f}]")
print(f"Left motion:  [Left={left_l:.1f}, Mid={mid_l:.1f}, Right={right_l:.1f}]")

pattern_distance = float(jnp.sqrt((left_r-left_l)**2 + (mid_r-mid_l)**2 + (right_r-right_l)**2))
print(f"\nSpatial pattern distance: {pattern_distance:.1f}")

if pattern_distance > 100:
    print("\n✓✓✓ SUCCESS: Three-region readout enables strong discrimination!")
elif pattern_distance > 50:
    print("\n~ WEAK: Some discrimination")
else:
    print("\n✗ FAILURE: Patterns too similar")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("\nWhat we've learned:")
print("1. ✗ Single neuron + soma readout + calcium = FAILS (9 vs 9 spikes)")
print("2. ? Single neuron + multi-point readout + calcium = Testing now...")

if abs(opponent_signal_right - opponent_signal_left) > 50 or pattern_distance > 100:
    print("\n✓ MULTI-COMPARTMENT READOUT IS THE SOLUTION")
    print("\nNext step: Train with gradient descent using multi-compartment loss")
else:
    print("\n✗ Even multi-compartment readout insufficient")
    print("   May need: opponent neurons, inhibition, or network architecture")

print("\n" + "="*80)