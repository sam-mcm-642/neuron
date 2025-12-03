"""
Definitive test: Does calcium enable motion direction discrimination?
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
print("DEFINITIVE TEST: Can Calcium Enable Motion Discrimination?")
print("="*80)

# ==============================================================================
# Step 1: Build neuron with HH + CaL
# ==============================================================================

print("\n1. Building neuron with HH + CaL channels...")

comp = jx.Compartment()
dendrite = jx.Branch(comp, ncomp=10)
cell = jx.Cell([dendrite], parents=[-1])

cell.insert(HH())
cell.insert(CaL())

print("✓ Neuron built")
print(f"  Channels: HH (Na, K) + CaL (L-type calcium)")

# ==============================================================================
# Step 2: Make CaL conductances trainable with spatial gradient
# ==============================================================================

print("\n2. Making CaL conductances trainable with spatial gradient...")

# Extreme gradient: 30× ratio from left to right
gCaL_extreme = [
    0.0009,  # Comp 0 - LEFT (very high)
    0.0008,  # Comp 1
    0.0007,  # Comp 2
    0.0006,  # Comp 3
    0.0005,  # Comp 4
    0.0004,  # Comp 5 - MIDDLE
    0.0003,  # Comp 6
    0.0002,  # Comp 7
    0.0001,  # Comp 8
    0.00003, # Comp 9 - RIGHT (very low)
]

print(f"  gCaL gradient: {gCaL_extreme[0]:.5f} (left) → {gCaL_extreme[-1]:.5f} (right)")
print(f"  Ratio: {gCaL_extreme[0]/gCaL_extreme[-1]:.1f}×")

# Make each compartment's CaL conductance trainable with initial values
for comp_idx in range(10):
    cell.branch(0).comp(comp_idx).make_trainable(
        "CaL_gCaL",
        init_val=gCaL_extreme[comp_idx],
        verbose=False
    )

print("✓ CaL conductances made trainable")

# Get parameters
params_extreme = cell.get_parameters()
print(f"  Total parameter groups: {len(params_extreme)}")

# Verify CaL parameters are set
print("\n  Verifying CaL parameters:")
cal_count = 0
for i, p in enumerate(params_extreme):
    if 'CaL_gCaL' in p:
        print(f"    Group {i}: CaL_gCaL = {float(p['CaL_gCaL'][0]):.5f}")
        cal_count += 1

if cal_count == 10:
    print(f"  ✓ All 10 compartments have CaL conductances")
elif cal_count > 0:
    print(f"  ⚠ Only {cal_count}/10 compartments have CaL")
else:
    print(f"  ✗ No CaL parameters found!")
    sys.exit(1)

# ==============================================================================
# Step 3: Create stimuli
# ==============================================================================

print("\n3. Creating motion stimuli...")

# Overlapping motion (realistic)
right_motion = create_1d_motion('right', n_frames=5)
left_motion = create_1d_motion('left', n_frames=5)

print("  Right motion activates: [2, 3, 4, 5, 6]")
print("  Left motion activates:  [7, 6, 5, 4, 3]")
print("  Overlap: [3, 4, 5, 6]")

# Scrambled control (same compartments as right, random order)
scrambled_motion = jnp.array([
    jnp.zeros(10).at[4].set(1.0),
    jnp.zeros(10).at[6].set(1.0),
    jnp.zeros(10).at[2].set(1.0),
    jnp.zeros(10).at[5].set(1.0),
    jnp.zeros(10).at[3].set(1.0),
])

print("  Scrambled activates:    [4, 6, 2, 5, 3] (same comps, random order)")

# ==============================================================================
# Step 4: Test discrimination
# ==============================================================================

print("\n4. Running simulations...")

try:
    v_right, _ = simulate_sequence(cell, right_motion, params=params_extreme, verbose=False)
    v_left, _ = simulate_sequence(cell, left_motion, params=params_extreme, verbose=False)
    v_scrambled, _ = simulate_sequence(cell, scrambled_motion, params=params_extreme, verbose=False)
    
    print("✓ Simulations completed")
    
    # Check voltage shape
    print(f"  Voltage shape: {v_right.shape}")
    
    # Get soma voltage (first compartment)
    v_soma_right = v_right[0] if v_right.ndim > 1 else v_right
    v_soma_left = v_left[0] if v_left.ndim > 1 else v_left
    v_soma_scrambled = v_scrambled[0] if v_scrambled.ndim > 1 else v_scrambled
    
    # Count spikes
    spikes_right = count_spikes(v_soma_right)
    spikes_left = count_spikes(v_soma_left)
    spikes_scrambled = count_spikes(v_soma_scrambled)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nSpike counts at soma:")
    print(f"  Right motion:     {spikes_right} spikes")
    print(f"  Left motion:      {spikes_left} spikes")
    print(f"  Scrambled:        {spikes_scrambled} spikes")
    print(f"\n  Difference (R-L): {spikes_right - spikes_left} spikes")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    diff = abs(spikes_right - spikes_left)
    
    if diff >= 5:
        print("\n✓✓✓ SUCCESS: Calcium enables strong discrimination!")
        print(f"    Difference of {diff} spikes is clearly distinguishable")
        print("\n    Why this works:")
        print("    - CaL time constant (~100ms) integrates the 125ms sequence")
        print("    - Right motion hits high-gCaL region FIRST")
        print("    - Early strong calcium → temporal summation → many spikes")
        print("    - Left motion hits high-gCaL region LAST")
        print("    - Late strong calcium can't build up → fewer spikes")
        print("\n    Next step: Train with gradient descent to learn this pattern")
        
    elif diff >= 2:
        print(f"\n~ WEAK: Some discrimination ({diff} spikes) but may not be robust")
        print("    Calcium helps but signal is weak")
        print("    Consider:")
        print("    - Stronger CaL gradient")
        print("    - Longer stimulus duration")
        print("    - Additional mechanisms (opponent neurons, inhibition)")
        
    else:
        print(f"\n✗✗✗ FAILURE: No discrimination ({diff} spikes)")
        print("    Calcium time constant alone is NOT sufficient")
        print("\n    Possible reasons:")
        print("    - Dendritic coupling too strong (spatial info destroyed)")
        print("    - Soma integration too coarse (loses spatial pattern)")
        print("    - Need network-level mechanisms:")
        print("      • Opponent neurons")
        print("      • Inhibitory gating")
        print("      • Multi-compartment readout")
    
    # Check if scrambled is between right and left
    print("\n" + "-"*80)
    print("Temporal order sensitivity check:")
    print("-"*80)
    
    if spikes_right > spikes_scrambled > spikes_left or spikes_left > spikes_scrambled > spikes_right:
        print(f"\n✓ Model IS sensitive to temporal order:")
        print(f"  Sequential (right): {spikes_right}")
        print(f"  Random (scrambled): {spikes_scrambled}")
        print(f"  Sequential (left):  {spikes_left}")
        print("\n  This proves discrimination comes from TEMPORAL ORDER,")
        print("  not just which compartments were activated!")
        
    elif diff < 2:
        print(f"\n✗ Model is NOT sensitive to temporal order")
        print(f"  All three conditions produce similar spike counts")
        
    else:
        print(f"\n? Unclear temporal sensitivity:")
        print(f"  Sequential vs scrambled pattern unclear")
    
    # Additional diagnostics
    print("\n" + "="*80)
    print("ADDITIONAL DIAGNOSTICS")
    print("="*80)
    
    # Compare mean voltages
    mean_v_right = float(jnp.mean(v_soma_right))
    mean_v_left = float(jnp.mean(v_soma_left))
    
    print(f"\nMean soma voltage:")
    print(f"  Right motion: {mean_v_right:.2f} mV")
    print(f"  Left motion:  {mean_v_left:.2f} mV")
    print(f"  Difference:   {mean_v_right - mean_v_left:.2f} mV")
    
    # Compare voltage variance
    var_v_right = float(jnp.var(v_soma_right))
    var_v_left = float(jnp.var(v_soma_left))
    
    print(f"\nSoma voltage variance (activity level):")
    print(f"  Right motion: {var_v_right:.1f} mV²")
    print(f"  Left motion:  {var_v_left:.1f} mV²")
    print(f"  Difference:   {var_v_right - var_v_left:.1f} mV²")
    
    # If multi-compartment data available, show spatial patterns
    if v_right.ndim == 2:
        print(f"\n" + "-"*80)
        print("Spatial activity patterns (variance per compartment):")
        print("-"*80)
        
        var_by_comp_right = [float(jnp.var(v_right[i])) for i in range(10)]
        var_by_comp_left = [float(jnp.var(v_left[i])) for i in range(10)]
        
        print("\nRight motion:")
        print("  Comp: " + "".join([f"{i:7d}" for i in range(10)]))
        print("  Var:  " + "".join([f"{v:7.1f}" for v in var_by_comp_right]))
        
        print("\nLeft motion:")
        print("  Comp: " + "".join([f"{i:7d}" for i in range(10)]))
        print("  Var:  " + "".join([f"{v:7.1f}" for v in var_by_comp_left]))
        
        # Compute where activity is concentrated
        left_region_r = float(jnp.mean(jnp.array(var_by_comp_right[0:3])))
        mid_region_r = float(jnp.mean(jnp.array(var_by_comp_right[4:6])))
        right_region_r = float(jnp.mean(jnp.array(var_by_comp_right[7:10])))
        
        left_region_l = float(jnp.mean(jnp.array(var_by_comp_left[0:3])))
        mid_region_l = float(jnp.mean(jnp.array(var_by_comp_left[4:6])))
        right_region_l = float(jnp.mean(jnp.array(var_by_comp_left[7:10])))
        
        print("\nSpatial concentration:")
        print("           LEFT    MID    RIGHT")
        print(f"  Right:  {left_region_r:6.1f}  {mid_region_r:6.1f}  {right_region_r:6.1f}")
        print(f"  Left:   {left_region_l:6.1f}  {mid_region_l:6.1f}  {right_region_l:6.1f}")

except Exception as e:
    print(f"\n✗ ERROR during simulation: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("END OF TEST")
print("="*80)