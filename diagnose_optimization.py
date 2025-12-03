"""
COMPREHENSIVE OPTIMIZATION DIAGNOSTIC

This script will trace through exactly what's happening in the optimization:
- Gradient values and patterns
- Parameter updates
- Loss landscape
- Variance contributions
- Cable coupling effects

Run this to understand WHY training fails.
"""

import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

from src.model import build_motion_detector, make_trainable
from src.stimulus import create_1d_motion
from src.simulate import simulate_sequence, count_spikes


def two_point_readout(voltage_traces):
    """Two-point readout"""
    left_dendrite = jnp.var(voltage_traces[2])
    right_dendrite = jnp.var(voltage_traces[7])
    return left_dendrite - right_dendrite


def motion_detection_loss(params, cell, right_motion, left_motion):
    """
    Loss function with proper normalization
    """
    
    v_right, _ = simulate_sequence(cell, right_motion, params=params, verbose=False)
    v_left, _ = simulate_sequence(cell, left_motion, params=params, verbose=False)
    
    opponent_right = two_point_readout(v_right)
    opponent_left = two_point_readout(v_left)
    
    # NORMALIZE by typical variance scale (~600 mV²)
    opponent_right_norm = opponent_right / 600.0
    opponent_left_norm = opponent_left / 600.0
    
    # Compute separation on normalized values
    separation = jnp.abs(opponent_right_norm - opponent_left_norm)
    
    # Loss: maximize separation
    loss = -separation
    
    return loss


def constrain_parameters(params):
    """Constrain parameters"""
    constrained = []
    for p in params:
        p_new = {}
        for key in p.keys():
            if 'CaL_gCaL' in key:
                p_new[key] = jnp.clip(p[key], 0.0001, 0.002)
            else:
                p_new[key] = p[key]
        constrained.append(p_new)
    return constrained


print("="*80)
print("COMPREHENSIVE OPTIMIZATION DIAGNOSTIC")
print("="*80)

# ==============================================================================
# Setup
# ==============================================================================

print("\n1. Building model...")
cell = build_motion_detector()
cell = make_trainable(cell, what='calcium')
params = cell.get_parameters()

right_motion = create_1d_motion('right', n_frames=5)
left_motion = create_1d_motion('left', n_frames=5)

print("✓ Model built")

# ==============================================================================
# DIAGNOSTIC 1: Initial State
# ==============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC 1: Initial Parameter and Response State")
print("="*80)

# Get initial parameters
gCaL_init = [float(params[i]['CaL_gCaL'][0]) for i in range(10)]
print("\nInitial gCaL values:")
print("  Comp: " + " ".join([f"{i:7d}" for i in range(10)]))
print("  gCaL: " + " ".join([f"{v:7.5f}" for v in gCaL_init]))
print(f"  Mean: {np.mean(gCaL_init):.6f}, Std: {np.std(gCaL_init):.6f}")

# Get initial responses
v_r_init, _ = simulate_sequence(cell, right_motion, params=params, verbose=False)
v_l_init, _ = simulate_sequence(cell, left_motion, params=params, verbose=False)

# Variance by compartment
var_r = [float(jnp.var(v_r_init[i])) for i in range(10)]
var_l = [float(jnp.var(v_l_init[i])) for i in range(10)]

print("\nVariance by compartment:")
print("  RIGHT motion:")
print("    Comp: " + " ".join([f"{i:6d}" for i in range(10)]))
print("    Var:  " + " ".join([f"{v:6.1f}" for v in var_r]))
print("  LEFT motion:")
print("    Comp: " + " ".join([f"{i:6d}" for i in range(10)]))
print("    Var:  " + " ".join([f"{v:6.1f}" for v in var_l]))

# Opponent signals
opp_r_init = float(two_point_readout(v_r_init))
opp_l_init = float(two_point_readout(v_l_init))
sep_init = abs(opp_r_init - opp_l_init)

print(f"\nOpponent signals:")
print(f"  Right motion: {opp_r_init:+.1f}")
print(f"  Left motion:  {opp_l_init:+.1f}")
print(f"  Separation:   {sep_init:.1f}")

# Initial loss
loss_init = float(motion_detection_loss(params, cell, right_motion, left_motion))
print(f"\nInitial loss: {loss_init:.2f}")

# ==============================================================================
# DIAGNOSTIC 2: Gradient Analysis
# ==============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC 2: Gradient Analysis at Initial State")
print("="*80)

# Compute gradients
grads = jax.grad(motion_detection_loss)(params, cell, right_motion, left_motion)

print("\nGradients with respect to gCaL:")
print("  Comp:     " + " ".join([f"{i:10d}" for i in range(10)]))

grad_vals = [float(grads[i]['CaL_gCaL'][0]) for i in range(10)]
print("  Gradient: " + " ".join([f"{g:+10.6f}" for g in grad_vals]))

print(f"\nGradient statistics:")
print(f"  Mean: {np.mean(grad_vals):+.8f}")
print(f"  Std:  {np.std(grad_vals):.8f}")
print(f"  Min:  {np.min(grad_vals):+.8f} (comp {np.argmin(grad_vals)})")
print(f"  Max:  {np.max(grad_vals):+.8f} (comp {np.argmax(grad_vals)})")

# Count gradient directions
n_positive = sum(1 for g in grad_vals if g > 0)
n_negative = sum(1 for g in grad_vals if g < 0)
print(f"\nGradient directions:")
print(f"  Positive (want to increase gCaL): {n_positive}/10")
print(f"  Negative (want to decrease gCaL): {n_negative}/10")

if n_negative >= 8:
    print("\n  ⚠ WARNING: Most gradients are negative!")
    print("    → Optimizer wants to DECREASE calcium everywhere")
    print("    → This explains parameter collapse!")

# Gradient to parameter ratio
ratios = [abs(grad_vals[i]) / (gCaL_init[i] + 1e-10) for i in range(10)]
print(f"\nGradient/Parameter ratios:")
print("  Comp:  " + " ".join([f"{i:8d}" for i in range(10)]))
print("  Ratio: " + " ".join([f"{r:8.2f}" for r in ratios]))
print(f"  Mean ratio: {np.mean(ratios):.2f}")

if np.mean(ratios) > 10:
    print("\n  ⚠ WARNING: Gradients are very large relative to parameters!")
    print("    → May cause instability or large jumps")

# ==============================================================================
# DIAGNOSTIC 3: Gradient Decomposition
# ==============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC 3: How Each Parameter Affects Each Direction")
print("="*80)

print("\nComputing partial derivatives...")
print("(This shows how changing each gCaL affects right vs left opponent signals)\n")

# Function to get opponent signal for right motion only
def opponent_right_only(params):
    v_r, _ = simulate_sequence(cell, right_motion, params=params, verbose=False)
    return two_point_readout(v_r)

# Function to get opponent signal for left motion only
def opponent_left_only(params):
    v_l, _ = simulate_sequence(cell, left_motion, params=params, verbose=False)
    return two_point_readout(v_l)

grad_opp_r = jax.grad(opponent_right_only)(params)
grad_opp_l = jax.grad(opponent_left_only)(params)

print("Effect of each gCaL on opponent signals:")
print("  Comp:      " + " ".join([f"{i:10d}" for i in range(10)]))

grad_r_vals = [float(grad_opp_r[i]['CaL_gCaL'][0]) for i in range(10)]
grad_l_vals = [float(grad_opp_l[i]['CaL_gCaL'][0]) for i in range(10)]

print("  ∂opp_R/∂g: " + " ".join([f"{g:+10.6f}" for g in grad_r_vals]))
print("  ∂opp_L/∂g: " + " ".join([f"{g:+10.6f}" for g in grad_l_vals]))
print("  Difference:" + " ".join([f"{grad_r_vals[i]-grad_l_vals[i]:+10.6f}" for i in range(10)]))

print("\nInterpretation:")
print("  - Positive ∂opp_R/∂g: increasing gCaL makes RIGHT opponent signal more positive")
print("  - Positive ∂opp_L/∂g: increasing gCaL makes LEFT opponent signal more positive")
print("  - Large difference: parameter has asymmetric effect (good for learning)")
print("  - Small difference: parameter affects both directions similarly (bad for learning)")

# Check if gradients are symmetric
symmetric_count = sum(1 for i in range(10) if abs(grad_r_vals[i] - grad_l_vals[i]) < 1e-6)
if symmetric_count > 5:
    print(f"\n  ⚠ WARNING: {symmetric_count}/10 parameters have symmetric effects!")
    print("    → These parameters affect both directions equally")
    print("    → Cannot contribute to discrimination")

# ==============================================================================
# DIAGNOSTIC 4: Loss Landscape Around Current Position
# ==============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC 4: Loss Landscape Exploration")
print("="*80)

print("\nTesting what happens if we perturb parameters...")

perturbations = {
    "Increase all gCaL by 10%": [p * 1.1 for p in gCaL_init],
    "Decrease all gCaL by 10%": [p * 0.9 for p in gCaL_init],
    "Increase left (0-4) by 20%": [gCaL_init[i] * 1.2 if i < 5 else gCaL_init[i] for i in range(10)],
    "Decrease right (5-9) by 20%": [gCaL_init[i] if i < 5 else gCaL_init[i] * 0.8 for i in range(10)],
    "Flatten (all equal to mean)": [np.mean(gCaL_init)] * 10,
    "Reverse gradient": gCaL_init[::-1],
}

print("\nLoss changes from perturbations:")
for desc, new_gCaL in perturbations.items():
    # Create new params
    params_test = []
    for i in range(10):
        params_test.append({'CaL_gCaL': jnp.array([new_gCaL[i]])})
    
    # Compute loss
    loss_test = float(motion_detection_loss(params_test, cell, right_motion, left_motion))
    delta_loss = loss_test - loss_init
    
    # Get separation
    v_r_test, _ = simulate_sequence(cell, right_motion, params=params_test, verbose=False)
    v_l_test, _ = simulate_sequence(cell, left_motion, params=params_test, verbose=False)
    sep_test = abs(float(two_point_readout(v_r_test)) - float(two_point_readout(v_l_test)))
    delta_sep = sep_test - sep_init
    
    symbol = "✓" if delta_loss < 0 else "✗"
    print(f"  {symbol} {desc:30s}: loss {loss_test:7.1f} ({delta_loss:+6.1f}), sep {sep_test:5.1f} ({delta_sep:+4.1f})")

print("\nKey findings:")
if perturbations_results := []:  # Would need to track this properly
    print("  (Analyze which perturbations improve loss)")

# ==============================================================================
# DIAGNOSTIC 5: Simulate One Training Step
# ==============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC 5: Detailed Single Training Step")
print("="*80)

print("\nSimulating one Adam update step...")

# Setup optimizer
learning_rate = 0.00005
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Compute updates
updates, opt_state_new = optimizer.update(grads, opt_state)

print("\nRaw gradient-based updates (before momentum/adaptation):")
print("  Comp:   " + " ".join([f"{i:10d}" for i in range(10)]))
raw_updates = [float(updates[i]['CaL_gCaL'][0]) for i in range(10)]
print("  Update: " + " ".join([f"{u:+10.8f}" for u in raw_updates]))

# Apply updates
params_new = optax.apply_updates(params, updates)

print("\nParameters before/after update:")
gCaL_new_unconstrained = [float(params_new[i]['CaL_gCaL'][0]) for i in range(10)]
print("  Comp:       " + " ".join([f"{i:10d}" for i in range(10)]))
print("  Before:     " + " ".join([f"{gCaL_init[i]:10.6f}" for i in range(10)]))
print("  After:      " + " ".join([f"{gCaL_new_unconstrained[i]:10.6f}" for i in range(10)]))
print("  Change:     " + " ".join([f"{gCaL_new_unconstrained[i]-gCaL_init[i]:+10.6f}" for i in range(10)]))

# Apply constraints
params_new_constrained = constrain_parameters(params_new)
gCaL_new_constrained = [float(params_new_constrained[i]['CaL_gCaL'][0]) for i in range(10)]

print("\nAfter constraints [0.0001, 0.002]:")
print("  After:      " + " ".join([f"{gCaL_new_constrained[i]:10.6f}" for i in range(10)]))

# Count how many hit bounds
hit_lower = sum(1 for g in gCaL_new_constrained if g <= 0.0001001)
hit_upper = sum(1 for g in gCaL_new_constrained if g >= 0.001999)

if hit_lower > 0:
    print(f"\n  ⚠ {hit_lower}/10 parameters hit LOWER bound (0.0001)")
    print("    → These parameters want to go lower but are constrained")
    print("    → Gradients are being clipped")

if hit_upper > 0:
    print(f"\n  ⚠ {hit_upper}/10 parameters hit UPPER bound (0.002)")

# Compute new loss
loss_new = float(motion_detection_loss(params_new_constrained, cell, right_motion, left_motion))

print(f"\nLoss change after one step:")
print(f"  Before: {loss_init:.2f}")
print(f"  After:  {loss_new:.2f}")
print(f"  Change: {loss_new - loss_init:+.2f}")

if loss_new < loss_init:
    print("  ✓ Loss decreased (separation increased)")
else:
    print("  ✗ Loss increased (separation decreased)")

# ==============================================================================
# DIAGNOSTIC 6: Cable Coupling Analysis
# ==============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC 6: Cable Coupling and Propagation")
print("="*80)

print("\nAnalyzing how voltage propagates through the cable...")

# Stimulate single compartment, see what happens
def stimulate_single_comp(comp_idx):
    """Stimulate only one compartment, see voltage everywhere"""
    single_stim = jnp.zeros((1, 10))
    single_stim = single_stim.at[0, comp_idx].set(1.0)
    v, _ = simulate_sequence(cell, single_stim, params=params, verbose=False)
    return v

print("\nStimulating individual compartments:")
print("(Shows how voltage spreads through cable)\n")

for test_comp in [2, 5, 7]:
    v_single = stimulate_single_comp(test_comp)
    mean_voltages = [float(jnp.mean(v_single[i])) for i in range(10)]
    
    print(f"  Stimulate comp {test_comp}:")
    print(f"    Mean voltage: " + " ".join([f"{v:6.2f}" for v in mean_voltages]))
    print(f"    Peak at comp: {np.argmax(mean_voltages)} (stimulus was at {test_comp})")
    
    if np.argmax(mean_voltages) != test_comp:
        print(f"    ⚠ Peak is NOT at stimulated compartment!")
        print(f"      → Cable coupling/reflection dominates")

# ==============================================================================
# Summary and Conclusions
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY AND DIAGNOSIS")
print("="*80)

print("\nKey Findings:")

findings = []

# Check gradient directions
if n_negative >= 8:
    findings.append("CRITICAL: Gradients overwhelmingly negative → optimizer wants less calcium")

# Check gradient magnitudes
if np.mean(ratios) > 10:
    findings.append("WARNING: Gradients very large relative to parameters")

# Check symmetry
if symmetric_count > 5:
    findings.append(f"PROBLEM: {symmetric_count}/10 parameters have symmetric effects on both directions")

# Check constraint hits
if hit_lower > 0:
    findings.append(f"CONSTRAINT: {hit_lower}/10 parameters hit lower bound immediately")

# Check cable effects
peak_mismatch = sum(1 for i in [2, 5, 7] if np.argmax([float(jnp.mean(stimulate_single_comp(i)[j])) for j in range(10)]) != i)
if peak_mismatch > 0:
    findings.append("CABLE ISSUE: Peak voltage NOT at stimulated compartment → reflections dominate")

for i, finding in enumerate(findings, 1):
    print(f"\n{i}. {finding}")

print("\n" + "="*80)
print("RECOMMENDED ACTIONS")
print("="*80)

print("\nBased on this diagnostic:")
# Will be filled in based on actual results

print("\n✓ Diagnostic complete")
print("="*80)