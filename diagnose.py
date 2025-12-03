"""
Systematic diagnostic: understand WHY the model isn't learning
"""

import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import optax

from src.model import build_motion_detector, make_trainable
from src.stimulus import create_1d_motion
from src.simulate import simulate_sequence, count_spikes


def create_embedding(voltage_traces):
    """Variance-based embedding"""
    spatial_pattern = jnp.array([jnp.var(voltage_traces[i]) for i in range(10)])
    return spatial_pattern / 1000


def contrastive_loss(params, cell, right_motion, left_motion):
    """
    Asymmetric target loss:
    - Right motion should produce HIGH total activity
    - Left motion should produce LOW total activity
    - This forces the model to make compartments selectively excitable
    """
    
    v_right_all, _ = simulate_sequence(cell, right_motion, params=params, verbose=False)
    v_left_all, _ = simulate_sequence(cell, left_motion, params=params, verbose=False)
    
    # Variance per compartment
    z_right = create_embedding(v_right_all)  # (10,) array
    z_left = create_embedding(v_left_all)    # (10,) array
    
    # Total activity (sum of variances across all compartments)
    total_activity_right = jnp.sum(z_right)
    total_activity_left = jnp.sum(z_left)
    
    # Component 1: Right should have HIGH activity
    # Target: total variance = 6.0 (high spiking across compartments)
    target_high = 6.0
    right_loss = (total_activity_right - target_high) ** 2
    
    # Component 2: Left should have LOW activity  
    # Target: total variance = 3.0 (low spiking)
    target_low = 3.0
    left_loss = (total_activity_left - target_low) ** 2
    
    # Component 3: Spatial diversity bonus (optional, helps)
    gNa_values = jnp.array([params[i]['HH_gNa'][0] for i in range(10)])
    spatial_diversity = jnp.var(gNa_values)
    
    # Combined: penalize deviation from targets, reward spatial patterns
    loss = right_loss + left_loss - spatial_diversity * 5.0
    
    return loss


print("="*80)
print("DIAGNOSTIC: Understanding Why The Model Isn't Learning")
print("="*80)

# Build model
cell = build_motion_detector()
cell = make_trainable(cell)
params = cell.get_parameters()

# Create stimuli
right_motion = create_1d_motion('right', n_frames=5)
left_motion = create_1d_motion('left', n_frames=5)

print("\n" + "="*80)
print("STEP 1: Initial State")
print("="*80)

# Get initial responses
v_r, _ = simulate_sequence(cell, right_motion, params=params, verbose=False)
v_l, _ = simulate_sequence(cell, left_motion, params=params, verbose=False)

z_r = create_embedding(v_r)
z_l = create_embedding(v_l)

spikes_r = count_spikes(v_r[0])
spikes_l = count_spikes(v_l[0])

print(f"\nInitial spike counts: Right={spikes_r}, Left={spikes_l}, Diff={spikes_r-spikes_l}")

print("\nInitial gNa values:")
gNa_init = [float(params[i]['HH_gNa'][0]) for i in range(10)]
print("  Comp: " + " ".join([f"{i:6d}" for i in range(10)]))
print("  gNa:  " + " ".join([f"{v:6.3f}" for v in gNa_init]))
print(f"  Mean: {jnp.mean(jnp.array(gNa_init)):.3f}, Std: {jnp.std(jnp.array(gNa_init)):.4f}")

print("\nInitial embeddings (variance per compartment):")
print("  Right: " + " ".join([f"{float(v):6.3f}" for v in z_r]))
print("  Left:  " + " ".join([f"{float(v):6.3f}" for v in z_l]))
print(f"\nInitial distance: {float(jnp.sqrt(jnp.sum((z_r - z_l)**2))):.4f}")

print("\n" + "="*80)
print("STEP 2: Compute Gradients")
print("="*80)

# Compute gradients
loss_value, grads = jax.value_and_grad(contrastive_loss)(params, cell, right_motion, left_motion)

print(f"\nLoss value: {float(loss_value):.4f}")

print("\nGradients for gNa:")
print("  Comp: " + " ".join([f"{i:8d}" for i in range(10)]))
grad_gNa = [float(grads[i]['HH_gNa'][0]) for i in range(10)]
print("  Grad: " + " ".join([f"{v:8.4f}" for v in grad_gNa]))

print("\nGradient statistics:")
print(f"  Mean gradient: {jnp.mean(jnp.array(grad_gNa)):.6f}")
print(f"  Std gradient:  {jnp.std(jnp.array(grad_gNa)):.6f}")
print(f"  Max gradient:  {jnp.max(jnp.array(grad_gNa)):.6f}")
print(f"  Min gradient:  {jnp.min(jnp.array(grad_gNa)):.6f}")

# Check if gradients are pushing in expected direction
print("\nGradient analysis:")
left_compartments = [0, 1, 2, 3, 4]
right_compartments = [5, 6, 7, 8, 9]

mean_grad_left = jnp.mean(jnp.array([grad_gNa[i] for i in left_compartments]))
mean_grad_right = jnp.mean(jnp.array([grad_gNa[i] for i in right_compartments]))

print(f"  Mean gradient (left compartments 0-4):  {float(mean_grad_left):+.6f}")
print(f"  Mean gradient (right compartments 5-9): {float(mean_grad_right):+.6f}")

if mean_grad_left < 0 and mean_grad_right > 0:
    print("  ✓ Gradients push LEFT compartments to INCREASE gNa (less negative)")
    print("  ✓ Gradients push RIGHT compartments to DECREASE gNa (more positive)")
elif mean_grad_left < 0 and mean_grad_right < 0:
    print("  ⚠ Both LEFT and RIGHT gradients are NEGATIVE")
    print("    → Optimizer will INCREASE gNa for BOTH sides")
    print("    → This explains why both spike counts increase!")
elif mean_grad_left > 0 and mean_grad_right > 0:
    print("  ⚠ Both LEFT and RIGHT gradients are POSITIVE")
    print("    → Optimizer will DECREASE gNa for BOTH sides")
    print("    → This explains why both spike counts decrease!")
else:
    print(f"  ? Mixed pattern: left={float(mean_grad_left):+.3f}, right={float(mean_grad_right):+.3f}")

print("\n" + "="*80)
print("STEP 3: Apply One Update")
print("="*80)

# Apply one gradient step
optimizer = optax.adam(0.005)
opt_state = optimizer.init(params)
updates, opt_state = optimizer.update(grads, opt_state)
params_new = optax.apply_updates(params, updates)

# Get new responses
v_r_new, _ = simulate_sequence(cell, right_motion, params=params_new, verbose=False)
v_l_new, _ = simulate_sequence(cell, left_motion, params=params_new, verbose=False)

spikes_r_new = count_spikes(v_r_new[0])
spikes_l_new = count_spikes(v_l_new[0])

print(f"\nAfter one update:")
print(f"  Right spikes: {spikes_r} → {spikes_r_new} (change: {spikes_r_new - spikes_r:+d})")
print(f"  Left spikes:  {spikes_l} → {spikes_l_new} (change: {spikes_l_new - spikes_l:+d})")

gNa_new = [float(params_new[i]['HH_gNa'][0]) for i in range(10)]
gNa_changes = [gNa_new[i] - gNa_init[i] for i in range(10)]

print("\ngNa changes:")
print("  Comp:   " + " ".join([f"{i:7d}" for i in range(10)]))
print("  Change: " + " ".join([f"{v:+7.4f}" for v in gNa_changes]))

mean_change_left = jnp.mean(jnp.array([gNa_changes[i] for i in left_compartments]))
mean_change_right = jnp.mean(jnp.array([gNa_changes[i] for i in right_compartments]))

print(f"\n  Mean change (left):  {float(mean_change_left):+.6f}")
print(f"  Mean change (right): {float(mean_change_right):+.6f}")

print("\n" + "="*80)
print("STEP 4: Diagnosis")
print("="*80)

print("\nKey Questions:")

# Question 1: Are embeddings different enough initially?
initial_distance = float(jnp.sqrt(jnp.sum((z_r - z_l)**2)))
if initial_distance < 0.5:
    print(f"\n1. Initial distance is SMALL ({initial_distance:.3f})")
    print("   → Embeddings are too similar from the start")
    print("   → Gradients will be weak")
else:
    print(f"\n1. Initial distance is REASONABLE ({initial_distance:.3f})")

# Question 2: Are gradients symmetric or asymmetric?
if abs(mean_grad_left) > 0.001 and abs(mean_grad_right) > 0.001:
    if (mean_grad_left < 0) == (mean_grad_right < 0):
        print(f"\n2. Gradients are SYMMETRIC (both {'+' if mean_grad_left > 0 else '-'})")
        print("   → Both sides changing in SAME direction")
        print("   → This prevents discrimination!")
    else:
        print(f"\n2. Gradients are ASYMMETRIC")
        print("   → Left and right changing in OPPOSITE directions")
        print("   → This should create discrimination")
else:
    print(f"\n2. Gradients are VERY SMALL (< 0.001)")
    print("   → Learning will be extremely slow")

# Question 3: Does update improve discrimination?
if (spikes_r_new - spikes_l_new) > (spikes_r - spikes_l):
    print(f"\n3. Update IMPROVED discrimination")
    print(f"   → Difference increased: {spikes_r - spikes_l} → {spikes_r_new - spikes_l_new}")
elif (spikes_r_new - spikes_l_new) < (spikes_r - spikes_l):
    print(f"\n3. Update WORSENED discrimination")
    print(f"   → Difference decreased: {spikes_r - spikes_l} → {spikes_r_new - spikes_l_new}")
else:
    print(f"\n3. Update had NO EFFECT on discrimination")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("\nBased on this analysis, the problem is:")
print("[To be determined by the diagnostic output above]")