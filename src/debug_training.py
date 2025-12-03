"""Debug training to find why NaN occurs"""

import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import optax
from src.model import build_motion_detector, make_trainable
from src.stimulus import create_1d_motion
from src.simulate import simulate_sequence, count_spikes


def classification_loss(params, cell, right_motion, left_motion):
    """Loss function - NO PRINTING (for gradients)"""
    
    v_right, _ = simulate_sequence(cell, right_motion, params=params, verbose=False)
    v_left, _ = simulate_sequence(cell, left_motion, params=params, verbose=False)
    
    max_v_right = jnp.max(v_right)
    max_v_left = jnp.max(v_left)
    
    voltage_difference = max_v_right - max_v_left
    target_difference = 40.0
    
    loss = (voltage_difference - target_difference) ** 2
    
    return loss


def debug_state(params, cell, right_motion, left_motion, step_name):
    """Debug function - WITH PRINTING (separate from gradients)"""
    
    print("\n" + "="*60)
    print(f"{step_name}")
    print("="*60)
    
    # Simulate
    v_right, _ = simulate_sequence(cell, right_motion, params=params, verbose=False)
    v_left, _ = simulate_sequence(cell, left_motion, params=params, verbose=False)
    
    max_v_right = float(jnp.max(v_right))
    max_v_left = float(jnp.max(v_left))
    min_v_right = float(jnp.min(v_right))
    min_v_left = float(jnp.min(v_left))
    mean_v_right = float(jnp.mean(v_right))
    mean_v_left = float(jnp.mean(v_left))
    
    voltage_difference = max_v_right - max_v_left
    
    loss = float(classification_loss(params, cell, right_motion, left_motion))
    
    spikes_right = count_spikes(v_right)
    spikes_left = count_spikes(v_left)
    
    print(f"\nVoltages:")
    print(f"  Right: min={min_v_right:.1f}, max={max_v_right:.1f}, mean={mean_v_right:.1f}")
    print(f"  Left:  min={min_v_left:.1f}, max={max_v_left:.1f}, mean={mean_v_left:.1f}")
    print(f"  Difference: {voltage_difference:.2f} (target: 40.0)")
    
    print(f"\nSpikes:")
    print(f"  Right: {spikes_right}")
    print(f"  Left:  {spikes_left}")
    
    print(f"\nLoss: {loss:.2f}")
    
    # Check parameters
    print(f"\nParameters (first 2 groups):")
    for i, p in enumerate(params[:2]):
        if 'HH_gNa' in p:
            vals = p['HH_gNa']
            print(f"  Param {i} (gNa): mean={float(jnp.mean(vals)):.4f}, min={float(jnp.min(vals)):.4f}, max={float(jnp.max(vals)):.4f}")
        elif 'HH_gK' in p:
            vals = p['HH_gK']
            print(f"  Param {i} (gK):  mean={float(jnp.mean(vals)):.4f}, min={float(jnp.min(vals)):.4f}, max={float(jnp.max(vals)):.4f}")
    
    return loss


print("Building model...")
cell = build_motion_detector()
cell = make_trainable(cell, what='conductances')
params = cell.get_parameters()

print("Creating stimuli...")
right_motion = create_1d_motion('right', n_frames=5)
left_motion = create_1d_motion('left', n_frames=5)

# STEP 0
debug_state(params, cell, right_motion, left_motion, "STEP 0: Initial State")

# Compute gradients (no printing here!)
print("\nComputing gradients...")
grads = jax.grad(classification_loss)(params, cell, right_motion, left_motion)

print("\nGradient statistics (first 2 groups):")
for i, g in enumerate(grads[:2]):
    if 'HH_gNa' in g:
        vals = g['HH_gNa']
        print(f"  Grad {i} (gNa): mean={float(jnp.mean(vals)):.8f}, min={float(jnp.min(vals)):.8f}, max={float(jnp.max(vals)):.8f}")
    elif 'HH_gK' in g:
        vals = g['HH_gK']
        print(f"  Grad {i} (gK):  mean={float(jnp.mean(vals)):.8f}, min={float(jnp.min(vals)):.8f}, max={float(jnp.max(vals)):.8f}")

# Check if gradients are zero
all_grads = jnp.concatenate([g['HH_gNa'] if 'HH_gNa' in g else g['HH_gK'] for g in grads])
grad_norm = float(jnp.linalg.norm(all_grads))
print(f"\nTotal gradient norm: {grad_norm:.8f}")

if grad_norm < 1e-8:
    print("⚠️  WARNING: Gradients are essentially ZERO! Model won't learn.")
else:
    print("✓ Gradients are non-zero.")

# Apply one update
print("\nApplying one gradient update...")
optimizer = optax.adam(0.001)
opt_state = optimizer.init(params)
updates, opt_state = optimizer.update(grads, opt_state)
params_new = optax.apply_updates(params, updates)

# STEP 1
debug_state(params_new, cell, right_motion, left_motion, "STEP 1: After One Update")

# Check parameter change
print("\nParameter changes:")
for i in range(min(2, len(params))):
    if 'HH_gNa' in params[i]:
        old_val = float(jnp.mean(params[i]['HH_gNa']))
        new_val = float(jnp.mean(params_new[i]['HH_gNa']))
        change = new_val - old_val
        print(f"  Param {i} (gNa): {old_val:.6f} → {new_val:.6f} (change: {change:.6f})")
    elif 'HH_gK' in params[i]:
        old_val = float(jnp.mean(params[i]['HH_gK']))
        new_val = float(jnp.mean(params_new[i]['HH_gK']))
        change = new_val - old_val
        print(f"  Param {i} (gK):  {old_val:.6f} → {new_val:.6f} (change: {change:.6f})")

print("\n✓ Debug complete!")