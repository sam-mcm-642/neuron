"""Test training stability with minimal setup"""

import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import optax
from src.model import build_motion_detector, make_trainable
from src.stimulus import create_1d_motion
from src.simulate import simulate_sequence


def simple_loss(params, cell, right_motion, left_motion):
    """Simplest possible loss: mean voltage difference"""
    v_right, _ = simulate_sequence(cell, right_motion, params=params, verbose=False)
    v_left, _ = simulate_sequence(cell, left_motion, params=params, verbose=False)
    
    mean_diff = jnp.mean(v_right) - jnp.mean(v_left)
    
    # We want right > left, so minimize negative difference
    return -mean_diff


# Build model
cell = build_motion_detector()
cell = make_trainable(cell)
params = cell.get_parameters()

# Stimuli
right_motion = create_1d_motion('right', n_frames=5)
left_motion = create_1d_motion('left', n_frames=5)

# Test different optimizer configurations
configs = [
    ("SGD lr=0.001", optax.sgd(0.001)),
    ("SGD lr=0.0001", optax.sgd(0.0001)),
    ("Adam lr=0.001", optax.adam(0.001)),
    ("Adam lr=0.0001", optax.adam(0.0001)),
    ("Adam lr=0.00001", optax.adam(0.00001)),
]

for name, optimizer in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    # Reset parameters
    params = cell.get_parameters()
    opt_state = optimizer.init(params)
    
    losses = []
    failed = False
    
    for step in range(100):
        loss_value, grads = jax.value_and_grad(simple_loss)(
            params, cell, right_motion, left_motion
        )
        
        if jnp.isnan(loss_value):
            print(f"  ✗ NaN at step {step}")
            failed = True
            break
        
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        losses.append(float(loss_value))
        
        if step % 20 == 0:
            print(f"  Step {step}: Loss = {loss_value:.4f}")
    
    if not failed:
        # Check if loss decreased overall
        if losses[-1] < losses[0]:
            print(f"  ✓ SUCCESS: Loss {losses[0]:.4f} → {losses[-1]:.4f}")
        else:
            print(f"  ~ OSCILLATING: Loss {losses[0]:.4f} → {losses[-1]:.4f}")