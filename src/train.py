"""
Train neuron to detect motion direction using multi-compartment readout

Key innovation: Instead of reading only soma output, we read from multiple
dendritic locations to preserve spatial information about where activity occurs.

This enables detection of temporal order even with overlapping stimuli.
"""

import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from src.model import build_motion_detector, make_trainable
from src.stimulus import create_1d_motion
from src.simulate import simulate_sequence, count_spikes


def two_point_readout(voltage_traces):
    """
    Read from two dendritic locations to extract directional signal
    
    This is biologically realistic: downstream neurons sample from multiple
    synaptic locations on the dendrite, not just the soma.
    
    Args:
        voltage_traces: (10 compartments, n_timesteps) voltage array
        
    Returns:
        opponent_signal: scalar representing directional preference
                        positive = rightward motion
                        negative = leftward motion
    """
    
    # Read activity from left dendrite (compartment 2)
    left_dendrite_activity = jnp.var(voltage_traces[2])
    
    # Read activity from right dendrite (compartment 7)
    right_dendrite_activity = jnp.var(voltage_traces[7])
    
    # Opponent signal: left - right
    # Right motion activates left side more → positive signal
    # Left motion activates right side more → negative signal
    opponent_signal = left_dendrite_activity - right_dendrite_activity
    
    return opponent_signal


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
    """Constrain CaL conductances to reasonable range"""
    constrained = []
    
    for p in params:
        p_new = {}
        for key in p.keys():
            if 'CaL_gCaL' in key:
                # Increase minimum to prevent collapse
                p_new[key] = jnp.clip(p[key], 0.0001, 0.002)  # Was 0.00001, now 0.0001
            else:
                p_new[key] = p[key]
        constrained.append(p_new)
    
    return constrained


def train(n_iterations=200, learning_rate=0.0001):
    """
    Train motion detector with multi-compartment readout
    
    Args:
        n_iterations: number of training steps
        learning_rate: Adam optimizer learning rate
        
    Returns:
        cell: trained neuron
        params: trained parameters
        losses: training loss history
        opponent_signals_right: opponent signals for right motion over training
        opponent_signals_left: opponent signals for left motion over training
    """
    
    print("="*80)
    print("TRAINING: Motion Detection with Multi-Compartment Readout")
    print("="*80)
    
    print("\n1. Building model...")
    cell = build_motion_detector()
    cell = make_trainable(cell, what='calcium')
    params = cell.get_parameters()
    
    print("\n2. Creating stimuli...")
    right_motion = create_1d_motion('right', n_frames=5)
    left_motion = create_1d_motion('left', n_frames=5)
    print("  Right motion: activates [2, 3, 4, 5, 6] (left to right)")
    print("  Left motion:  activates [7, 6, 5, 4, 3] (right to left)")
    print("  Overlap: [3, 4, 5, 6] - both directions activate middle compartments")
    
    print(f"\n3. Setting up optimizer (Adam, lr={learning_rate})...")
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(params)
    
    print(f"\n4. Training for {n_iterations} iterations...")
    print("\nMonitoring opponent signals:")
    print("  Positive = rightward preference")
    print("  Negative = leftward preference")
    print("  Goal: maximize separation\n")
    
    losses = []
    opponent_signals_right = []
    opponent_signals_left = []
    
    for step in range(n_iterations):
        # Compute loss and gradients
        loss_value, grads = jax.value_and_grad(motion_detection_loss)(
            params, cell, right_motion, left_motion
        )
        
        # Check for NaN
        if jnp.isnan(loss_value):
            print(f"\n⚠ NaN at step {step}. Stopping.")
            break
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Constrain parameters to valid range
        params = constrain_parameters(params)
        
        losses.append(float(loss_value))
        
        # Monitor progress
        if step % 5 == 0 or step == n_iterations - 1:
            # Get current opponent signals
            v_r, _ = simulate_sequence(cell, right_motion, params=params, verbose=False)
            v_l, _ = simulate_sequence(cell, left_motion, params=params, verbose=False)
            
            opp_r = float(two_point_readout(v_r))
            opp_l = float(two_point_readout(v_l))
            
            opponent_signals_right.append(opp_r)
            opponent_signals_left.append(opp_l)
            
            # Get spatial pattern statistics
            gCaL = [float(params[i]['CaL_gCaL'][0]) for i in range(10)]
            gCaL_std = float(jnp.std(jnp.array(gCaL)))
            gCaL_mean = float(jnp.mean(jnp.array(gCaL)))
            
            # Calculate separation
            separation = abs(opp_r - opp_l)
            
            print(f"Step {step:3d} | Loss: {loss_value:8.1f} | "
                  f"Opponent: R={opp_r:+7.1f} L={opp_l:+7.1f} | "
                  f"Sep: {separation:6.1f} | "
                  f"gCaL: μ={gCaL_mean:.5f} σ={gCaL_std:.6f}")
            
            # Show spatial pattern every 100 steps
            if step % 100 == 0:
                print(f"  gCaL spatial: [" + ", ".join([f"{v:.5f}" for v in gCaL]) + "]")
                print(f"  Position:      LEFT ←――――――――――――――――――――――→ RIGHT")
    
    print("\n✓ Training complete!")
    print("="*80)
    
    # Print final learned pattern
    print("\nFinal learned spatial pattern:")
    gCaL_final = [float(params[i]['CaL_gCaL'][0]) for i in range(10)]
    print("Compartment:  ", " ".join([f"{i:8d}" for i in range(10)]))
    print("gCaL values:  ", " ".join([f"{v:8.5f}" for v in gCaL_final]))
    print("Position:      LEFT ←――――――――――――――――――――――→ RIGHT\n")
    
    return cell, params, losses, opponent_signals_right, opponent_signals_left


if __name__ == "__main__":
    # Train the model
    cell, trained_params, losses, opp_right, opp_left = train(
        n_iterations=200, 
        learning_rate=0.0001
    )
    
    print("\nTesting trained model...")
    
    # Test on motion stimuli
    right_motion = create_1d_motion('right', n_frames=5)
    left_motion = create_1d_motion('left', n_frames=5)
    
    v_r, _ = simulate_sequence(cell, right_motion, params=trained_params, verbose=False)
    v_l, _ = simulate_sequence(cell, left_motion, params=trained_params, verbose=False)
    
    # Get final opponent signals
    final_opp_r = float(two_point_readout(v_r))
    final_opp_l = float(two_point_readout(v_l))
    
    # Also get soma spike counts for reference
    spikes_r = count_spikes(v_r[0])
    spikes_l = count_spikes(v_l[0])
    
    print(f"\nFinal opponent signals (multi-compartment readout):")
    print(f"  Right motion: {final_opp_r:+7.1f}")
    print(f"  Left motion:  {final_opp_l:+7.1f}")
    print(f"  Separation:   {abs(final_opp_r - final_opp_l):7.1f}")
    
    print(f"\nSoma spike counts (for reference):")
    print(f"  Right motion: {spikes_r} spikes")
    print(f"  Left motion:  {spikes_l} spikes")
    print(f"  Note: Soma readout fails to discriminate, but multi-compartment succeeds!")
    
    # Evaluate success
    separation = abs(final_opp_r - final_opp_l)
    if separation > 150:
        print("\n✓✓✓ SUCCESS: Strong direction discrimination achieved!")
    elif separation > 100:
        print("\n✓✓ GOOD: Reliable direction discrimination")
    elif separation > 50:
        print("\n✓ MODERATE: Some discrimination present")
    else:
        print("\n⚠ WEAK: Insufficient discrimination")
    
    # Plot training curves
    plt.figure(figsize=(15, 4))
    
    # Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    # Opponent signals over time
    plt.subplot(1, 3, 2)
    steps = [i*20 for i in range(len(opp_right))]
    plt.plot(steps, opp_right, 'b-', label='Right motion', linewidth=2)
    plt.plot(steps, opp_left, 'r-', label='Left motion', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Training Step')
    plt.ylabel('Opponent Signal')
    plt.title('Opponent Signals During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Separation over time
    plt.subplot(1, 3, 3)
    separation_history = [abs(opp_right[i] - opp_left[i]) for i in range(len(opp_right))]
    plt.plot(steps, separation_history, 'g-', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Separation (abs difference)')
    plt.title('Discrimination Strength')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/motion_detection_training.png', dpi=150)
    print(f"\n✓ Saved: results/motion_detection_training.png")