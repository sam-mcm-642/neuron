"""
Simulate neuron response to motion stimuli
"""

import jax
import jax.numpy as jnp
import jaxley as jx


def pixels_to_currents(pixels, max_current=0.05):  # Reduced from 0.2 to 0.05
    """
    Convert pixel values to current amplitudes
    
    Args:
        pixels: (10,) array, values in [0, 1]
        max_current: maximum current in nA
        
    Returns:
        currents: (10,) array of current amplitudes
    """
    return pixels * max_current


def simulate_sequence(cell, sequence, params=None, dt=0.025, frame_duration=25.0, verbose=False):
    """
    Present motion sequence to neuron and record response
    
    Args:
        cell: JAXLEY neuron model
        sequence: (n_frames, 10) array of pixel values
        params: model parameters (if None, use defaults)
        dt: time step in ms
        frame_duration: duration of each frame in ms
        verbose: if True, print debug info
        
    Returns:
        voltages: (n_timesteps,) voltage trace at soma
        times: (n_timesteps,) time points
    """
    
    n_frames = len(sequence)
    t_max = n_frames * frame_duration
    
    # CRITICAL: Clear previous stimuli and recordings
    # This was causing accumulation across training iterations
    try:
        cell.delete_stimuli()
    except:
        pass
    
    try:
        cell.delete_recordings()
    except:
        pass
    
    # Create stimulus for each frame
    for frame_idx, pixels in enumerate(sequence):
        currents = pixels_to_currents(pixels)
        
        # Inject current to each compartment
        for comp_idx in range(10):
            if currents[comp_idx] > 0:  # Only if pixel is active
                current = jx.step_current(
                    i_delay=frame_idx * frame_duration,
                    i_dur=frame_duration,
                    i_amp=float(currents[comp_idx]),
                    delta_t=dt,
                    t_max=t_max
                )
                
                if not verbose:
                    # Suppress JAXLEY output
                    import io
                    import contextlib
                    
                    with contextlib.redirect_stdout(io.StringIO()):
                        cell.branch(0).comp(comp_idx).stimulate(current)
                else:
                    cell.branch(0).comp(comp_idx).stimulate(current)
    
    # Record voltage at ALL compartments for spatial pattern analysis
    if not verbose:
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            for comp_idx in range(10):
                cell.branch(0).comp(comp_idx).record("v")
    else:
        for comp_idx in range(10):
            cell.branch(0).comp(comp_idx).record("v")
    
    # Simulate
    if params is None:
        voltages = jx.integrate(cell, delta_t=dt)
    else:
        voltages = jx.integrate(cell, params=params, delta_t=dt)
    
    # Time axis
    times = jnp.arange(len(voltages)) * dt
    
    return voltages, times


def count_spikes(voltages, threshold=0.0):
    """
    Count action potentials in voltage trace
    
    Args:
        voltages: (n_timesteps,) voltage array
        threshold: spike threshold in mV
        
    Returns:
        n_spikes: number of spikes
    """
    # Find where voltage crosses threshold (upward)
    above_threshold = voltages > threshold
    crossings = jnp.diff(above_threshold.astype(int))
    upward_crossings = crossings > 0
    
    n_spikes = jnp.sum(upward_crossings)
    
    return int(n_spikes)


if __name__ == "__main__":
    """Test simulation"""
    
    print("Testing neuron simulation...\n")
    
    # Import stimulus
    import sys
    sys.path.append('.')
    from src.stimulus import create_1d_motion
    from src.model import build_motion_detector
    
    # Build neuron
    cell = build_motion_detector()
    
    # Create right motion
    print("Simulating RIGHT motion...")
    right_motion = create_1d_motion('right', n_frames=5)
    v_right, t_right = simulate_sequence(cell, right_motion)
    spikes_right = count_spikes(v_right)
    
    print(f"  Duration: {t_right[-1]:.1f} ms")
    print(f"  Voltage range: [{jnp.min(v_right):.1f}, {jnp.max(v_right):.1f}] mV")
    print(f"  Spikes: {spikes_right}")
    
    # Create left motion
    print("\nSimulating LEFT motion...")
    left_motion = create_1d_motion('left', n_frames=5)
    v_left, t_left = simulate_sequence(cell, left_motion)
    spikes_left = count_spikes(v_left)
    
    print(f"  Duration: {t_left[-1]:.1f} ms")
    print(f"  Voltage range: [{jnp.min(v_left):.1f}, {jnp.max(v_left):.1f}] mV")
    print(f"  Spikes: {spikes_left}")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        axes[0].plot(t_right, v_right, 'b-', linewidth=2, label='Right motion')
        axes[0].axhline(0, color='r', linestyle='--', alpha=0.3, label='Spike threshold')
        axes[0].set_ylabel('Voltage (mV)')
        axes[0].set_title(f'Right Motion (spikes: {spikes_right})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t_left, v_left, 'orange', linewidth=2, label='Left motion')
        axes[1].axhline(0, color='r', linestyle='--', alpha=0.3, label='Spike threshold')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Voltage (mV)')
        axes[1].set_title(f'Left Motion (spikes: {spikes_left})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/initial_response.png', dpi=150)
        print("\n✓ Saved: results/initial_response.png")
        plt.show()
        
    except ImportError:
        print("\n(Matplotlib not available, skipping plot)")
    
    print("\n✓ Simulation working!")