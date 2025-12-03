"""
Generate 1D motion stimuli

A motion stimulus is a bright spot moving left or right across 10 pixels.
"""

import numpy as np


def create_1d_motion(direction='right', n_frames=5, speed=1):
    """
    Create 1D motion stimulus
    
    Args:
        direction: 'right' or 'left'
        n_frames: number of frames in sequence
        speed: pixels per frame (default 1)
    
    Returns:
        frames: (n_frames, 10) array, values in [0, 1]
        
    Example:
        >>> motion = create_1d_motion('right', n_frames=3)
        >>> print(motion.shape)
        (3, 10)
    """
    n_pixels = 10
    frames = []
    
    for t in range(n_frames):
        pixels = np.zeros(n_pixels)
        
        if direction == 'right':
            # Start at left, move right
            pos = 2 + (t * speed)
        elif direction == 'left':
            # Start at right, move left
            pos = 7 - (t * speed)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        # Set pixel if in bounds
        if 0 <= pos < n_pixels:
            pixels[int(pos)] = 1.0
        
        frames.append(pixels)
    
    return np.array(frames)


def visualize_motion(frames, title="Motion Stimulus"):
    """
    Print motion stimulus as ASCII art
    
    Args:
        frames: (n_frames, n_pixels) array
        title: plot title
    """
    print(f"\n{title}")
    print("-" * 40)
    
    for t, frame in enumerate(frames):
        # Convert to ASCII
        ascii_frame = ['█' if p > 0 else '·' for p in frame]
        print(f"Frame {t}: {''.join(ascii_frame)}")
    
    print("-" * 40)


def create_dataset(n_samples=100, n_frames=5):
    """
    Create training dataset with both directions
    
    Args:
        n_samples: number of examples per direction
        n_frames: frames per sequence
        
    Returns:
        data: list of dicts with 'sequence' and 'label'
              label = 0 for right, 1 for left
    """
    data = []
    
    # Right motion (label = 0)
    for _ in range(n_samples):
        sequence = create_1d_motion('right', n_frames=n_frames)
        data.append({
            'sequence': sequence,
            'label': 0,
            'direction': 'right'
        })
    
    # Left motion (label = 1)
    for _ in range(n_samples):
        sequence = create_1d_motion('left', n_frames=n_frames)
        data.append({
            'sequence': sequence,
            'label': 1,
            'direction': 'left'
        })
    
    # Shuffle
    np.random.shuffle(data)
    
    return data


if __name__ == "__main__":
    """Test stimulus generation"""
    
    print("Testing 1D motion stimulus generation...")
    
    # Create right motion
    right_motion = create_1d_motion('right', n_frames=5)
    visualize_motion(right_motion, title="Right Motion")
    
    # Create left motion
    left_motion = create_1d_motion('left', n_frames=5)
    visualize_motion(left_motion, title="Left Motion")
    
    # Test dataset creation
    dataset = create_dataset(n_samples=10, n_frames=5)
    print(f"\n✓ Created dataset with {len(dataset)} examples")
    print(f"  Example 0: {dataset[0]['direction']} motion")
    print(f"  Example 1: {dataset[1]['direction']} motion")
    
    print("\n✓ Stimulus generation working!")