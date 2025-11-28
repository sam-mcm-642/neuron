import numpy as np

def create_1d_motion(direction='right', n_frames=5):
    """
    Create 1D motion stimulus
    
    direction: 'right' or 'left'
    n_frames: number of frames in sequence
    
    Returns: (n_frames, 10) array
    """
    frames = []
    
    for t in range(n_frames):
        pixels = np.zeros(10)
        
        if direction == 'right':
            pos = 2 + t  # Start at position 2, move right
        else:
            pos = 7 - t  # Start at position 7, move left
        
        if 0 <= pos < 10:
            pixels[pos] = 1.0
        
        frames.append(pixels)
    
    return np.array(frames)

# Test it
right_motion = create_1d_motion('right')
print("Right motion:")
for frame in right_motion:
    print(['█' if p > 0 else '·' for p in frame])

# Output:
# [·, ·, █, ·, ·, ·, ·, ·, ·, ·]
# [·, ·, ·, █, ·, ·, ·, ·, ·, ·]
# [·, ·, ·, ·, █, ·, ·, ·, ·, ·]
# [·, ·, ·, ·, ·, █, ·, ·, ·, ·]
# [·, ·, ·, ·, ·, ·, █, ·, ·, ·]