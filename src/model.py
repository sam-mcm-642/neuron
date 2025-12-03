"""
Build biophysical neuron model for motion detection

The neuron has:
- 1 dendrite with 10 compartments (one per pixel)
- Hodgkin-Huxley ion channels (Na, K) - fast spiking
- L-type calcium channels (CaL) - slow temporal integration (~100ms)
- Trainable: CaL conductances with spatial gradient
"""

import jaxley as jx
from jaxley.channels import HH
from jaxley.channels.pospischil import CaL


def build_motion_detector():
    """
    Build a 10-compartment dendritic neuron with HH + CaL channels
    
    Returns:
        cell: JAXLEY neuron model
    """
    
    comp = jx.Compartment()
    dendrite = jx.Branch(comp, ncomp=10)
    cell = jx.Cell([dendrite], parents=[-1])
    
    # Insert ion channels
    cell.insert(HH())   # Hodgkin-Huxley (Na + K) for spiking
    cell.insert(CaL())  # L-type calcium for temporal integration
    
    print(f"✓ Created neuron with 10 compartments")
    print(f"  Channels: HH (Na, K) + CaL (L-type calcium)")
    
    return cell


# def make_trainable(cell, what='calcium'):
#     """
#     Make parameters trainable with SPATIAL INITIALIZATION
    
#     Initializes CaL with spatial gradient (left high → right low)
#     This breaks symmetry and gives gradient descent something to amplify
    
#     Args:
#         cell: JAXLEY neuron
#         what: 'calcium' to make CaL trainable
        
#     Returns:
#         cell: modified cell
#     """
    
#     if what == 'calcium':
#         # Initialize with spatial gradient (left high → right low)
#         # This creates initial directional bias
#         initial_gCaL = [
#             0.0008,  # Comp 0 - LEFT (high calcium sensitivity)
#             0.0007,  # Comp 1
#             0.0006,  # Comp 2
#             0.0005,  # Comp 3
#             0.0004,  # Comp 4
#             0.0003,  # Comp 5 - MIDDLE
#             0.0003,  # Comp 6
#             0.0002,  # Comp 7
#             0.0001,  # Comp 8
#             0.00005, # Comp 9 - RIGHT (low calcium sensitivity)
#         ]
        
#         print("\nMaking CaL conductances trainable...")
        
#         # Make EACH compartment's CaL conductance independently trainable
#         for comp_idx in range(10):
#             cell.branch(0).comp(comp_idx).make_trainable(
#                 "CaL_gCaL",
#                 init_val=initial_gCaL[comp_idx],
#                 verbose=False
#             )
        
#         print(f"✓ Created 10 independent CaL conductances with spatial gradient")
#         print(f"  Initial gCaL: {initial_gCaL[0]:.5f} (left) → {initial_gCaL[-1]:.5f} (right)")
#         print(f"  Ratio: {initial_gCaL[0]/initial_gCaL[-1]:.1f}×")
    
#     return cell


def make_trainable(cell, what='calcium'):
    """Initialize with NO spatial pattern"""
    
    if what == 'calcium':
        # ALL IDENTICAL - no spatial bias
        initial_gCaL = [0.0005] * 10  # Uniform
        
        for comp_idx in range(10):
            cell.branch(0).comp(comp_idx).make_trainable(
                "CaL_gCaL",
                init_val=initial_gCaL[comp_idx],
                verbose=False
            )
        
        print(f"✓ Initialized with UNIFORM gCaL (no spatial pattern)")
        print(f"  Model must learn spatial gradient from scratch")
    
    return cell

if __name__ == "__main__":
    """Test model creation"""
    
    print("Building motion detector neuron...\n")
    
    cell = build_motion_detector()
    cell = make_trainable(cell, what='calcium')
    
    params = cell.get_parameters()
    print(f"\n✓ Total trainable parameter groups: {len(params)}")
    
    # Verify CaL parameters
    cal_count = 0
    for i, p in enumerate(params):
        if 'CaL_gCaL' in p:
            cal_count += 1
            if cal_count <= 3:  # Show first 3
                print(f"  Param {i}: CaL_gCaL = {float(p['CaL_gCaL'][0]):.5f}")
    
    if cal_count == 10:
        print(f"  ... ({cal_count} CaL parameters total)")
        print("\n✓ Model building working!")
    else:
        print(f"\n⚠ Warning: Expected 10 CaL parameters, found {cal_count}")