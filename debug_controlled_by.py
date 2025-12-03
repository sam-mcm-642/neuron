"""Check how parameters are grouped"""

import sys
sys.path.insert(0, '.')

import jaxley as jx
from jaxley.channels import HH

comp = jx.Compartment()
branch = jx.Branch(comp, ncomp=10)
cell = jx.Cell([branch], parents=[-1])
cell.insert(HH())

print("BEFORE make_trainable:")
print("="*60)
print(cell.nodes[['HH_gNa', 'controlled_by_param']].head(10))

print("\n" + "="*60)
print("Making each compartment trainable separately...")

# Make each compartment separately
for comp_idx in range(10):
    cell.branch(0).comp(comp_idx).make_trainable("HH_gNa", verbose=False)

print("\nAFTER make_trainable:")
print("="*60)
print(cell.nodes[['HH_gNa', 'controlled_by_param']].head(10))

print("\n" + "="*60)
params = cell.get_parameters()
print(f"Number of param groups: {len(params)}")

# Check first 3 groups
for i in range(min(3, len(params))):
    p = params[i]
    key = list(p.keys())[0]
    val = p[key]
    print(f"  Group {i}: {key}, shape={val.shape}, value={float(val):.6f}")

print("\n✓ If you see 10 groups with shape=(1,), it worked!")