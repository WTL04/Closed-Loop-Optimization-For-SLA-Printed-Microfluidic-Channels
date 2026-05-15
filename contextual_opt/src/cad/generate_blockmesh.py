"""
Generate blockMeshDict for rectangular channel CFD simulation.

Creates a parametric blockMesh with boundary layer grading for accurate
flow rate simulation. Replaces the snappyHexMesh approach which had
mesh quality issues with small channel geometries.

Usage:
    python generate_blockmesh.py <L_um> <W_um> <H_um> <delta_L_um> <delta_W_um> <delta_H_um>

    L, W, H: CBO suggested dimensions in micrometers
    delta_*: Printer error deltas in micrometers
    
    Physical dimension = CBO_suggested - delta (the actual printed result)
"""

import os
import sys
import math

# CFD extensions (mm) - inlet and outlet regions for flow development
INLET_LENGTH_MM = 5.0
OUTLET_LENGTH_MM = 5.0

# Output path
BLOCKMESH_DICT_PATH = "cfd/channelCase/system/blockMeshDict"

# Mesh parameters - Conservative settings for boundary layer resolution
FIRST_CELL_SIZE_UM = 10.0  # First cell height at wall (micrometers)
TARGET_X_CELL_SIZE_UM = 200.0    # Target cell size along flow direction


def fmt_float(value, decimals=6):
    """Format float stripping unnecessary trailing zeros."""
    return f"{value:.{decimals}f}".rstrip('0').rstrip('.')


def compute_physical_dimensions(cbo_l_um, cbo_w_um, cbo_h_um, 
                                 delta_l_um, delta_w_um, delta_h_um):
    """
    Calculate actual physical dimensions after printer error.
    Physical = CBO_suggested - delta
    Returns dimensions in millimeters.
    """
    phys_l_mm = (cbo_l_um - delta_l_um) / 1000.0
    phys_w_mm = (cbo_w_um - delta_w_um) / 1000.0
    phys_h_mm = (cbo_h_um - delta_h_um) / 1000.0
    
    if phys_l_mm <= 0 or phys_w_mm <= 0 or phys_h_mm <= 0:
        raise ValueError(
            f"Physical dimensions must be positive. Got: "
            f"L={phys_l_mm:.4f}mm, W={phys_w_mm:.4f}mm, H={phys_h_mm:.4f}mm"
        )
    
    return phys_l_mm, phys_w_mm, phys_h_mm


def calculate_expansion_grading(half_length_mm, n_cells, first_cell_mm):
    """
    Calculate the grading ratio to achieve desired first cell size.
    
    For geometric grading: L = first * (r^n - 1) / (r - 1)
    Solving for r given L, n, and first cell size.
    
    Returns the expansion ratio (center_cell / wall_cell).
    """
    if n_cells <= 1:
        return 1.0
    
    uniform_size = half_length_mm / n_cells
    
    if first_cell_mm >= uniform_size:
        return 1.0
    
    target_sum = half_length_mm
    first = first_cell_mm
    n = n_cells
    
    r_low, r_high = 1.01, 10.0
    r_mid = (r_low + r_high) / 2.0
    
    for _ in range(50):
        r_mid = (r_low + r_high) / 2.0
        if r_mid <= 1.0:
            r_mid = 1.01
        
        calc_sum = first * (r_mid**n - 1) / (r_mid - 1)
        
        if abs(calc_sum - target_sum) < 1e-9:
            break
        elif calc_sum < target_sum:
            r_low = r_mid
        else:
            r_high = r_mid
    
    r_mid = max(1.1, min(r_mid, 8.0))
    return r_mid


def generate_blockmesh_dict(L_mm, W_mm, H_mm):
    """
    Generate the complete blockMeshDict content for a rectangular channel.
    
    Coordinate system:
    - X: flow direction (0 to total_length)
    - Y: width direction (-W/2 to +W/2, centered at 0)
    - Z: height direction (0 to H)
    
    Patches:
    - inlet: X=0 face
    - outlet: X=total_length face
    - walls: Y and Z boundary faces (4 faces total)
    """
    total_L = INLET_LENGTH_MM + L_mm + OUTLET_LENGTH_MM
    half_W = W_mm / 2.0
    
    min_dim_mm = min(W_mm, H_mm)
    min_cells_across = 20  # Ensure good resolution
    
    cell_size_mm = min_dim_mm / min_cells_across
    cell_size_um = cell_size_mm * 1000
    
    n_x = max(20, int(math.ceil(total_L / (TARGET_X_CELL_SIZE_UM / 1000))))
    n_y = max(min_cells_across, int(math.ceil(W_mm / cell_size_mm)))
    n_z = max(min_cells_across, int(math.ceil(H_mm / cell_size_mm)))
    
    # Ensure even numbers for symmetric grading
    if n_y % 2 != 0:
        n_y += 1
    if n_z % 2 != 0:
        n_z += 1
    
    # Calculate symmetric grading ratios
    # For boundary layer: fine at walls, coarser in center
    # Grading ratio = center_cell_size / wall_cell_size
    # For simpleGrading with symmetric, we use ratio > 1 on one side, < 1 on other
    
    # Simple approach: use edge grading with expansion toward center
    # grading ratio for half the domain
    half_n_y = n_y // 2
    half_n_z = n_z // 2
    
    # Calculate grading to get FIRST_CELL_SIZE_UM at walls
    # For geometric series: total = first * (r^n - 1)/(r - 1)
    # We want first cell = FIRST_CELL_SIZE_UM, expanding toward center
    
    # Target first cell at wall
    first_cell_mm = FIRST_CELL_SIZE_UM / 1000.0
    
    # For Y direction
    y_grading = calculate_expansion_grading(half_W, half_n_y, first_cell_mm)
    
    # For Z direction  
    z_grading = calculate_expansion_grading(H_mm / 2.0, half_n_z, first_cell_mm)
    
    inv_y_grading = 1.0 / y_grading
    inv_z_grading = 1.0 / z_grading
    total_cells = n_x * n_y * n_z
    
    # Build the blockMeshDict content
    content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

// Mesh generated by generate_blockmesh.py
// Channel dimensions: L={L_mm:.4f}mm, W={W_mm:.4f}mm, H={H_mm:.4f}mm
// Total cells: {total_cells} ({n_x} x {n_y} x {n_z})

scale 0.001;  // Dimensions below are in mm, converted to meters

vertices
(
    (0       {fmt_float(-half_W)}  0)
    ({fmt_float(total_L)}  {fmt_float(-half_W)}  0)
    ({fmt_float(total_L)}  {fmt_float(half_W)}   0)
    (0       {fmt_float(half_W)}   0)
    (0       {fmt_float(-half_W)}  {fmt_float(H_mm)})
    ({fmt_float(total_L)}  {fmt_float(-half_W)}  {fmt_float(H_mm)})
    ({fmt_float(total_L)}  {fmt_float(half_W)}   {fmt_float(H_mm)})
    (0       {fmt_float(half_W)}   {fmt_float(H_mm)})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({n_x} {n_y} {n_z})
    simpleGrading
    (
        1
        (
            (0.5 0.5 {y_grading:.4f})
            (0.5 0.5 {inv_y_grading:.4f})
        )
        (
            (0.5 0.5 {z_grading:.4f})
            (0.5 0.5 {inv_z_grading:.4f})
        )
    )
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    
    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
    
    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);
"""
    return content


def write_blockmesh_dict(content, output_path):
    """Write blockMeshDict to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Generated {output_path}")


def run(cbo_l_um, cbo_w_um, cbo_h_um, delta_l_um, delta_w_um, delta_h_um):
    """
    Main function to generate blockMeshDict from CBO parameters.
    """
    # Calculate physical dimensions
    L_mm, W_mm, H_mm = compute_physical_dimensions(
        cbo_l_um, cbo_w_um, cbo_h_um,
        delta_l_um, delta_w_um, delta_h_um
    )
    
    print(f"Physical channel dimensions: L={L_mm:.4f}mm, W={W_mm:.4f}mm, H={H_mm:.4f}mm")
    print(f"Total CFD length (with extensions): {INLET_LENGTH_MM + L_mm + OUTLET_LENGTH_MM:.4f}mm")
    
    # Generate blockMeshDict
    content = generate_blockmesh_dict(L_mm, W_mm, H_mm)
    
    # Write to file
    write_blockmesh_dict(content, BLOCKMESH_DICT_PATH)
    
    return L_mm, W_mm, H_mm


if __name__ == "__main__":
    if len(sys.argv) >= 7:
        cbo_l = float(sys.argv[1])
        cbo_w = float(sys.argv[2])
        cbo_h = float(sys.argv[3])
        delta_l = float(sys.argv[4])
        delta_w = float(sys.argv[5])
        delta_h = float(sys.argv[6])
    elif len(sys.argv) >= 4:
        print("Received 3 args - treating as deltas only, using nominal CBO inputs.")
        cbo_l = 40000.0
        cbo_w = 500.0
        cbo_h = 500.0
        delta_l = float(sys.argv[1])
        delta_w = float(sys.argv[2])
        delta_h = float(sys.argv[3])
    else:
        print("No arguments provided. Using nominal dimensions with zero deltas.")
        cbo_l, cbo_w, cbo_h = 40000.0, 500.0, 500.0
        delta_l, delta_w, delta_h = 0.0, 0.0, 0.0
    
    run(cbo_l, cbo_w, cbo_h, delta_l, delta_w, delta_h)
