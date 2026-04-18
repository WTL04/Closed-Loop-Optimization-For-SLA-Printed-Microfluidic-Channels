from paraview.simple import *
import sys

# Load the foam file
case_file = "case.foam"
try:
    data = OpenFOAMReader(FileName=case_file)
    data.UpdatePipeline()
except Exception as e:
    print(f"Error loading foam file: {e}")
    print("FLOW_RATE:0.0")
    sys.exit(1)

# Create a slice at the outlet
# Assuming outlet is perpendicular to X axis at the end of the channel
bounds = data.GetDataInformation().GetBounds()
max_x = bounds[1]

slice_filter = Slice(Input=data)
slice_filter.SliceType.Normal = [1, 0, 0]
slice_filter.SliceType.Origin = [max_x, 0, 0]
slice_filter.UpdatePipeline()

# Integrate the velocity over the slice to get the flow rate
integrate = IntegrateVariables(Input=slice_filter)
integrate.UpdatePipeline()

# Extract the integrated value of U (velocity)
# IntegrateVariables results in a 1-point dataset containing the integrated values
try:
    # Get the point data
    point_data = integrate.GetPointData()
    u_array = point_data.GetArray("U")
    
    if u_array:
        # The integrated result is the value at the first (and only) point
        # u_array[0] is the integrated vector [integral(Ux), integral(Uy), integral(Uz)]
        integrated_vector = u_array[0]
        flow_rate = integrated_vector[0]  # X-component
        print(f"FLOW_RATE:{flow_rate}")
    else:
        print("FLOW_RATE:0.0")
except Exception as e:
    print(f"Error during integration: {e}")
    print("FLOW_RATE:0.0")
