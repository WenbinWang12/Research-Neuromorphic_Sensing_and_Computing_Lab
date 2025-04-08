import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def algorithm1(events, L, delta_t, threshold1, threshold2):
    """
    Local Plane Fitting Algorithm in Python

    Parameters:
        events (np.ndarray): N × 4 array, each row is [x, y, t, polarity]
        L (float): Size of the spatial neighborhood (L × L)
        delta_t (float): Time window size
        threshold1 (float): Convergence threshold for fitting error
        threshold2 (float): Threshold for rejecting events with large errors

    Returns:
        vx (np.ndarray): Array of velocities in the x direction
        vy (np.ndarray): Array of velocities in the y direction
    """
    N = events.shape[0]
    vx = np.zeros(N)
    vy = np.zeros(N)
    
    for i in range(N):
        e = events[i, :]
        p = e[0:2]  # Event position [x, y]
        t = e[2]    # Event time
        
        # Get the spatiotemporal neighborhood for the event
        neighborhood = find_neighborhood(events, p, t, L, delta_t)
        
        # Initialize plane fitting
        prev_plane = np.array([np.inf, np.inf, np.inf])
        epsilon = 1e6  # Initial error set to a large value
        
        # Iteratively fit the plane until convergence
        while epsilon > threshold1 and neighborhood.shape[0] > 0:
            plane, err = fit_plane(neighborhood)
            neighborhood = reject_far_events(neighborhood, plane, threshold2)
            epsilon = np.linalg.norm(plane - prev_plane)
            prev_plane = plane
        
        # Extract velocity information from the fitted plane (assume plane: A*x + B*y + C = t)
        # Velocity computation: vx = 1/A, vy = 1/B
        # Note: Ensure that A and B are not zero, otherwise handle appropriately
        if plane[0] != 0 and plane[1] != 0:
            vx[i] = 1 / plane[0]
            vy[i] = 1 / plane[1]
        else:
            vx[i] = np.nan
            vy[i] = np.nan
            
    return vx, vy

def find_neighborhood(events, p, t, L, delta_t):
    """
    Find the spatiotemporal neighborhood for a given event in the events array

    Parameters:
        events (np.ndarray): N×3 array, each row is [x, y, t]
        p (array-like): Spatial position of the event [x, y]
        t (float): Event time
        L (float): Side length of the spatial neighborhood
        delta_t (float): Time window size

    Returns:
        neighborhood (np.ndarray): Subset of events within [p - L/2, p + L/2] and with time in [t - delta_t, t + delta_t]
    """
    x_min, x_max = p[0] - L/2, p[0] + L/2
    y_min, y_max = p[1] - L/2, p[1] + L/2
    t_min, t_max = t - delta_t, t + delta_t
    
    mask = ((events[:, 0] >= x_min) & (events[:, 0] <= x_max) &
            (events[:, 1] >= y_min) & (events[:, 1] <= y_max) &
            (events[:, 2] >= t_min) & (events[:, 2] <= t_max))
    
    return events[mask, :]

def fit_plane(neighborhood):
    """
    Fit a plane using least squares: A*x + B*y + C = t

    Parameters:
        neighborhood (np.ndarray): Subset of events in the neighborhood, each row is [x, y, t]

    Returns:
        plane (np.ndarray): Fitted plane parameters [A, B, C]
        error (float): Fitting error, computed as the norm of (A*plane - b)
    """
    # Construct matrix A and vector b
    A = np.hstack((neighborhood[:, 0:1], neighborhood[:, 1:2], np.ones((neighborhood.shape[0], 1))))
    b = neighborhood[:, 2]
    
    # Perform least squares plane fitting
    plane, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Compute the fitting error
    error = np.linalg.norm(A @ plane - b)
    return plane, error

def reject_far_events(neighborhood, plane, threshold):
    """
    Reject events with large fitting errors from the fitted plane

    Parameters:
        neighborhood (np.ndarray): Current set of events in the neighborhood, each row is [x, y, t]
        plane (np.ndarray): Fitted plane parameters [A, B, C]
        threshold (float): Error threshold

    Returns:
        neighborhood (np.ndarray): Subset of events with fitting error less than the threshold
    """
    A = np.hstack((neighborhood[:, 0:1], neighborhood[:, 1:2], np.ones((neighborhood.shape[0], 1))))
    b = neighborhood[:, 2]
    errors = np.abs(A @ plane - b)
    return neighborhood[errors < threshold, :]

def read_events_from_file(file_path):
    """
    Read event data from a text file.
    
    Each line should be in the format: [x , y, t, polarity]

    Parameters:
        file_path (str): Path to the text file

    Returns:
        events (np.ndarray): N×3 NumPy array, each row is [x, y, t, polarity]
    """
    events = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Remove surrounding square brackets if present
            if line.startswith('[') and line.endswith(']'):
                line = line[1:-1]
            # Split by comma
            parts = line.split(' ')
            if len(parts) >= 4:
                try:
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    t = float(parts[2].strip())
                    polarity = float(parts[3].strip())  
                    events.append([x, y, t, polarity])
                except ValueError:
                    # Skip the line if conversion fails
                    continue
    return np.array(events)

def run_algorithm_from_file(file_path, L, delta_t, threshold1, threshold2):
    """
    Read event data from a text file, run the plane fitting algorithm to compute velocities, and output the results.

    Parameters:
        file_path (str): Path to the event data file
        L (float): Spatial neighborhood size
        delta_t (float): Time window size
        threshold1 (float): Convergence threshold for fitting
        threshold2 (float): Threshold for rejecting events with large errors
    """
    # Read event data (ignoring the polarity column)
    events = read_events_from_file(file_path)
    print(f"Read {events.shape[0]} events from file {file_path}.")
    
    # Run the algorithm
    vx, vy = algorithm1(events, L, delta_t, threshold1, threshold2)
    
    # Print the computed velocities
    print("Computed vx:", vx)
    print("Computed vy:", vy)
    return events, vx, vy

# Example usage
if __name__ == "__main__":
    # Set the file path and parameters (modify as needed)
    file_path = r"Visual_Flow\Reproduction\Event-Based Visual Flow\Test_Dataset\IROS_Dataset-2018-independent-motion\IROS_Dataset\multiple_objects\2_objs\events.txt"  # Event data file, each line is in the format [x,y,t,polarity]
    L = 3
    delta_t = 1e-3
    threshold1 = 1e-5
    threshold2 = 0.05
    
    [events, vx, vy] = run_algorithm_from_file(file_path, L, delta_t, threshold1, threshold2)
    N = events.shape[0]
    x = events[:, 0]
    y = events[:, 1]
    t = events[:, 2]
    polarity = events[:, 3]

    print("t_max:", np.max(t))
    print("t_min:", np.min(t))

    # Image file path
    image_path = r"Visual_Flow\Reproduction\Event-Based Visual Flow\Test_Dataset\IROS_Dataset-2018-independent-motion\IROS_Dataset\multiple_objects\2_objs\images\frame_00000000.png"
    image = Image.open(image_path)
    height, width = image.size

    # Define the time windows
    dt = 5
    start_t = 0.0
    end_t = start_t + dt

    while end_t < max(t):
        # Filter events that fall within the defined time window
        mask = (t >= start_t) & (t < end_t)
        
        x_window = x[mask]
        y_window = y[mask]
        polarity_window = polarity[mask]
        vx_window = vx[mask]
        vy_window = vy[mask]

        # Accumulate events into an image representation
        event_image = np.zeros((height, width))
        for xi, yi, pi in zip(x_window, y_window, polarity_window):
            if 0 <= xi < width and 0 <= yi < height:
                event_image[yi, xi] += 1 if pi > 0 else -1

        # Update the time window
        start_t = end_t
        end_t = start_t + dt

    # Visualize the accumulated events image
    plt.figure(figsize=(8, 8))
    plt.imshow(event_image, cmap='gray')
    plt.title('Accumulated Events Representation')
    plt.colorbar()
    plt.show()