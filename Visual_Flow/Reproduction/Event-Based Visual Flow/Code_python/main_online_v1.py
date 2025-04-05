import numpy as np

def algorithm1_online(events, L, delta_t, threshold1, threshold2):
    """
    Online (causal) version of the Local Plane Fitting Algorithm.
    
    For each event (processed in increasing time order), only past events within 
    a time window [t - delta_t, t] are considered for spatial neighborhood extraction.
    
    Parameters:
        events (np.ndarray): N×3 array, each row is [x, y, t] (must be sorted by t)
        L (float): Size of the spatial neighborhood (L×L)
        delta_t (float): Time window size (only past events are used)
        threshold1 (float): Convergence threshold for fitting error
        threshold2 (float): Threshold for rejecting events with large errors
        
    Returns:
        vx (np.ndarray): Array of computed velocities in the x direction
        vy (np.ndarray): Array of computed velocities in the y direction
    """
    N = events.shape[0]
    vx = np.empty(N)
    vy = np.empty(N)
    
    # The sliding window buffer that stores past events (each as [x, y, t])
    buffer = []
    
    for i in range(N):
        current_event = events[i]
        p = current_event[:2]
        t = current_event[2]
        
        # Remove events from buffer that are older than t - delta_t (causal processing)
        buffer = [ev for ev in buffer if ev[2] >= t - delta_t]
        
        # Convert buffer to a numpy array for vectorized operations (if not empty)
        if len(buffer) > 0:
            buffer_arr = np.array(buffer)
        else:
            buffer_arr = np.empty((0, 3))
        
        # Find spatial neighbors in the buffer within a square window [p - L/2, p + L/2]
        if buffer_arr.shape[0] > 0:
            x_min, x_max = p[0] - L/2, p[0] + L/2
            y_min, y_max = p[1] - L/2, p[1] + L/2
            spatial_mask = ((buffer_arr[:, 0] >= x_min) & (buffer_arr[:, 0] <= x_max) &
                            (buffer_arr[:, 1] >= y_min) & (buffer_arr[:, 1] <= y_max))
            neighborhood = buffer_arr[spatial_mask, :]
        else:
            neighborhood = np.empty((0, 3))
        
        # If not enough neighbors for a plane fit, mark velocity as NaN
        if neighborhood.shape[0] < 3:
            vx[i] = np.nan
            vy[i] = np.nan
        else:
            # Iterative plane fitting using only past neighbors
            prev_plane = np.array([np.inf, np.inf, np.inf])
            epsilon = 1e6
            plane = None
            # Continue iterating until convergence (or until neighborhood becomes empty)
            while epsilon > threshold1 and neighborhood.shape[0] > 0:
                plane, err = fit_plane(neighborhood)
                neighborhood = reject_far_events(neighborhood, plane, threshold2)
                epsilon = np.linalg.norm(plane - prev_plane)
                prev_plane = plane
            # Extract velocity information from the fitted plane (assume plane: A*x + B*y + C = t)
            if plane is not None and plane[0] != 0 and plane[1] != 0:
                vx[i] = 1 / plane[0]
                vy[i] = 1 / plane[1]
            else:
                vx[i] = np.nan
                vy[i] = np.nan
        
        # Append the current event to the buffer for future processing
        buffer.append(current_event)
    
    return vx, vy

def fit_plane(neighborhood):
    """
    Fit a plane using least squares: A*x + B*y + C = t.
    
    Parameters:
        neighborhood (np.ndarray): Array of events in the neighborhood, each row is [x, y, t]
        
    Returns:
        plane (np.ndarray): Fitted plane parameters [A, B, C]
        error (float): Fitting error, computed as the norm ||A*plane - b||
    """
    # Construct matrix A and vector b
    A = np.hstack((neighborhood[:, 0:1], neighborhood[:, 1:2], np.ones((neighborhood.shape[0], 1))))
    b = neighborhood[:, 2]
    
    # Perform least squares fitting
    plane, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Compute the fitting error
    error = np.linalg.norm(A @ plane - b)
    return plane, error

def reject_far_events(neighborhood, plane, threshold):
    """
    Reject events whose absolute error from the fitted plane exceeds the threshold.
    
    Parameters:
        neighborhood (np.ndarray): Array of events in the neighborhood, each row is [x, y, t]
        plane (np.ndarray): Fitted plane parameters [A, B, C]
        threshold (float): Error threshold
        
    Returns:
        neighborhood (np.ndarray): Subset of events with error less than the threshold.
    """
    A = np.hstack((neighborhood[:, 0:1], neighborhood[:, 1:2], np.ones((neighborhood.shape[0], 1))))
    b = neighborhood[:, 2]
    errors = np.abs(A @ plane - b)
    return neighborhood[errors < threshold, :]

def read_events_from_file(file_path):
    """
    Read event data from a text file.
    
    Each line should be in the format: [x,y,t,polarity]
    This function ignores the polarity column and returns only the [x, y, t] values.
    
    Parameters:
        file_path (str): Path to the text file
        
    Returns:
        events (np.ndarray): N×3 NumPy array, each row is [x, y, t], sorted by time.
    """
    events = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Remove surrounding square brackets if present
            if line.startswith('[') and line.endswith(']'):
                line = line[1:-1]
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    t = float(parts[2].strip())
                    # polarity = float(parts[3].strip())  # Polarity is ignored
                    events.append([x, y, t])
                except ValueError:
                    continue
    # Convert to numpy array and sort by time (column 2)
    events = np.array(events)
    events = events[np.argsort(events[:, 2])]
    return events

def run_online_algorithm_from_file(file_path, L, delta_t, threshold1, threshold2):
    """
    Read event data from a file, process events online using the optimized causal algorithm,
    and print the computed velocities.
    
    Parameters:
        file_path (str): Path to the event data file
        L (float): Spatial neighborhood size
        delta_t (float): Time window size (for past events only)
        threshold1 (float): Convergence threshold for plane fitting
        threshold2 (float): Threshold for rejecting events with large fitting errors
    """
    events = read_events_from_file(file_path)
    print(f"Read {events.shape[0]} events from file {file_path}.")
    
    vx, vy = algorithm1_online(events, L, delta_t, threshold1, threshold2)
    
    print("Computed vx:", vx)
    print("Computed vy:", vy)
    return vx, vy

# Example usage
if __name__ == "__main__":
    file_path = "events.txt"  # Event data file: each line in the format [x,y,t,polarity]
    L = 10.0
    delta_t = 5.0  # Only past events within this time window are used
    threshold1 = 0.01
    threshold2 = 5.0
    
    run_online_algorithm_from_file(file_path, L, delta_t, threshold1, threshold2)
