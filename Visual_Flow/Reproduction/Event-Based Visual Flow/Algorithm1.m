% Local Plane Fitting Algorithm

function [vx, vy] = event_based_motion_flow(events, L, delta_t, threshold1, threshold2)
    % Inputs:
    % events: Matrix containing the event data, formatted as an N×3 matrix, each row contains (x, y, t) for event position and time
    % L: Size of the spatial neighborhood (L×L)
    % delta_t: Time window size for events
    % threshold1: Convergence threshold for fitting error
    % threshold2: Threshold for rejecting events with large errors
    
    N = size(events, 1); % Number of events
    vx = zeros(N, 1); % Stores the computed x-direction velocity
    vy = zeros(N, 1); % Stores the computed y-direction velocity
    
    for i = 1:N
        e = events(i, :); % Current event [x, y, t]
        p = e(1:2); % Event position [x, y]
        t = e(3); % Event time t
        
        % Define the spatiotemporal neighborhood
        neighborhood = find_neighborhood(events, p, t, L, delta_t);
        
        % Initialize plane fitting
        prev_plane = [Inf, Inf, Inf, Inf]; % Initial plane coefficients [a, b, c, d]
        epsilon = 1e6; % Initial error set to a large value
        
        % Fit the plane until convergence
        while epsilon > threshold1
            % Fit the plane using least squares
            [plane, error] = fit_plane(neighborhood);
            
            % Reject events with large errors
            neighborhood = reject_far_events(neighborhood, plane, threshold2);
            
            % Update fitting error and plane coefficients
            epsilon = norm(plane - prev_plane);
            prev_plane = plane;
        end
        
        % Extract velocity information from the fitted plane
        vx(i) = 1 / plane(1); % Velocity in the x-direction
        vy(i) = 1 / plane(2); % Velocity in the y-direction
    end
end

% Find the neighborhood for each event
function neighborhood = find_neighborhood(events, p, t, L, delta_t)
    % Find the spatiotemporal neighborhood, with a range of L×L×2*delta_t
    x_range = p(1) + [-L/2, L/2];
    y_range = p(2) + [-L/2, L/2];
    t_range = [t - delta_t, t + delta_t];
    
    % Find events within the neighborhood range
    neighborhood = events(events(:, 1) >= x_range(1) & events(:, 1) <= x_range(2) & ...
                          events(:, 2) >= y_range(1) & events(:, 2) <= y_range(2) & ...
                          events(:, 3) >= t_range(1) & events(:, 3) <= t_range(2), :);
end

% Fit the plane using least squares
function [plane, error] = fit_plane(neighborhood)
    % Fit a plane: Ax + By + C = t using least squares
    % Construct matrix A and vector b
    A = [neighborhood(:, 1), neighborhood(:, 2), ones(size(neighborhood, 1), 1)];
    b = neighborhood(:, 3);
    
    % Least squares plane fitting
    plane = A \ b;
    
    % Calculate the fitting error
    error = norm(A * plane - b);
end

% Reject events that have large fitting errors
function neighborhood = reject_far_events(neighborhood, plane, threshold)
    % Calculate the deviation of each event from the fitted plane
    A = [neighborhood(:, 1), neighborhood(:, 2), ones(size(neighborhood, 1), 1)];
    b = neighborhood(:, 3);
    error = abs(A * plane - b);
    
    % Retain events with error below the threshold
    neighborhood = neighborhood(error < threshold, :);
end