function plot_flow_with_events(img, x, y, flow_u, flow_v, step)
    % Plot optical flow vectors overlaid on an accumulated event image.
    %
    % Inputs:
    %   img     - Accumulated event image (size: H x W).
    %   x, y    - Coordinates of events (vectors matching flow_u, flow_v).
    %   flow_u  - Optical flow velocities in the horizontal direction (vector).
    %   flow_v  - Optical flow velocities in the vertical direction (vector).
    %   step    - Sampling interval to reduce the density of vectors for visualization.
    
    figure;
    imagesc(img); colormap gray; axis image; hold on;

    % Sparse sampling to avoid overly dense vector visualization
    idx = 1:step:length(x);
    X = x(idx);
    Y = y(idx);
    U = flow_u(idx);
    V = flow_v(idx);

    % Plot vectors; negative sign for V compensates image coordinate system direction
    quiver(X, Y, U, -V, 'r');

    title('Event Frame with Visual Flow');
    axis off;
end
