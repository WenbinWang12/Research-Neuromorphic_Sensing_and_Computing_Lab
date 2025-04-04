function img = Accumulate_events(events, H, W, dt)
    % events: N x 4 matrix, each row is [x, y, t, polarity]
    % H, W: image height and width
    % dt: time window

    % sort by time (3rd column)
    events = sortrows(events, 3);
    t0 = events(1, 3);

    % select events within [t0, t0 + dt]
    mask = (events(:, 3) >= t0) & (events(:, 3) < t0 + dt);
    selected_events = events(mask, :);

    img = zeros(H, W);
    for i = 1:size(selected_events, 1)
        x = selected_events(i, 1);
        y = selected_events(i, 2);
        p = selected_events(i, 4);
        
        % ensure indices are within image bounds
        if x >= 1 && x <= W && y >= 1 && y <= H
            if p > 0
                img(y, x) = img(y, x) + 1;
            else
                img(y, x) = img(y, x) - 1;
            end
        end
    end
end
