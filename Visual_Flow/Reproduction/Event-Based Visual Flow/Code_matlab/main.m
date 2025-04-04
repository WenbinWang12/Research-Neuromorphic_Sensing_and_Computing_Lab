clear, clc
events = load('..\Test_Dataset\IROS_Dataset-2018-independent-motion\IROS_Dataset\what_is_background\events.txt');
L = 3;
delta_t = 1e-3;
threshold1 = 1e-5;
threshold2 = 0.05;
[vx, vy] = Algorithm1(events, L, delta_t, threshold1, threshold2);


%% Visualization
img = imread('..\Test_Dataset\IROS_Dataset-2018-independent-motion\IROS_Dataset\what_is_background\images\frame_00000000.png');
[H, W, ~] = size(img);
% fprintf('Image size: %d x %d\n', H, W);

dt = 0.05; step = 5;
img = Accumulate_events(events, H, W, dt);
plot_flow_with_events(img, events(:,1), events(:,2), vx, vy, step);