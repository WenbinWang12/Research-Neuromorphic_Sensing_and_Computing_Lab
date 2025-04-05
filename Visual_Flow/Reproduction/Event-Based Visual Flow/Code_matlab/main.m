clear, clc
events = load('..\Test_Dataset\IROS_Dataset-2018-independent-motion\IROS_Dataset\what_is_background\events.txt');
L = 3;
delta_t = 1e-3;
threshold1 = 1e-5;
threshold2 = 0.05;
[vx, vy] = Algorithm1(events, L, delta_t, threshold1, threshold2);