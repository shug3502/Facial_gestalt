%script to load struct with all data in and load a numbered image
close all
load('entries_of_interest_ma.mat');
num_to_load = 40;
figure
imshow(imread(entries_of_interest(num_to_load).imagePath));
hold on
rectangle('Position',entries_of_interest(num_to_load).face_coordinates,'EdgeColor','g');
plot(entries_of_interest(num_to_load).ground_truth_points(:,1),entries_of_interest(num_to_load).ground_truth_points(:,2),'bo')
plot(entries_of_interest(num_to_load).landmarks_locations_jb(:,1),entries_of_interest(num_to_load).landmarks_locations_jb(:,2),'rx')
