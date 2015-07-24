function [input_data, input_labels, input_id] = crop_faces(entries_of_interest)
%function to crop and rotate faces
close all
tic;
if nargin <1
    load('entries_of_interest_ma.mat');
end

scale = [125,125]; %[145, 131];
input_data = zeros(scale(1),scale(2),length(entries_of_interest));
input_labels = zeros(1, length(entries_of_interest));
for i=1:length(entries_of_interest)
current_im = imread(entries_of_interest(i).imagePath);
cropped_im = imcrop(current_im,entries_of_interest(i).face_coordinates);

%now rescale and rotate according to canonical transform
J = imresize(cropped_im, scale); 
% figure(1)
% imshow(J);
J = rgb2gray(J);
%  if num_to_load==109
%      figure(5), imshow(current_im)
%  end
 
% figure;
% [ proc_image ] = pre_proc_test( current_im, entries_of_interest(num_to_load).ground_truth_points, 0);
% s = size(proc_image);
input_data(:,:,i) = reshape(J,scale(1),scale(2));
input_labels(i) = 1*strcmp(entries_of_interest(i).syndrome,'Angelman') ...
    + 2*strcmp(entries_of_interest(i).syndrome,'Apert') ...
    + 3*strcmp(entries_of_interest(i).syndrome,'Cornelia_de_Lange_Syndrome')...
    + 4*strcmp(entries_of_interest(i).syndrome,'Down')...
    + 5*strcmp(entries_of_interest(i).syndrome,'FragileX')...
    + 6*strcmp(entries_of_interest(i).syndrome,'Progeria')...
    + 7*strcmp(entries_of_interest(i).syndrome,'Treacher-Collins')...
    + 8*strcmp(entries_of_interest(i).syndrome,'Williams')...
    + 9*strcmp(entries_of_interest(i).syndrome,'Control');
end
input_id = 1:length(entries_of_interest); %entries_of_interest(i).id;
toc
