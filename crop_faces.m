function input_data = crop_faces(entries_of_interest,num_to_load)
%function to crop and rotate faces
close all
tic;
if nargin ~=2
    load('entries_of_interest_ma.mat');
    num_to_load = 44;
end
scale = [145, 131];
input_data = zeros(scale(1)*scale(2),length(entries_of_interest));
for i=1:length(entries_of_interest)
num_to_load = i
current_im = imread(entries_of_interest(num_to_load).imagePath);
%cropped_im = imcrop(current_im,entries_of_interest(num_to_load).face_coordinates);

%now rescale and rotate according to canonical transform
%J = imresize(cropped_im, [224,224]); 
%imshow(J);

 if num_to_load==109
     figure(5), imshow(current_im)
 end
% hold on
% plot(entries_of_interest(num_to_load).ground_truth_points(:,1),entries_of_interest(num_to_load).ground_truth_points(:,2),'bo')
figure
[ proc_image ] = pre_proc_test( current_im, entries_of_interest(num_to_load).ground_truth_points, 1);
s = size(proc_image)
input_data(:,i) = reshape(proc_image,s(1)*s(2),1);

end
toc
