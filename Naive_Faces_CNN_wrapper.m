function Naive_Faces_CNN_wrapper(entries_of_interest,imdb,varargin)
% Based on Part 4 of the VGG CNN practical
%Make sure in correct directory
%setup ;

% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------
if isempty(entries_of_interest)
% Load dataset
fprintf('Loading entries and dataset\n');
[imdb.images.data, imdb.images.label, imdb.images.id] = crop_faces;
rr = rand(1,size(imdb.images.data,3));
imdb.images.set = 1*(rr<0.8)+2*(rr>=0.8);
elseif isempty(imdb)
    % Load dataset
fprintf('Loading dataset\n');
[imdb.images.data, imdb.images.label, imdb.images.id] = crop_faces(entries_of_interest) ;
rr = rand(1,size(imdb.images.data,3));
imdb.images.set = 1*(rr<0.8)+2*(rr>=0.8);
else
    fprintf('Data already loaded\n');
end
% % Visualize some of the data 
figure(1) ; clf ; colormap gray ;
subplot(1,2,1) ;
vl_imarraysc(imdb.images.data(:,:,imdb.images.label==1 & imdb.images.set==1)) ;
axis image off ;
title('training examples for ''angelman''') ;

subplot(1,2,2) ;
vl_imarraysc(imdb.images.data(:,:,imdb.images.label==1 & imdb.images.set==2)) ;
axis image off ;
title('validation examples for ''angelman''') ;

% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
fprintf('Initializing CNN\n');
net = initializeNaiveFacesCNN() ;

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

trainOpts.batchSize = 100 ;
trainOpts.numEpochs = 150 ;
trainOpts.continue = true ;
trainOpts.useGpu = false ;
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = 'Facial_gestalt/naive-experiment' ;
trainOpts = vl_argparse(trainOpts, varargin);

% Take the average image out
imageMean = mean(imdb.images.data(:)) ;
%subtract average
imdb.images.data = imdb.images.data - imageMean ;

% Convert to a GPU array if needed
if trainOpts.useGpu
    error('gpu not yet supported');
  imdb.images.data = gpuArray(imdb.images.data) ;
end

fprintf('Training CNN\n');
% Call training function in MatConvNet
[net,info] = cnn_train_adapted(net, imdb, @getBatch, 'batchSize',trainOpts.batchSize,...
    'numEpochs', trainOpts.numEpochs, 'continue', trainOpts.continue,...
    'learningRate', trainOpts.learningRate,...
    'expDir', trainOpts.expDir) ; 

% Move the CNN back to the CPU if it was trained on the GPU
if trainOpts.useGpu
  error('gpu not yet supported');  
  net = vl_simplenn_move(net, 'cpu') ;
end

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save('Facial_gestalt/naive-experiment/NaiveFacesCNN.mat', '-struct', 'net') ;

% -------------------------------------------------------------------------
% Part 4.4: visualize the learned filters
% -------------------------------------------------------------------------

figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.filters),'spacing',2)
axis equal ; title('filters in the first layer') ;

% -------------------------------------------------------------------------
% Part 4.5: apply the model
% -------------------------------------------------------------------------

% Load the CNN learned before
%net = load('Facial_gestalt/naive-experiment/NaiveFacesCNN.mat') ;
%net = load('data/chars-experiment/charscnn-jit.mat') ;

%Now apply the model to some kind of example test image/data set

%If initialis all working, then can also try applying to extension with jitter 
% % -------------------------------------------------------------------------
% % Part 4.6: train with jitter
% % -------------------------------------------------------------------------
% 
% trainOpts.batchSize = 100 ;
% trainOpts.numEpochs = 15 ;
% trainOpts.continue = true ;
% trainOpts.learningRate = 0.001 ;
% trainOpts.expDir = 'tutorial/chars-jit-experiment' ;
% 
% % Initlialize a new network
% net = initializeCharacterCNN() ;
% 
% % Call training function in MatConvNet
% [net,info] = cnn_train_adapted(net, imdb, @getBatchWithJitter,  'batchSize',trainOpts.batchSize,...
%     'numEpochs', trainOpts.numEpochs, 'continue', trainOpts.continue,...
%     'learningRate', trainOpts.learningRate,...
%     'expDir', trainOpts.expDir) ;
% 
% % Move the CNN back to CPU if it was trained on GPU
% if trainOpts.useGpu
%   net = vl_simplenn_move(net, 'cpu') ;
% end
% 
% % Save the result for later use
% net.layers(end) = [] ;
% net.imageMean = imageMean ;
% save('tutorial/chars-experiment/charscnn-jit.mat', '-struct', 'net') ;
% 
% % Visualize the results on the sentence
% figure(4) ; clf ;
% decodeCharacters(net, imdb, im, vl_simplenn(net, im)) ;

end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,batch);
im = 256 * reshape(im, 125, 125, 1, []) ;
labels = imdb.images.label(1,batch);
end
% --------------------------------------------------------------------
% function [im, labels] = getBatchWithJitter(imdb, batch)
% % --------------------------------------------------------------------
% im = imdb.images.data(:,:,batch) ;
% labels = imdb.images.label(1,batch) ;
% 
% n = numel(batch) ;
% train = find(imdb.images.set == 1) ;
% 
% sel = randperm(numel(train), n) ;
% im1 = imdb.images.data(:,:,sel) ;
% 
% sel = randperm(numel(train), n) ;
% im2 = imdb.images.data(:,:,sel) ;
% 
% ctx = [im1 im2] ;
% ctx(:,17:48,:) = min(ctx(:,17:48,:), im) ;
% 
% dx = randi(11) - 6 ;
% im = ctx(:,(17:48)+dx,:) ;
% sx = (17:48) + dx ;
% 
% dy = randi(5) - 2 ;
% sy = max(1, min(32, (1:32) + dy)) ;
% 
% im = ctx(sy,sx,:) ;
% 
% % Visualize the batch:
% % figure(100) ; clf ;
% % vl_imarraysc(im) ;
% 
% im = 256 * reshape(im, 32, 32, 1, []) ;
% end
% 
% 
