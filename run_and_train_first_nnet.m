%script to run nnet that crashes everything currently

input_data = crop_faces;
C = mat2cell(input_data, [size(input_data,1)], [ones(1, size(input_data,2))]);
net2 = feedforwardnet(10);
[net2,tr] = train(net2,C{1:10},'useParallel','yes','CheckpointFile','MyCheckpoint','CheckpointDelay',120);
