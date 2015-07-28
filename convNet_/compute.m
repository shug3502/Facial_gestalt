%  Copyright (c) 2015, Omkar M. Parkhi 
%  All rights reserved.

function desc = compute(obj, inputImg, box, varargin)

%inputImg: Full size image before cropping face out
%box: Face detection box [x1 y1 x2 y2];

opts.imSize = 256;
opts = vl_argparse(opts,varargin);

%% Get the face crop from the input image     
     bw = box(3)-box(1); bh = box(4)-box(2);
     cx = box(1)+bw/2; cy = box(2)+bh/2;
     %ny1 = cy-1.5*bh; ny2 = cy+0.75*bh;
     %nx1 = cx-0.75*bw;nx2 = cx+0.75*bw;
     
     ny1 = cy-1.1*bh; ny2 = cy+0.75*bh;
     nx1 = cx-0.75*bw;nx2 = cx+0.75*bw;
     
     nbox = floor([nx1 ny1 nx2 ny2]);
     nbox = max([1 1 1 1; nbox]);
     nbox = min([size(inputImg,2) size(inputImg,1) size(inputImg,2) size(inputImg,1);nbox]);
     inputImg = inputImg(nbox(2):nbox(4),nbox(1):nbox(3),:);
 
     if(size(inputImg,1)<size(inputImg,2))
         inputImg = imresize(inputImg,[opts.imSize NaN],'bicubic');
     elseif(size(inputImg,1)>size(inputImg,2))
         inputImg = imresize(inputImg,[NaN opts.imSize],'bicubic');
     else
         inputImg = imresize(inputImg,[opts.imSize opts.imSize],'bicubic');
     end
     inputImg = single(inputImg);
 

sy =  size(obj.net.normalization.averageImage,1);
sx = size(obj.net.normalization.averageImage,2);

diffY = size(inputImg,1)-sy;
diffX = size(inputImg,2)-sx;

desc = zeros(4096,1);
cropx = size(inputImg,2)-sx+1;
cropy = size(inputImg,1)-sy+1;


cx = size(inputImg,2)/2;
cy = size(inputImg,1)/2;

%Tiled crop
cropDim = [1 1;...
                 cropx 1;...
                 1 cropy;...
                 cropx cropy;...
                 floor(cx-(sx/2)) floor(cy-(sy/2))];



count = 0 ;
for i=1:size(cropDim,1)
        
        x = cropDim(i,1);
        y = cropDim(i,2);
        faceCrop = inputImg(y:y+sy-1,x:x+sx-1,:);
        faceCrop = faceCrop - obj.net.normalization.averageImage;
        res   = vl_simplenn(obj.net,faceCrop,[],[],'disableDropout',true);
        desc = desc + squeeze(res(end).x);
        count = count + 1;
        
        faceCrop = inputImg(y:y+sy-1,x:x+sx-1,:);

	%Horizontal Flip
        faceCrop = flipdim(faceCrop, 2);
        
        faceCrop = faceCrop - obj.net.normalization.averageImage;
        res   = vl_simplenn(obj.net,faceCrop,[],[],'disableDropout',true);
        
        
        desc = desc + squeeze(res(end).x);
        count = count + 1;
end

if(count>0)
    desc = desc./count;
end

    
end
