function outImg = linearizeImage(img, p)

sz = size(img);
RGB = reshape(img,prod(sz)/3,3);
RGB(:,1)  = p.r(RGB(:,1));
RGB(:,2)  = p.g(RGB(:,2));
RGB(:,3)  = p.b(RGB(:,3));

outImg = reshape(RGB,sz);
outImg(outImg<0)=0;
outImg(outImg>1)=1;

