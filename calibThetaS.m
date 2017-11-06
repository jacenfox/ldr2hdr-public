%% match ThetaS(linearized) to Canon(as in skydb)
clear
sz = [1000, nan];
%% Load calibration results for two cameras
calibMatThetaS = load('data/calibCanonThetaS.mat');

%% load files
fn_ricoh = './examples/ricoh.jpg';
ricoh = im2double(imread(fn_ricoh));
ricoh = imresize(ricoh, sz);

%% find the relative EV for the calibrate data
RicohEV = 14.6439;
calibMat = calibMatThetaS;
CanonEV = calibMat.CanonEV;

[T, scale] = rescaleColorMat(calibMat.T, calibMat.CanonEV, calibMat.RicohEV, CanonEV, RicohEV);
T = inv(T).* scale;
p = calibMat.p;

%% align & match the full image
ricoh_linear = linearizeImage(ricoh, p);
ricoh_color = colorMatch(ricoh_linear, T);

%% write back
imwrite(ricoh_color, './examples/ricoh_calib.jpg', 'Quality', 100);


