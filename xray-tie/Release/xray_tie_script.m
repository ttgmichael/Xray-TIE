%This image is healthy 0 from the norm folder

%Uncomment one or the other but not both depending on which one
%you want to see
%I2 = imread('.\norm\20131116_192634_GWSS_D5_Healthy_0.tif'); %rectangular image
<<<<<<< HEAD
I2 = imread('.\input\xray_tie_input_2.tif'); %square power of 2 image
%I2 = ones(400,400);
%remove the glue part (approximately the last 100 rows of the image)
[rows,cols] = size(I2);
%I2_cropped = I2(1:(rows-100),1:2560); %for rectangular image
 I2_cropped = I2; %for square power of two image, no glue removal
% figure();
% imagesc(I2_cropped);
% colormap gray;
% colorbar;
% title('Original Image');

%Parameters to projected thickness function
IinVal = 1.0; %incident intensity
Mag = 1.0; %magnification
R2 = 25; %in millimeters (defocus distance)
indexofrefractionbeta = 3.4789e-10; %unitless
xrayenergy = 20000; %eV
wavelength = 4.135667516e-15*299792458/xrayenergy; %(m)
mu = 4*pi*indexofrefractionbeta/(wavelength*1000) ; %mm-1
%mu = 0.028; %in millimeters^-1 (linear attenuation coefficient)
delta = .000001;  %unitless (change in refractive index)
ps = .00325;  %in millimeters (pixel size)
reg = 0.00001; %unitless (regularization)
=======
I2 = imread('.\input\xray_tie_input_0.tif'); %square power of 2 image

%remove the glue part (approximately the last 100 rows of the image)
[rows,cols] = size(I2);
%I2_cropped = I2(1:(rows-100),1:2560); %for rectangular image
I2_cropped = I2; %for square power of two image, no glue removal
figure();
imagesc(I2_cropped);
colormap gray;
colorbar;
title('Original Defocused Image');

%Parameters to projected thickness function
IinVal=1; %incident intensity
Mag = 1.0; %magnification
R2 = 30; %in millimeters (defocus distance)
mu = 0.00828; %in millimeters^-1 (linear attenuation coefficient)
delta = .0001;  %unitless (change in refractive index)
ps = .00325;  %in millimeters (pixel size)
reg = 0.1; %unitless (regularization)
>>>>>>> bf31d68f9ba4f08251c11785798d5cf592377dae

%Above values were obtained by the information below:
%incident intensity: 1 (since the images I gave you were normalized using brightfield images taken without the sample present, I think this should be right)
%linear attenuation coefficient: 0.00828 mm^-1
%pixel size: .00325 mm
%magnification: 2x
%distance of defocused image: unsure, but on the order of 30-40 mm

<<<<<<< HEAD
%Now call the function with te above parameters and output the thickness
output1 = test_xray_tie(I2_cropped, IinVal, Mag, R2, mu, delta, ps, reg);
output2 = xray_tie(I2_cropped, IinVal, Mag, R2, mu, delta, ps, reg);
output3 = output1-output2;
figure();
imagesc(output1);
colormap gray;
colorbar;
title('-Log scaled Image');
%hist(output1(:),100000);
figure();
imagesc(output2);
colormap gray;
colorbar;
title('Output Image (Projected Thickness)');
%hist(output2(:),100000);
figure();
imagesc(output3);
colormap gray;
colorbar;
title('Divided output');
%hist(output3(:),100000);
=======


%Now call the function with te above parameters and output the thickness
output = xray_tie(I2_cropped, IinVal, Mag, R2, mu, delta, ps, reg);
figure();
imagesc(output);
colormap gray;
colorbar;
title('Output Image (Projected Thickness)');
>>>>>>> bf31d68f9ba4f08251c11785798d5cf592377dae
