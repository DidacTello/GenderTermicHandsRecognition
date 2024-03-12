
function [NormalizedHandAppearance,NormalizedHandShape,NormalizedPalm]=prepare_one_hand(file,hand_type,texture_to_shape_ratio,out_th);
% main function which performs normalization of an input hand
%
% USAGE
% [NormalizedHandAppearance,NormalizedHandShape,NormalizedPalm]=prepare_one_hand(file,hand_type,texture_to_shape_ratio);
%
% INPUTS
% file: string for full image filename
% hand_type: string to indicate whether the input hand is left hand or
% right hand, two choices: 'left' and 'right'
% texture_to_shape_ratio: A positive number, preferably between 0 and 1, that increases the texture component of
% the Normalized Hand Appearance. The default value is 0.3
%
% OUTPUTS
% NormalizedHandAppearance: hand's pose-normalized and resized gray level intensity image with
% size 200x200
% NormalizedHandShape: hand's pose normalized and resized binary shape image with size
% 200x200
% NormalizedPalm: 100x100 palm image of the hand, anatomically extracted and
% scaled to a uniform size

switch nargin
    case 2,                   
        texture_to_shape_ratio=0.3;
end

handguide{1}=[140 106 94 82 70]*pi/180;
handguide{2}=[145 150];
handguide{3}=[148 -56;36 -56;29 -19;36 18;52 50];
tsr=1;
base_points=[110.750000000000,26.7500000000000;80.7500000000000,58.7499999999999;80.7500000000000,66.7499999999999;66.7500000000000,91.2500000000000;75.7500000000000,137.750000000000;152.250000000000,103.250000000000;151.750000000000,79.2499999999999;158.250000000000,48.2499999999999;172.750000000000,190.250000000000;173.250000000000,205.750000000000;193.750000000000,141.750000000000;260.750000000000,52.2499999999999;264.250000000000,131.250000000000;];
input_points=[199,119;142,178;143,193;116,239;134,326;276,266;275,220;288,161;312,423;312,450;352,337;479,169;483,317;];

meano = mean(input_points - base_points);
meanx = uint16(meano(:,1));
meany = uint16(meano(:,2));

I=file;
[height,width,channel]=size(I);
sizefactor=round(10*526/height)/10;

out_th=double(out_th);
out_th=imresize(out_th,'OutputSize',[540 704]);
[height_th,width_th,channel_th]=size(out_th);
out_th=imresize(out_th,sizefactor/height_th*height,'bicubic');
% Shape_th=segment(out_th);
% [Shape_th,out_th]=morphcorrect(Shape_th,out_th);

% %                                                                     for i=1:(size(out_th,1)),
% % 
% %                                                                          for j=1:(size (out_th,2)),
% %                                                                             if (meanx>0&&meany>0)
% %                                                                                 img2(i+meanx,j+meany)=out_th(i,j);
% % 
% %                                                                             elseif (distX<0&&meany<0)
% %                                                                                 img2(i-meanx,j-meany)=out_th(i,j);
% %                                                                             elseif (meanx>0&&meany<0)
% %                                                                                 img2(i+meanx,j-meany)=out_th(i,j);
% %                                                                             else
% %                                                                                 img2(i-meanx,j+meany)=out_th(i,j);
% %                                                                             end
% %                                                                           end
% %                                                                     end
% %                                                                     out_th = img2;

mytform=cp2tform(input_points,base_points, 'similarity');
out_th = imtransform(out_th, mytform);

[height_th,width_th,channel_th]=size(out_th);
out_th=imresize(out_th,sizefactor/height_th*height,'bicubic');

 for i=1:(size(out_th,1)),

  for j=1:(size (out_th,2)),
      if (meanx>0&&meany>0)
         img2(i+meanx,j+meany)=out_th(i,j);

       elseif (distX<0&&meany<0)
          img2(i-meanx,j-meany)=out_th(i,j);
       elseif (meanx>0&&meany<0)
           img2(i+meanx,j-meany)=out_th(i,j);
       else
           img2(i-meanx,j+meany)=out_th(i,j);
      end
      end
end
out_th = img2;

[j,k]=size(out_th);
j=round((740-j)/2);
k=round((904-k)/2);
L=[j,k];
out_th=wextend(2,'zpd',out_th,L);                      
out_th=imresize(out_th,'OutputSize',[740 904]);
meanx=1; meany=110;
for j=1:(size (out_th,2)),
      if (meanx>0&&meany>0)
         img2(i+meanx,j+meany)=out_th(i,j);

       elseif (distX<0&&meany<0)
          img2(i-meanx,j-meany)=out_th(i,j);
       elseif (meanx>0&&meany<0)
           img2(i+meanx,j-meany)=out_th(i,j);
       else
           img2(i-meanx,j+meany)=out_th(i,j);
      end
end
out_th = img2;
[j,k]=size(out_th);
j=round((740-j)/2);
k=round((904-k)/2);
L=[j,k];
out_th=wextend(2,'zpd',out_th,L);                      
out_th=imresize(out_th,'OutputSize',[740 904]);
if channel==3
    I=rgb2gray(I);   
end
if sizefactor==1;
    I=double(I);
else
    I=imresize(double(I),sizefactor,'bicubic');
end;

if isequal(hand_type,'right');
    for c=1:channel
        I(:,:,c)=fliplr(I(:,:,c));
    end
end;

Shape=segment(I);
[Shape,I]=morphcorrect(Shape,I);
I=zeropad(I,100,100);
Shape=zeropad(Shape,100,100);
contour=extractcontour(Shape);
[contour,tips,valleys]=findextremities(contour);
F=fingerinfo(contour,tips,valleys);
Shape=removeringcavities(Shape,contour,F);
contour=extractcontour(Shape);
[contour,tips,valleys]=findextremities(contour);
F=fingerinfo(contour,tips,valleys);
[H,F]=handinfo(F,contour,Shape);
[HAND,Shape]=decomposehand(Shape,F,H,contour,handguide);
Appearance=out_th;
Appearance=removeringcolor(HAND,Shape,Appearance);
Appearance=ilumcorrect(Appearance,Shape);
Appearance2=normalizeappearance(Appearance,Shape,tsr);
[NormalizedHandAppearance,NormalizedHandShape,NormalizedPalm]=synthesizehand(HAND,Appearance2,Shape,handguide);
NormalizedHandAppearance=NormalizedHandShape+texture_to_shape_ratio*NormalizedHandAppearance;
