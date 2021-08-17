% Alex Zabaldo Summer 2021
%--------------------------------------------------------------------------
% TO RUN:
% You will need to instal Tesseract, make sure you can run it from the cmd line as a test
% Install any needed packages
% Make sure to change the directories below that map to the Tesseract Directory and the test_space directory
% To change images used, change them all to a name of choice with "(x)" where x is the order number, and create a file listing text for each image on each line
% All test parameters can be change in the area below
% Input data must be in the form of t, xX, xY, vX, vY
% Plots can be created by running plot_sim_ouput.m directly agfter a full run
%--------------------------------------------------------------------------
clear global runInfo;
clear global OCR_benchmark;
clear global all_OCRText;
clear global all_Bound;
clear global all_Conf;
clear global all_Images;
clear global all_psf;
clear global input_list;
clear global score_list;
clear global best_input;
clear; clc;
global runInfo
global OCR_benchmark;
global all_OCRText; 
global all_Bound;
global all_Conf;
global all_Images;
global all_psf;
global input_list;
global score_list;

%--------------------------------------------------------------------------
%SIMULATION SETTINGS AND VALUES

runInfo.dataset = 'CASSIE_XZ_UNFILTERED';%'CASSIE_YZ_UNFILTERED'; %Choose dataset, must have variable t, xX, XY, vX, vY as horizontal rows
load(runInfo.dataset);

runInfo.runType = 'Both'; %Images, Opti, or Both - Run just image set creation, optimization, or both
runInfo.imageSetSaveName = [runInfo.dataset, '_TEMP_SAVE']; %Name of output file for image set

runInfo.num_images = 27; %Number of images in order used from test_space
runInfo.num_samples = 3; %Number of captures in each iteration
runInfo.secondsStart = 48; %Time start of segment
runInfo.secondsRange = 0.7505; %Length of segment
runInfo.blur_ratio = 0.00002; %0.00078 m/pixel

runInfo.Te = 1/60; %Exsposure time
runInfo.gaIterCap = 5000; %Maximum optimization iterations
runInfo.maxStallTime = 10; %Time with no score improvement until optimization stops
runInfo.maxOptiScore = 3; %Maximum score for OCR comparison (No text or high score)
      
runInfo.tesseractExeDir = 'C:\Program Files\Tesseract-OCR\'; %Tesseract Directory
runInfo.filepath = 'Z:\OneDrive\My Stuff\DOCUMENTS\School\Grad School\Spring & Summer 2021\Research\CASSIE\test_space\'; %Filepath of the test_space directory
runInfo.outputImageName = 'test_image000.jpg'; %Name of OCR iteration file 
runInfo.outputName = 'ocr_result000'; %OCR output file name
runInfo.inputImageName = 'test_image'; %Prefix of test set images
runInfo.inputImageType = 'jpg'; %Test set image file type
runInfo.benchTextFile = 'image_text.txt'; %Name of benchmark text file

%--------------------------------------------------------------------------
switch runInfo.runType 
    case 'Images'
        t = t-t(1)+(t(2)-t(1));
        dt = t(3)-t(2);
        len = runInfo.secondsRange./dt;
        start = find(round(t,6)==runInfo.secondsStart);
        segment = start+1:(start+len+1);
        t = t(segment);
        xX = xX(segment);
        vX = vX(segment);
        xY = xY(segment);
        vY = vY(segment);
        runInfo.t = t;
        runInfo.xX = xX;
        runInfo.vX = vX;
        runInfo.xY = xY;
        runInfo.vY = vY;
        runAllImages(t, xX, vX, xY, vY);
        clear dt len segment start t vX vY xX xY
        save(runInfo.imageSetSaveName)
    case 'Opti'
        runOptimize()
    case 'Both'
        t = t-t(1)+(t(2)-t(1));
        dt = t(3)-t(2);
        len = runInfo.secondsRange./dt;
        start = find(round(t,6)==runInfo.secondsStart);
        segment = start+1:(start+len+1);
        t = t(segment);
        xX = xX(segment);
        vX = vX(segment);
        xY = xY(segment);
        vY = vY(segment);
        runInfo.t = t;
        runInfo.xX = xX;
        runInfo.vX = vX;
        runInfo.xY = xY;
        runInfo.vY = vY;
        runAllImages(t, xX, vX, xY, vY);
        clear dt len segment start t vX vY xX xY
        save(runInfo.imageSetSaveName)
        runOptimize();
end
%--------------------------------------------------------------------------

function [] = runOptimize()
%This function starts the optimization of the existing image set
%infromation using integer inputs that represent indexed captures along the
%trajectory

global runInfo
global OCR_benchmark;
global all_OCRText;
global all_Bound;
global all_Conf;
global all_Images;
global all_psf;
global input_list;
global score_list;
input_list = [];
score_list = [];

%Fills OCRText spaces where no text was found with spacers
num_segments = length(all_OCRText{1});
for i = 1:1:length(all_OCRText)
    if length(all_OCRText{i}) < num_segments
        current = length(all_OCRText{i});
        diff = num_segments - current;
        for j = 1:1:diff
            all_OCRText{i}(current+j) = "";
        end
    end
end

IntCon = 1:1:runInfo.num_samples; %makes sure all inputs are integers
options = optimoptions('ga', 'display', 'iter', 'MaxStallTime', runInfo.maxStallTime);
ga(@OptiFunc, runInfo.num_samples, [],[],[],[],ones(1,runInfo.num_samples),num_segments.*ones(1,runInfo.num_samples), [], IntCon, options);

end

function score = OptiFunc(input)
%This function outputs the score of comparing the combined string to the
%benchmark string for the indexed trajectory segments 

global runInfo
global OCR_benchmark;
global all_OCRText;
global all_Bound;
global all_Conf;
global all_Images;
global all_psf;
global input_list;
global score_list;

%Get the text, confidence, and boundaries that corresponf to the given
%input
select_Conf = {};
select_OCRText = {};
for i = 1:1:length(all_OCRText)
    for j = 1:1:length(input)
        if j == 1
            select_Conf{i} = all_Conf{i}(input(j),:);
            select_OCRText{i} = all_OCRText{i}(input(j),:);
            select_Bound{i} = all_Bound{i}(:,:,input(j));
        else
            select_Conf{i} = [select_Conf{i}; all_Conf{i}(input(j),:)];
            select_OCRText{i} = [select_OCRText{i}; all_OCRText{i}(input(j),:)];
            select_Bound{i} = cat(3, select_Bound{i}, all_Bound{i}(:,:,input(j)));
        end
    end
end

%find the combined string for each image
strings = combineStrings(select_OCRText, select_Conf, select_Bound, length(input));

%Find score for each image
score = [];
for i = 1:1:length(all_OCRText)
    changes = editDistance(OCR_benchmark{i}, strings{i});
    cap = runInfo.maxOptiScore;
    if (strlength(strings{i}) < 1)
        percent = cap;
    else
        percent = changes./strlength(strings{i});
    end
    score = [score, percent];
end
%Final score
score = mean(score);

%Save score and input inforamtion to list
score_list = [score_list; score];
input_list = [input_list; input];
num = length(score_list);

disp('-------------------------------');
disp(['Iteration: ', num2str(num)]);
disp(['Input: ', num2str(input)]);
disp(['Score: ', num2str(score)]);
disp('-------------------------------');

%Stops the function if the Cap is reached
if num == runInfo.gaIterCap
    error('MAXIMUM ITERATIONS REACHED - THIS IS NOT AN ERROR!')
end

end

function [] = runAllImages(t, xX, vX, xY, vY)
%creates segmented trajectories, blur kernels, benchmark text, and OCR text
% as well as bounding boxes and confidence levels for all images

global runInfo
global OCR_benchmark;
global all_OCRText;
global all_Bound;
global all_Conf;
global all_Images;
global all_psf;

%-------------------------Get benchmark text for each image
fullfile = [runInfo.filepath, runInfo.benchTextFile];
fileID = fopen(fullfile,'r');
OCR_benchmark = {};
for i = 1:1:runInfo.num_images
    OCR_benchmark{i} = string(fgetl(fileID));
end
fclose('all');

%-------------------------Determine number and length of segments
num_segments = floor((t(end)-t(1))./runInfo.Te);
resize_len = 10000;
t = linspace(runInfo.Te, t(end), resize_len);
vX = interp1(vX, linspace(1,length(vX), resize_len));
vY = interp1(vY, linspace(1,length(vY), resize_len));
xX = interp1(xX, linspace(1,length(xX), resize_len));
xY = interp1(xY, linspace(1,length(xY), resize_len));
len_segments = floor(length(t)./num_segments);

%-------------------------Split up trajectories
traj_xX = [];
traj_vX = [];
traj_xY = [];
traj_vY = [];
for i = 1:1:num_segments
    traj_xX = [traj_xX; xX(((i-1)*len_segments+1):i*len_segments)];
    traj_vX = [traj_vX; vX(((i-1)*len_segments+1):i*len_segments)];
    traj_xY = [traj_xY; xY(((i-1)*len_segments+1):i*len_segments)];
    traj_vY = [traj_vY; vY(((i-1)*len_segments+1):i*len_segments)];
end

%-------------------------Create blur kernels for all trajectories
filestart = [runInfo.filepath, runInfo.inputImageName, ' ('];
fileend = [').', runInfo.inputImageType];
all_psf = {};
single_psf = {};
for i = 1:1:runInfo.num_images
    I = imread([filestart, num2str(i), fileend]);
	[height, width,~] = size(I);
    for j = 1:1:num_segments
        psf = getPSF(traj_xX(j, :), traj_vX(j, :), traj_xY(j, :), traj_vY(j, :), [width, height], runInfo.blur_ratio);
        single_psf{j} = psf;
    end
    all_psf{i} = single_psf;
    single_psf = [];
end

%-------------------------Create and store deblurred images and pull OCR
%text and boundaries and confidence levels
all_Images = {};
single_Images = [];
all_OCRText = {};
single_OCRText = [];
all_Conf = {};
single_Conf = [];
all_Bound = {};
single_Bound = [];
for i = 1:1:runInfo.num_images
    I = imread([filestart, num2str(i), fileend]);
    for j = 1:1:num_segments
        psf = all_psf{i}{j};
        if psf == 1
            If = rgb2gray(I);
        else
            If = deconvlucy(imnoise(imfilter(rgb2gray(I), psf),'gaussian', 0, 0.001),  psf);
        end
        disp(['Segment ', num2str(j), '/', num2str(num_segments), ', - Image ', num2str(i), '/', num2str(runInfo.num_images), ' Processed']);
        [text, conf, bound] = Tesseract_OCR(If);
        text = string(text);
        conf = [conf, -5.*ones(1, 200 - length(conf))];
        bound = [bound; -5.*ones(200 - size(bound, 1), 4)];
        single_Images = cat(3, single_Images, If);
        single_OCRText = [single_OCRText; text];
        single_Conf = [single_Conf; conf];
        single_Bound = cat(3, single_Bound, bound);
    end
    all_Images{i} = single_Images;
    all_OCRText{i} = single_OCRText;
    all_Conf{i} = single_Conf;
    all_Bound{i} = single_Bound;
    single_Images = [];
    single_OCRText = [];
    single_Conf = [];
    single_Bound = [];
end

%-------------------------Store trajectory info
runInfo.traj_xX = traj_xX;
runInfo.traj_vX = traj_vX;
runInfo.traj_xY = traj_xY;
runInfo.traj_vY = traj_vY;

end

function [text, conf, bound] = Tesseract_OCR(I)
%pulls OCR text, bounding boxes, and character level confidence intervals
%from and image

global runInfo;

% I = ImageProcess(I); %preprocessing

%write input image with known name and location
imwrite(I, [runInfo.filepath, runInfo.outputImageName]);
%run tesseract
cmd = ['"',runInfo.tesseractExeDir, 'tesseract.exe" "', runInfo.filepath, runInfo.outputImageName, '" "', runInfo.filepath, runInfo.outputName, '" --oem 1 -l eng -c hocr_char_boxes=1 hocr'];
[~,~] = system(cmd);

%pull entire output file
fileID = fopen([runInfo.filepath, runInfo.outputName, '.hocr'], 'r');
hocrtext = fscanf(fileID, '%c');
fclose('all');

%use known string to find character lines from output file
ind = strfind(hocrtext, 'x_conf');
cuts = [];
for i = 1:1:length(ind)
    cut = hocrtext(ind(i)-26:ind(i)+18);
    cuts = [cuts; cut];
end

%parse lines to find specific information 
conf = [];
text = [];
bound = [];
for i = 1:1:length(ind)
    cut = cuts(i,:);
    cut_num_ind = isstrprop(cut(1:end-3),'digit');
    num_check = find(cut_num_ind);
    
    cutoff1 = num_check(1);
    cut = cut(cutoff1:end);
    mask_bb = cut == ';';
    cutoffbb = find(mask_bb, 1);
    bb = cut(1:cutoffbb-1);
    cut = cut(cutoffbb+9:end);
    
    cut_num_ind = isstrprop(cut(1:end-3),'digit');
    num_check = find(cut_num_ind);
    cutoff = num_check(end);
    
    num = str2double(cut(1:cutoff));
    
    num_bounds = [];
    for j = 1:1:3
        space = find(bb == ' ', 1);
        temp_num = str2num(bb(1:space-1));
        bb = bb(space+1:end);
        num_bounds = [num_bounds, temp_num];
    end
    temp_num = str2num(bb);
    num_bounds = [num_bounds, temp_num];
    
    ch = cut(cutoff+3);
    conf = [conf, num];
    text = [text, ch];
    bound = [bound; num_bounds];
end

%adjust bounding boxes to different format, filter for overly small and
%large areas
if length(conf > 0)
    bound(:, 3) = bound(:,3) - bound(:,1);
    bound(:, 4) = bound(:,4) - bound(:,2);
    
    area = bound(:,3).*bound(:,4);
    mean_area = mean(area);
    area_factor = 45;
    area_check = area > area_factor.* mean_area | area < mean_area./area_factor;
    space_check = text == ' ';
    erase_check = area_check | space_check';
    conf(erase_check) = [];
    text(erase_check) = [];
    bound(erase_check,:) = [];
end

end

function psf = getPSF(xX, vX, xY, vY, im_size, blur_ratio)
%creates a blur kernel from trajectory information and image information

%--------------------------- %Ksize determination, KsizeX and KsizeY
%represent the total number of pixels crossed in each dimension. 
distX = abs(max(xX)-min(xX));
distY = abs(max(xY)-min(xY));
KsizeX = floor((distX./(im_size(1).*blur_ratio)).*im_size(1))+1;
KsizeY = floor((distY./(im_size(2).*blur_ratio)).*im_size(2))+1;
if ~mod(KsizeX,2)
    KsizeX = KsizeX +1;
end
if ~mod(KsizeY,2)
    KsizeY = KsizeY +1;
end
Ksize = max([KsizeX, KsizeY]);
%---------------------------

if Ksize == 1
    psf = 1; %K is a point, no blur
else
   
    xX = interp1(xX, linspace(1,length(xX), 2.*Ksize), 'spline'); %resample to Ksize
    xY = interp1(xY, linspace(1,length(xY), 2.*Ksize), 'spline');
    vX = interp1(vX, linspace(1,length(vX), 2.*Ksize), 'spline');
    vY = interp1(vY, linspace(1,length(vY), 2.*Ksize), 'spline');
    distX = abs(max(xX)-min(xX));
    distY = abs(max(xY)-min(xY));
    
    xXn = floor((1-(xX-min(xX))./distX).*(Ksize-1))+1; %creates basline positions for the intensity values
    xYn = floor((1-(xY-min(xY))./distY).*(Ksize-1))+1;
    
    xXn = round(xXn./max(xXn).*KsizeX); % rescale for KsizeX and KsizeY
    xYn = round(xYn./max(xYn).*KsizeY);
    
    xX = interp1(xX, linspace(1,length(xX), 2.*Ksize+1), 'spline'); %interp is larger for position due to needing internal difference
    xY = interp1(xY, linspace(1,length(xY), 2.*Ksize+1), 'spline');
    
    intensity = [];
    for i = 2:1:length(vX)+1
        intensity(i-1) = sqrt(((xX(i)-xX(i-1))./vX(i-1)).^2 + ((xY(i)-xY(i-1))./vY(i-1)).^2); %calculate intesity at each point based on velocity and distance traveled in each direction
    end
    intensity = intensity./sum(intensity); %normalize
    [sorted, ind] = sort(intensity); %sorting to largest intensity value
    
    xXnS = xXn(ind); %matching position order to intesity values order 
    xYnS = xYn(ind);
    
    center = Ksize; %fix positions to center
    Xcenter = center - xXnS(end);
    Ycenter = center - xYnS(end);
    xXnS = xXnS + Xcenter;
    xYnS = xYnS + Ycenter;
    
    
    psf = zeros(2.*Ksize); %place values
    for i = 1:1:length(xXnS)
        psf(xYnS(i), 2.*Ksize-xXnS(i)) = sorted(i) + psf(xYnS(i), 2.*Ksize-xXnS(i));
    end
    
    psf = psf./sum(sum(psf)); %normalize
    
    psf_test = psf >0; %cut off excess zeros based on most outlying value
    psf_testX = sum(psf_test, 2);
    psf_testY = sum(psf_test, 1);
    Xcoord = abs(find(psf_testX)-center);
    Ycoord = abs(find(psf_testY)-center);
    radius = max([Xcoord', Ycoord]);
    psf = psf(center-radius:center+radius,center-radius:center+radius);
end

end

function strings = combineStrings(select_strings, select_conf, select_bound, num_samples)
%This function takes the input strings and their confidence levels and
%boundary boxes and combines them based on boundary box overlap and
%a character histogram

true_num_samples = num_samples;
strings = {};

for i = 1:1:length(select_strings)
    for t = length(select_strings{i}):-1:1 %removes samples without any text read
        if  select_strings{i}(t) == ""
            num_samples = num_samples - 1;
            select_conf{i}(t,:) = [];
            select_bound{i}(:,:,t) = [];
        end
    end
    
    matches = [];
    mapping = [];
    for j = 1:1:num_samples
        for k = j+1:1:num_samples
            
            bound1 = select_bound{i}(:,:,j);
            conf1 = select_conf{i}(j,:);
            bound2 = select_bound{i}(:,:,k);
            conf2 = select_conf{i}(k,:);
            
            cutoff1 = find(conf1 ==-5 ,1)-1;%fixes sizes of bounds
            cutoff2 = find(conf2 ==-5 ,1)-1;
            bound1 = bound1(1:cutoff1,:);
            bound2 = bound2(1:cutoff2,:);
            
            fail_bound1 = (bound1(:,3) <= 0) | (bound1(:,4) <= 0); %fixing condition for zero size bounds
            fail_bound2 = (bound2(:,3) <= 0) | (bound2(:,4) <= 0);
            bound1(fail_bound1,3:4) = 1;
            bound2(fail_bound2,3:4) = 1;
            
            if length(bound1) == 0 || length(bound2) == 0
                compared1 = 0;
                compared2 = 0;
            else
                overlap = bboxOverlapRatio(bound1, bound2);
                
                overlap(overlap < 0.2) = 0; %remove minor overlaps
                [~, compared1] = max(overlap, [], 1);
                [~, compared2] = max(overlap, [], 2);
                compared2 = compared2';
                
                sum1 = sum(overlap,1);
                sum2 = sum(overlap,2);
                
                mask1 = ~(sum1 == 0);
                mask2 = ~(sum2 == 0);
                mask2 = mask2';
                
                compared1 = compared1.*mask1;
                compared2 = compared2.*mask2;
                
            end
            compared1 = [compared1, -5.*ones(1, 200 - length(compared1))];
            compared2 = [compared2, -5.*ones(1, 200 - length(compared2))];
            connections = [j,k];
            matches = [matches; connections; flip(connections)];
            mapping = [mapping; compared1; compared2]; %mapping and matching represent character level overlaps and likely pairs of characters
        end
    end
    
    [h, l] = size(mapping); %create character histogram
    string_hist = string.empty(0,l);
    conf_hist = zeros(h,l);
    for x = 1:1:h
        string_num = matches(x,1);
        temp_char = char(select_strings{i}(string_num));
        temp_conf = select_conf{i}(string_num,:);
        for y = 1:1:l
            ind = mapping(x, y);
            if ind == -5 || ind == 0
                string_hist(x,y) = "";
            else
                if length(temp_char) == 0 || length(temp_char) < ind
                    string_hist(x,y) = '';
                    conf_hist(x,y) = -5;
                else
                    string_hist(x,y) = string(temp_char(ind));
                    conf_hist(x,y) = temp_conf(ind);
                end
                
            end
        end
    end
    
    single_string = []; %collapse histogram to form string
    final_string = "";
    for z = 1:1:l
        column_string = string_hist(:,z);
        column_conf = conf_hist(:,z);
        
        letters = unique(column_string);
        letter_score = zeros(size(letters));
        for p = 1:1:length(letters)
            letter_mask = column_string == letters(p);
            letter_score(p) = sum(column_conf(letter_mask));
        end
        [~, chosen] = max(letter_score);
        chosen_letter = letters(chosen);
        single_string = [single_string, chosen_letter];
        
        final_string = [];
        for q = 1:1:length(single_string)
            final_string = [final_string, char(single_string(q))];
        end
        final_string = string(final_string);
    end

    strings{i} = final_string;
    num_samples = true_num_samples;   
end
end

function I = ImageProcess(I)

I = imbinarize(I, 'adaptive');
I = imclose(I, strel('disk', 4)); 

end