%Read header of CASSIE_SIMULATION.m for additional instructions.

%This is a variant of the Cassie Simulation that allows for running many
%trajectories at once. 

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

%--------------------------------------------------------------------------

global runInfo;

runInfo.dataset = 'CASSIE_XZ_UNFILTERED';%'CASSIE_YZ_UNFILTERED'; %Choose dataset, must have variable t, xX, XY, vX, vY as horizontal rows
load(runInfo.dataset);

runInfo.runType = 'Both'; %Images, Opti, or Both - Run just image set creation, optimization, or both
runInfo.imageSetSaveName = [runInfo.dataset, '_TEMP_SAVE']; %Name of output file for image set

runInfo.num_images = 1; %Number of images in order used from test_space
runInfo.num_samples = 3; %Number of captures in each iteration
runInfo.secondsStart = 25; %Time start of segment
runInfo.secondsEnd = 25+1;%123.1; %Time end of segment
runInfo.secondsRange = 1; %Length of segment
runInfo.blur_ratio = 0.00002; %0.00078 m/pixel

runInfo.Te = 1/30; %Exsposure time
runInfo.gaIterCap = 5000; %Maximum optimization iterations
runInfo.maxStallTime = 10; %Time with no score improvement until optimization stops
runInfo.maxOptiScore = 3; %Maximum score for OCR comparison (No text or high score)
      
runInfo.tesseractExeDir = 'C:\Program Files\Tesseract-OCR\'; %Tesseract Directory
runInfo.filepath = 'Z:\OneDrive\My Stuff\DOCUMENTS\School\Grad School\Spring & Summer 2021\Research\Alex NSF Robotic Vision 2021\IRIM-CASSIE\test_space\'; %Filepath of the test_space directory
runInfo.outputImageName = 'test_image000.jpg'; %Name of OCR iteration file 
runInfo.outputName = 'ocr_result000'; %OCR output file name
runInfo.inputImageName = 'test_image'; %Prefix of test set images
runInfo.inputImageType = 'jpg'; %Test set image file type
runInfo.benchTextFile = 'image_text.txt'; %Name of benchmark text file

num_trials = floor((runInfo.secondsEnd - runInfo.secondsStart)./runInfo.secondsRange);

all_input = [];
all_score = [];
all_t = [];
all_xX = [];
all_vX = [];
all_xY = [];
all_vY = [];

for i = 1:1:num_trials
    seg_start = i.*runInfo.secondsRange+runInfo.secondsStart-runInfo.secondsRange;
    seg_end = i.*runInfo.secondsRange+runInfo.secondsStart;
    
    ind_start = find(round(t,4) == round(seg_start,4));
    ind_end = find(round(t,4) == round(seg_end,4));

    segment = ind_start:1:ind_end;
    t_seg = t(segment);
    xX_seg = xX(segment);
    vX_seg = vX(segment);
    xY_seg = xY(segment);
    vY_seg = vY(segment);
    [input_list, score_list] = run_One(t_seg, xX_seg, vX_seg, xY_seg, vY_seg);
    %CHANGE THIS LINE^^^ with run_... to change type of function run
    
    [score, ind] = min(score_list);
    input = input_list(ind,:);
    
    all_input = [all_input, input'];
    all_score = [all_score, score'];
    all_t = [all_t, t_seg'];
    all_xX = [all_xX, xX_seg'];
    all_vX = [all_vX, vX_seg'];
    all_xY = [all_xY, xY_seg'];
    all_vY = [all_vY, vY_seg'];
end

clear i ind ind_end ind_start input input_list score score_list seg_end seg_start segment t t_seg vX vX_seg vY vY_seg xX xX_seg xY xY_seg
clc;
sound(15.*sin(1:15000));
save(runInfo.imageSetSaveName)
%plot_single_trial(ind, runInfo, all_input, all_score, all_t, all_xX, all_vX, all_xY, all_vY)