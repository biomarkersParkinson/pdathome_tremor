% #########################################################################

% This code was used to train and validate the tremor detection algorithm
% on the PD@Home dataset. The following steps are performed:
% (1) Load PD@Home data;
% (2) Extract relevant labelled intervals;
% (3) Synchronize activity and tremor labels;
% (4) Perform Pre-Processing: downsampling; 
% (5) Perform feature extraction: MFCCs, frequency of the peak and 0.5-3 Hz power;
% (6) Train a Logistic Regression Classifier and evaluate performance in
%     the scenario: Leave-One-Subject-Out;
% (7) Train the Logistic Regression Classifier on the complete dataset


% Copyright 2025 Nienke A. Timmermans & Diogo C. Soriano
% 
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
% 
%     http://www.apache.org/licenses/LICENSE-2.0
% 
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

% #########################################################################

addpath(genpath('functions')); % Add all functions in the subfolder

TremorIDs  = ["pd02", "pd19", "pd06", "pd20", "pd07", "pd21", "pd22", "pd18"];
TremorArms = ["Right",  "Left",   "Left",   "Right",  "Right",  "Left", "Right",   "Left"]; % tremor side

NonTremorIDs = ["pd01","pd03","pd04","pd05","pd08","pd09","pd10","pd11","pd12","pd13","pd23","pd24","pd14","pd15","pd16","pd17"];
NonTremorArms = ["Left", "Right", "Right","Left", "Left", "Left", "Left", "Left", "Left", "Left", "Right", "Right", "Left","Right", "Left", "Right"]; % matched on hand dominance

HC_IDs = ["hc01","hc02","hc03","hc04","hc05","hc06","hc07","hc08","hc09","hc10","hc11","hc12",...
    "hc13","hc14","hc15","hc16","hc17","hc18","hc19","hc20","hc21","hc22","hc23","hc24"];
HC_Arms = ["Left", "Right", "Right", "Left", "Right", "Right", "Right", "Right", "Left", "Right", "Right","Right",...
    "Right", "Right", "Right", "Left", "Right", "Right", "Right", "Left", "Right", "Left", "Left", "Right"]; % matched on hand dominance

Fs = 200;   % Original sampling rate

FlagExtractFeatures =  1 ; % 1 - Extract features;
                         % 0 - Load previously extracted features for further classification;

if  FlagExtractFeatures == 1

    % First load data and extract features from PD tremor patients:

    % Load Data - set your path here
    load('...\video_annotations\labels_PD_phys_tremor.mat');
    load('...\sensor_data\phys_cur_PD_merged.mat');

    %% Loop over all tremor patients:

    for n = 1:length(TremorIDs)

        n % display subject index

        % Ensure that the id in Labels refers to same id in sensor data
        % structure
        for id = 1:25
            if phys(id).id == TremorIDs(n)
                current_id_phys = id;
            end
            if labels(id).id == TremorIDs(n)
                current_id_labels = id;
            end
        end

        % Take the side of interest
        if TremorArms(n) == "Left"
            ind_arm = 'LW';
        else
            ind_arm = 'RW';
        end

        %% Take Tremor Time Stamps
        ConfGet.current_id_labels = current_id_labels;
        ConfGet.current_id_phys = current_id_phys;
        ConfGet.ind_arm = ind_arm;

        % Premed
        ConfGet.KindofLabel = 'tremor';
        ConfGet.Condition   = 'premed';
        [TimeStampsTremorPremed] = GetTimeLogIntervals(ConfGet,phys,labels);

        % Postmed
        ConfGet.KindofLabel = 'tremor';
        ConfGet.Condition   = 'postmed';
        [TimeStampsTremorPostmed] = GetTimeLogIntervals(ConfGet,phys,labels);

        %% Take Activity Time Stamps
        % Premed
        ConfGet.KindofLabel = 'actv';
        ConfGet.Condition   = 'premed';
        [TimeStampsActvPremed] = GetTimeLogIntervals(ConfGet,phys,labels);

        % Postmed
        ConfGet.KindofLabel = 'actv';
        ConfGet.Condition   = 'postmed';
        [TimeStampsActvPostmed] = GetTimeLogIntervals(ConfGet,phys,labels);

        %% Synchronizing Tremor and Activity Labels taking a single signal
        % Define the Logical indexes in which there is a label for Tremor AND for Activity (logical union of label intervals)

        % Premed
        SignalLogicalIntervalPremed = TimeStampsTremorPremed.LogicalInterval & TimeStampsActvPremed.LogicalInterval;
        % Postmed
        SignalLogicalIntervalPostmed = TimeStampsTremorPostmed.LogicalInterval & TimeStampsActvPostmed.LogicalInterval;

        %% Organizing Label Vectors and Performing Feature Extraction
        
        % Configuration to generating Structure Data (including features)
        ConfGen.current_id_labels = current_id_labels;
        ConfGen.current_id_phys   = current_id_phys;
        ConfGen.ind_arm           = ind_arm;
        ConfGen.FillMissingLabelTransitions = 0;

        % Configuration Structure for Extracting Features
        ConfExtr.PerformExtract = 1; % 1 - Perform Feature Extrac, 0 - Return  Feature Vector Empty

        % Get Previous Pre-Processing Settings
        ConfExtr.Decimate       = 1;     % 1 - Perform decimation to 100 Hz

        % Features extraction parameters
        ConfExtr.FsOriginal     = Fs;   % Original Sampling Freq.
        ConfExtr.FsToAdjust     = 100;  % Sampling Freq to be adjusted inside.
        ConfExtr.WindowSizeTime = 4;    % Window size in time (s)
        ConfExtr.shift          = 0.5;  % percentual of shift in samples for the sliding window - 2 s shift
        ConfExtr.WinMin         = 3;    % Minimum size of the window in (s) to perform feature extraction. For windows shorter than that - features NaN
        ConfExtr.numFilters     = 15;  % Number of filters in the Mel Bank Filter
        ConfExtr.NumMelCoeff    = 12;  % Number of Coeficcients in the Discrete Cosine Transform of the Output - usually 12 is fine
        ConfExtr.MaxFreqFilter  = 25;  % Maximum Frequency to be considered in the Mel Bank Filter;
        ConfExtr.MFCCwin        = ConfExtr.WindowSizeTime/2; % Window size for MFCC computing
        ConfExtr.pwelchwin      = 3;    % Window size for PSD computing

        ConfGen.KindofLabel = 'tremor';
        ConfGen.Condition   = 'premed';

        [EventsDataTremorPremed(n)] = SlidingEventsDataComp(TremorIDs{n},ConfGen,ConfExtr,phys,labels,TimeStampsTremorPremed,TimeStampsActvPremed,SignalLogicalIntervalPremed);

        ConfGen.KindofLabel = 'tremor';
        ConfGen.Condition   = 'postmed';

        [EventsDataTremorPostmed(n)] = SlidingEventsDataComp(TremorIDs{n},ConfGen,ConfExtr,phys,labels,TimeStampsTremorPostmed,TimeStampsActvPostmed,SignalLogicalIntervalPostmed);

        %Concatenating Data Structure - Premed and Postmed
        EventsTremor(n).ID             = EventsDataTremorPremed(n).ID;
        EventsTremor(n).IndArm         = EventsDataTremorPremed(n).IndArm;
        EventsTremor(n).PxxVectorGyro  = [EventsDataTremorPremed(n).PxxVectorGyro,  EventsDataTremorPostmed(n).PxxVectorGyro];
        EventsTremor(n).RawLabel          = [EventsDataTremorPremed(n).RawLabel, EventsDataTremorPostmed(n).RawLabel];
        EventsTremor(n).TremorLabel = [EventsDataTremorPremed(n).TremorLabel, EventsDataTremorPostmed(n).TremorLabel];
        EventsTremor(n).ActvLabel = [EventsDataTremorPremed(n).ActvLabel, EventsDataTremorPostmed(n).ActvLabel];

    end

    %% Loop over all non-tremor patients:

    for n = 1:length(NonTremorIDs)

        n % display subject index

        % Ensure that the id in Labels refers to same id in sensor data
        % structure
        for id = 1:25
            if phys(id).id == NonTremorIDs(n)
                current_id_phys = id;
            end
            if labels(id).id == NonTremorIDs(n)
                current_id_labels = id;
            end
        end

        % Take the side of interest
        if NonTremorArms(n) == "Left"
            ind_arm = 'LW';
        else
            ind_arm = 'RW';
        end

        %% Take Activity Time Stamps
        ConfGet.current_id_labels = current_id_labels;
        ConfGet.current_id_phys = current_id_phys;
        ConfGet.ind_arm = ind_arm;

        % Premed
        ConfGet.KindofLabel = 'actv';
        ConfGet.Condition   = 'premed';
        [TimeStampsActvPremed] = GetTimeLogIntervals(ConfGet,phys,labels);

        % Postmed
        ConfGet.KindofLabel = 'actv';
        ConfGet.Condition   = 'postmed';
        [TimeStampsActvPostmed] = GetTimeLogIntervals(ConfGet,phys,labels);

        % Premed
        SignalLogicalIntervalPremed =  TimeStampsActvPremed.LogicalInterval;
        % Postmed
        SignalLogicalIntervalPostmed = TimeStampsActvPostmed.LogicalInterval;

        %% Organizing Label Vectors and Performing Feature Extraction

        % Configuration to generating Structure Data (inncluding features)
        ConfGen.current_id_labels = current_id_labels;
        ConfGen.current_id_phys   = current_id_phys;
        ConfGen.ind_arm           = ind_arm;

        ConfGen.KindofLabel = 'actv';
        ConfGen.Condition   = 'premed';

        [EventsDataNonTremorPremed(n)] = SlidingEventsDataComp(NonTremorIDs{n},ConfGen,ConfExtr,phys,labels,...
            [],TimeStampsActvPremed,SignalLogicalIntervalPremed);

        ConfGen.KindofLabel = 'actv';
        ConfGen.Condition   = 'postmed';

        [EventsDataNonTremorPostmed(n)] = SlidingEventsDataComp(NonTremorIDs{n},ConfGen,ConfExtr,phys,labels,...
            [],TimeStampsActvPostmed,SignalLogicalIntervalPostmed);

        %Concatenating Actv Data Structure - Premed and Postmed
        EventsNonTremor(n).ID             = EventsDataNonTremorPremed(n).ID;
        EventsNonTremor(n).IndArm         = EventsDataNonTremorPremed(n).IndArm;
        EventsNonTremor(n).PxxVectorGyro  = [EventsDataNonTremorPremed(n).PxxVectorGyro,  EventsDataNonTremorPostmed(n).PxxVectorGyro];
        EventsNonTremor(n).ActvLabel = [EventsDataNonTremorPremed(n).ActvLabel, EventsDataNonTremorPostmed(n).ActvLabel];

    end

    clear('EventsDataTremorPremed','EventsDataTremorPostmed','EventsDataNonTremorPremed','EventsDataNonTremorPostmed','labels','phys')

    %% Load data from non-PD controls
    load('...\video_annotations\labels_HC_phys.mat');
    load('...\sensor data\phys_cur_HC_merged.mat');
    
    %% Loop over all non-PD controls

    for n = 1:length(HC_IDs)

        n % display subject index

        % Ensure that the id in Labels refers to same id in sensor data
        % structure
        for id = 1:length(HC_IDs)
            if phys(id).id == HC_IDs{n}
                current_id_phys = id;
            end
            if labels(id).id == HC_IDs{n}
                current_id_labels = id;
            end
        end

        % Take the side of interest
        if HC_Arms(n) == "Left"
            ind_arm = 'LW';
        else
            ind_arm = 'RW';
        end

        %% Take Activity Time Stamps
        ConfGet.current_id_labels = current_id_labels;
        ConfGet.current_id_phys = current_id_phys;
        ConfGet.ind_arm = ind_arm;

        % Configuration Structure for Extracting Features
        ConfExtr.PerformExtract = 1; % 1 - Perform Feature Extract
        % 0 - Return  Feature Vector Empty

        % Get Previous Pre-Processing Settings
        ConfExtr.Decimate       = 1;     % 1 - Perform decimation to 100 Hz

        % Features extraction parameters
        ConfExtr.FsOriginal     = Fs;   % Original Sampling Freq.
        ConfExtr.FsToAdjust     = 100;  % Sampling Freq to be adjusted inside.
        ConfExtr.WindowSizeTime = 4;    % Window size in time (s)
        ConfExtr.shift          = 0.5;  % percentual of shift in samples for the sliding window - 2 s shift
        ConfExtr.WinMin         = 3;    % Minimum size of the window in (s) to perform feature extraction. For windows shorter than that - features NaN
        ConfExtr.numFilters      = 15;  % Number of filters in the Mel Bank Filter
        ConfExtr.NumMelCoeff     = 12;  % Number of Coeficcients in the Discrete Cosine Transform of the Output - usually 12 is fine
        ConfExtr.MaxFreqFilter   = 25;  % Maximum Frequency to be considered in the Mel Bank Filter;
        ConfExtr.MFCCwin         = ConfExtr.WindowSizeTime/2; % Window size for MFCC computing
        ConfExtr.pwelchwin      = 3;    % Window size for PSD computing
        
        % Premed
        ConfGet.KindofLabel = 'actv';
        ConfGet.Condition   = 'pre';
        [TimeStampsActvPre] = GetTimeLogIntervals(ConfGet,phys,labels);
       
        % Premed
        SignalLogicalIntervalPre =  TimeStampsActvPre.LogicalInterval;
        
        if ~isempty(labels(current_id_labels).post)
            % Postmed
            ConfGet.KindofLabel = 'actv';
            ConfGet.Condition   = 'post';

            [TimeStampsActvPost] = GetTimeLogIntervals(ConfGet,phys,labels);
            SignalLogicalIntervalPost = TimeStampsActvPost.LogicalInterval;
        end

        %% Organizing Label Vectors and Performing Feature Extraction

        % Configuration to generating Structure Data (inncluding features)
        ConfGen.current_id_labels = current_id_labels;
        ConfGen.current_id_phys   = current_id_phys;
        ConfGen.ind_arm           = ind_arm;

        ConfGen.KindofLabel = 'actv';
        ConfGen.Condition   = 'pre';

        [EventsDataHCPre(n)] = SlidingEventsDataComp(HC_IDs{n},ConfGen,ConfExtr,phys,labels,...
            [],TimeStampsActvPre,SignalLogicalIntervalPre);

        ConfGen.KindofLabel = 'actv';
        ConfGen.Condition   = 'post';

        if exist("TimeStampsActvPost")
            [EventsDataHCPost(n)] = SlidingEventsDataComp(HC_IDs{n},ConfGen,ConfExtr,phys,labels,...
                [],TimeStampsActvPost,SignalLogicalIntervalPost);
        else
            EventsDataHCPost(n).PxxVectorGyro =[];
            EventsDataHCPost(n).ActvLabel = [];
        end

        %Concatenating Actv Data Structure - Pre and Post
        EventsHC(n).ID             = EventsDataHCPre(n).ID;
        EventsHC(n).IndArm         = EventsDataHCPre(n).IndArm;
        EventsHC(n).PxxVectorGyro  = [EventsDataHCPre(n).PxxVectorGyro, EventsDataHCPost(n).PxxVectorGyro];
        EventsHC(n).ActvLabel = [EventsDataHCPre(n).ActvLabel, EventsDataHCPost(n).ActvLabel];

        clear('TimeStampsActvPre','TimeStampsActvPost')
    end

    clear('EventsDataHCPre','EventsDataHCPost','labels','phys')

else % load previously extracted features
    load('EventsTremor.mat')
    load('EventsNonTremor.mat')
    load('EventsHC.mat')
end

%% Delete cycling windows of tremor patient hbv017, since presence of tremor is not clear on the video

Cycling_idx = find(EventsTremor(3).ActvLabel==9);
EventsTremor(3).PxxVectorGyro(Cycling_idx) = [];
EventsTremor(3).RawLabel(Cycling_idx) = [];
EventsTremor(3).TremorLabel(Cycling_idx) = [];
EventsTremor(3).ActvLabel(Cycling_idx) = [];

%% Organize data for training the classifier

LabelsTremor = 1;
LabelsNonTremor = 0;

m = 100; % penalty factor for cycling windows in nontremor patients and healthy controls

FeaturesTremor = [];
FeaturesNonTremor = [];
ActvLabelsTremor = [];
ActvLabelsNonTremor = [];
IDsTremor = [];
IDsNonTremor = [];
FreqPeakTremor = [];
FreqPeakNonTremor = [];
RawLabelTremor = [];
RawLabelNonTremor = [];
ArmActvPowerTremor = [];
ArmActvPowerNonTremor = [];

for n = 1:size(EventsTremor,2)
    n
    for i = 1:size(EventsTremor(n).PxxVectorGyro,2)
        if ~isnan(EventsTremor(n).PxxVectorGyro(i).NumWindows)
            if ismember(EventsTremor(n).TremorLabel(i),LabelsTremor)
                FeaturesTremor = [FeaturesTremor; EventsTremor(n).PxxVectorGyro(i).MelCepsCoeff];
                ActvLabelsTremor = [ActvLabelsTremor; EventsTremor(n).ActvLabel(i)];
                IDsTremor = [IDsTremor; EventsTremor(n).ID];
                FreqPeakTremor = [FreqPeakTremor; EventsTremor(n).PxxVectorGyro(i).FreqPeak];
                RawLabelTremor = [RawLabelTremor; EventsTremor(n).RawLabel(i)];
                ArmActvPowerTremor = [ArmActvPowerTremor; EventsTremor(n).PxxVectorGyro(i).ArmActvPower];
            elseif ismember(EventsTremor(n).TremorLabel(i),LabelsNonTremor)
                FeaturesNonTremor = [FeaturesNonTremor; EventsTremor(n).PxxVectorGyro(i).MelCepsCoeff];
                ActvLabelsNonTremor = [ActvLabelsNonTremor; EventsTremor(n).ActvLabel(i)];
                IDsNonTremor = [IDsNonTremor; EventsTremor(n).ID];
                FreqPeakNonTremor = [FreqPeakNonTremor; EventsTremor(n).PxxVectorGyro(i).FreqPeak];
                RawLabelNonTremor = [RawLabelNonTremor; EventsTremor(n).RawLabel(i)];
                ArmActvPowerNonTremor = [ArmActvPowerNonTremor; EventsTremor(n).PxxVectorGyro(i).ArmActvPower];
            end
        end
    end
end

for n = 1:size(EventsNonTremor,2)
    n
    for i = 1:size(EventsNonTremor(n).PxxVectorGyro,2)
        if ~isnan(EventsNonTremor(n).PxxVectorGyro(i).NumWindows)
            if EventsNonTremor(n).ActvLabel(i)==9
                FeaturesNonTremor = [FeaturesNonTremor; repmat(EventsNonTremor(n).PxxVectorGyro(i).MelCepsCoeff,[m,1])];
                ActvLabelsNonTremor = [ActvLabelsNonTremor; repmat(EventsNonTremor(n).ActvLabel(i),[m,1])];
                IDsNonTremor = [IDsNonTremor; repmat(EventsNonTremor(n).ID,[m,1])];
                FreqPeakNonTremor = [FreqPeakNonTremor; repmat(EventsNonTremor(n).PxxVectorGyro(i).FreqPeak,[m,1])];
                RawLabelNonTremor = [RawLabelNonTremor; NaN(m,1)];
                ArmActvPowerNonTremor = [ArmActvPowerNonTremor; repmat(EventsNonTremor(n).PxxVectorGyro(i).ArmActvPower,[m,1])];
            else
                FeaturesNonTremor = [FeaturesNonTremor; EventsNonTremor(n).PxxVectorGyro(i).MelCepsCoeff];
                ActvLabelsNonTremor = [ActvLabelsNonTremor; EventsNonTremor(n).ActvLabel(i)];
                IDsNonTremor = [IDsNonTremor; EventsNonTremor(n).ID];
                FreqPeakNonTremor = [FreqPeakNonTremor; EventsNonTremor(n).PxxVectorGyro(i).FreqPeak];
                RawLabelNonTremor = [RawLabelNonTremor; NaN];
                ArmActvPowerNonTremor = [ArmActvPowerNonTremor; EventsNonTremor(n).PxxVectorGyro(i).ArmActvPower];
            end
        end
    end
end

for n = 1:size(EventsHC,2)
    n
    for i = 1:size(EventsHC(n).PxxVectorGyro,2)
        if ~isnan(EventsHC(n).PxxVectorGyro(i).NumWindows)
            if EventsHC(n).ActvLabel(i)==9
                FeaturesNonTremor = [FeaturesNonTremor; repmat(EventsHC(n).PxxVectorGyro(i).MelCepsCoeff,[m,1])];
                ActvLabelsNonTremor = [ActvLabelsNonTremor; repmat(EventsHC(n).ActvLabel(i),[m,1])];
                IDsNonTremor = [IDsNonTremor; repmat(EventsHC(n).ID,[m,1])];
                FreqPeakNonTremor = [FreqPeakNonTremor; repmat(EventsHC(n).PxxVectorGyro(i).FreqPeak,[m,1])];
                RawLabelNonTremor = [RawLabelNonTremor; NaN(m,1)];
                ArmActvPowerNonTremor = [ArmActvPowerNonTremor; repmat(EventsHC(n).PxxVectorGyro(i).ArmActvPower,[m,1])];
            else
                FeaturesNonTremor = [FeaturesNonTremor; EventsHC(n).PxxVectorGyro(i).MelCepsCoeff];
                ActvLabelsNonTremor = [ActvLabelsNonTremor; EventsHC(n).ActvLabel(i)];
                IDsNonTremor = [IDsNonTremor; EventsHC(n).ID];
                FreqPeakNonTremor = [FreqPeakNonTremor; EventsHC(n).PxxVectorGyro(i).FreqPeak];
                RawLabelNonTremor = [RawLabelNonTremor; NaN];
                ArmActvPowerNonTremor = [ArmActvPowerNonTremor; EventsHC(n).PxxVectorGyro(i).ArmActvPower];
            end
        end
    end
end

% Concatenate nontremor and tremor structures

Features = [FeaturesNonTremor; FeaturesTremor];
Labels = [zeros(length(IDsNonTremor),1); ones(length(IDsTremor),1)];
ActvLabels = [ActvLabelsNonTremor; ActvLabelsTremor];
IDsWindow = [IDsNonTremor; IDsTremor];
FreqPeak = [FreqPeakNonTremor; FreqPeakTremor];
RawLabels = [RawLabelNonTremor; RawLabelTremor];
ArmActvPower = [ArmActvPowerNonTremor; ArmActvPowerTremor];

%% Leave-one-subject-out cross-validation
Performance_LOSOCV = [];

IDs = [TremorIDs,NonTremorIDs,HC_IDs];

for n = 1:length(IDs)

n

TrainData=[];
TrainLabels=[];
XTestScaled=[];
TrainingTremorPerformance=[];

    TestID  = IDs(n);
    TestData = Features(strcmp(IDsWindow,TestID),:).Variables;
    TestLabels = Labels(strcmp(IDsWindow,TestID),:);
    TestActvLabels = ActvLabels(strcmp(IDsWindow,TestID),:);
    TestFreqPeak = FreqPeak(strcmp(IDsWindow,TestID),:);
    TestRawLabels = RawLabels(strcmp(IDsWindow,TestID),:);
    TestArmActvPower = ArmActvPower(strcmp(IDsWindow,TestID),:);

    TrainID = IDs;
    TrainID(strcmp(IDs,TestID)) = [];
    for i = 1:length(TrainID)
        TrainData = [TrainData; Features(strcmp(IDsWindow,TrainID(i)),:)];
        TrainLabels = [TrainLabels; Labels(strcmp(IDsWindow,TrainID(i)),:)];
    end

    % Scaling
    [XTrainScaled,MeanVector,SigmaVector] = zscore(TrainData.Variables);
    [~,Natr] = size(XTrainScaled);

    Mdl = fitglm(XTrainScaled,TrainLabels,'linear','distr','binomial','link','logit');
    
    for col = 1:Natr
        XTestScaled(:,col) = (TestData(:,col) - MeanVector(col))./SigmaVector(col);
    end

    TrainingTremorPerformance = ScoresClassification(Mdl.Fitted.Response,TrainLabels,Mdl.Fitted.Response,TrainLabels);
    
    [ScoresTest] = predict(Mdl,XTestScaled);

    Threshold(n) =  TrainingTremorPerformance.ThrSpec95;

    Labels_LR = NaN(length(ScoresTest),1);
    for i = 1:length(ScoresTest)
        if ScoresTest(i)>Threshold(n)
            Labels_LR(i)=1;
        else
            Labels_LR(i)=0;
        end
    end
    
    Labels_predicted = Labels_LR;
    Labels_predicted(Labels_LR==1 & (TestFreqPeak <3 | TestFreqPeak >7 | TestArmActvPower > 50))=0; % perform extra checks

    Performance_LOSOCV.overall.FN(n) = length(find(Labels_predicted == 0 & TestLabels == 1));
    Performance_LOSOCV.overall.FP(n) = length(find(Labels_predicted == 1 & TestLabels == 0));

    Performance_LOSOCV.overall.TN(n) = length(find(Labels_predicted == 0 & TestLabels == 0));
    Performance_LOSOCV.overall.TP(n) = length(find(Labels_predicted == 1 & TestLabels == 1));

    Performance_LOSOCV.overall.Sensitivity(n) = Performance_LOSOCV.overall.TP(n)/(Performance_LOSOCV.overall.TP(n)+Performance_LOSOCV.overall.FN(n));
    Performance_LOSOCV.overall.Specificity(n) = Performance_LOSOCV.overall.TN(n)/(Performance_LOSOCV.overall.TN(n)+Performance_LOSOCV.overall.FP(n));
 
    activities = {'gait','cycling','sitting', 'standing', 'turning', 'posttransit', 'exercise', 'driving', 'armact', 'susp'};
    labels = [3, 9, 1, 2, 4, 7, [8,10,13], [11,12], 98, 96];

    for i = 1:length(activities)

        activity = activities{i};
        label = labels(i);

        if any(strcmp(activity, {'armact','susp'}))
            
            Performance_LOSOCV.(activity).TN(n) = length(find(Labels_predicted == 0 & TestLabels == 0 & ismember(TestRawLabels, label)));
            Performance_LOSOCV.(activity).FP(n) = length(find(Labels_predicted == 1 & TestLabels == 0 & ismember(TestRawLabels, label)));
            Performance_LOSOCV.(activity).TP(n) = length(find(Labels_predicted == 1 & TestLabels == 1 & ismember(TestRawLabels, label)));
            Performance_LOSOCV.(activity).FN(n) = length(find(Labels_predicted == 0 & TestLabels == 1 & ismember(TestRawLabels, label)));

            Performance_LOSOCV.(activity).Sensitivity(n) = Performance_LOSOCV.(activity).TP(n) / (Performance_LOSOCV.(activity).TP(n) + Performance_LOSOCV.(activity).FN(n));
            Performance_LOSOCV.(activity).Specificity(n) = Performance_LOSOCV.(activity).TN(n) / (Performance_LOSOCV.(activity).TN(n) + Performance_LOSOCV.(activity).FP(n));
        
        else

            Performance_LOSOCV.(activity).TN(n) = length(find(Labels_predicted == 0 & TestLabels == 0 & ismember(TestActvLabels, label)));
            Performance_LOSOCV.(activity).FP(n) = length(find(Labels_predicted == 1 & TestLabels == 0 & ismember(TestActvLabels, label)));
            Performance_LOSOCV.(activity).TP(n) = length(find(Labels_predicted == 1 & TestLabels == 1 & ismember(TestActvLabels, label)));
            Performance_LOSOCV.(activity).FN(n) = length(find(Labels_predicted == 0 & TestLabels == 1 & ismember(TestActvLabels, label)));

            Performance_LOSOCV.(activity).Sensitivity(n) = Performance_LOSOCV.(activity).TP(n) / (Performance_LOSOCV.(activity).TP(n) + Performance_LOSOCV.(activity).FN(n));
            Performance_LOSOCV.(activity).Specificity(n) = Performance_LOSOCV.(activity).TN(n) / (Performance_LOSOCV.(activity).TN(n) + Performance_LOSOCV.(activity).FP(n));
        end
    end

end

%% Train tremor detector on complete trainingset

% Scaling
[XTrainScaled,MeanVector,SigmaVector] = zscore(Features.Variables);
[~,Natr] = size(XTrainScaled);

% Train a logistic regression classifier on the complete dataset
Mdl = fitglm(XTrainScaled,Labels,'linear','distr','binomial','link','logit');
TrainingTremorPerformance = ScoresClassification(Mdl.Fitted.Response,Labels,Mdl.Fitted.Response,Labels);

Threshold_total =  TrainingTremorPerformance.ThrSpec95; % threshold at 95% specificity

Labels_LR = zeros(length(Labels),1);
Labels_LR(TrainingTremorPerformance.Scores>Threshold_total)=1;

% Perform extra checks for rest tremor
Labels_predicted = Labels_LR;
Labels_predicted(Labels_LR==1 & (FreqPeak <3 | FreqPeak >7 | ArmActvPower>50))=0;

FN_total= length(find(Labels_predicted == 0 & Labels == 1));
FP_total = length(find(Labels_predicted == 1 & Labels == 0));

TN_total = length(find(Labels_predicted == 0 & Labels == 0));
TP_total = length(find(Labels_predicted == 1 & Labels == 1));

Sensitivity_total = TP_total/(TP_total+FN_total);
Specificity_total = TN_total/(TN_total+FP_total);
