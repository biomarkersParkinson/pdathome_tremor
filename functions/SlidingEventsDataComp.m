function [EventsData] = SlidingEventsDataComp(ID,ConfGen,ConfExtr,phys,labels,...
    TimeStampsTremor,TimeStampsActv,SignalLogicalInterval)

% #########################################################################

% Extracts tremor and activity labels, performs preprocessing of gyroscope data and extracts features across sliding windows

% Input
% ID         - string (Patient ID)
% ConfGen    - Structure for Data generation
% ConfExtr   - Structure for Feature Extraction Parameters
% phys       - Provided Data Structure
% labels     - Provided Label Structure
% TimeStampsTremor - Structure that indicates time events for tremor
% TimeStampsTremor - Structure that indicates time events for activity
% SignalLogicalInterval - Logical vector indicating in time the
%                         relevant labelled data

% Output
% EventsData - Structure with ID and Feature Vector
% EventsData.ID        - patient ID - string
% EventsData.IndArm    - side of interest - string
% EventsData.Condition - Condition;   % type of condition (premed or postmed) - string
% EventsData.RawLabel     - Raw tremor label
% EventsData.TremorLabel - Tremor label (0 or 1)
% EventsData.TremorTransitionFlag - Tremor transition flag
% EventsData.ActvLabel - Activity label
% EventsData.ActvTransitionFlag - Activity transition flag
% EventsData.PxxVector - Features Extracted - Structure

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

current_id_labels = ConfGen.current_id_labels;
KindofLabel       = ConfGen.KindofLabel;
Condition         = ConfGen.Condition;
current_id_phys   = ConfGen.current_id_phys;
ind_arm           = ConfGen.ind_arm;

% Information of the Output Data Structure
EventsData.ID        = ID;          % patient ID
EventsData.IndArm    = ind_arm;     % side of interest
EventsData.Condition = Condition;   % type of condition (premed or postmed)

FsToAdjust = ConfExtr.FsToAdjust;
EventsData.Fs = FsToAdjust; % Store the Final Sampling Rate

%% Extract tremor labels

switch KindofLabel

    case 'tremor' % only for tremor PD subjects, not for the non-tremor PD subjects or HC subjects

        switch Condition

            case 'premed'

                field = 'premed_tremor';

            case 'postmed'

                field = 'postmed_tremor';
        end

        % Taking the vector of starting events from a fixed reference
        IndexTableToData    = TimeStampsTremor.IndVector(1) + labels(current_id_labels).(field).Start;

        % Taking the duration vector of the events
        Duration            = labels(current_id_labels).(field).Duration;

        % Taking the label of the events
        LabelVector         = double(string(labels(current_id_labels).(field).Code));

        % Take the number of events
        [NTrialsTable,~]    = size(labels(current_id_labels).(field));

        % Vector with the same size of the relevant data interval - starting as NaN
        SignalSynchLabel = NaN(length(SignalLogicalInterval),1);

        IndVector    = [];
        SlidingLabel = [];

        for kk = 1:NTrialsTable % kk is the index for each event in the table

            % Take the indexes of a labelled event in the table
            IndVectorEvent = IndexTableToData(kk):(IndexTableToData(kk) + Duration(kk) - 1);
            IndVector      = [IndVector; IndVectorEvent'];

            % Put the label for each sample according to the event label. This builds a label signal
            % with a common reference for tremor or activity labels. This allows to
            % visualize both label signals in a synchronized way.

            SlidingLabel                     = [SlidingLabel; LabelVector(kk)*ones(length(IndVectorEvent),1)];
            SignalSynchLabel(IndVectorEvent) = LabelVector(kk)*ones(length(IndVectorEvent),1);
            clear('IndVectorEvent');

        end

        contpremedlabels = SignalSynchLabel(SignalLogicalInterval); % continuous tremor labels in the correct interval
        contpremedlabelsdown = downsample(contpremedlabels,ConfExtr.FsOriginal/ConfExtr.FsToAdjust); % downsample the continuous tremor labels

        WindowSizeTime = ConfExtr.WindowSizeTime;
        NwindowPoints  = WindowSizeTime*FsToAdjust;
        shift          = round(ConfExtr.shift*NwindowPoints);

        % Create windows
        [LabelWinData,inipoint,endpoint,center] = SlidingWindowCell(double(string(contpremedlabelsdown)), ...
            NwindowPoints,shift);

        [~,NLabelWindows] = size(LabelWinData);

        NClassPossible = [0 1 2 3 96 97 98 99]; % Possible raw tremor labels

        for jj = 1:NLabelWindows

            NclassObs = unique(LabelWinData{jj});

            if length(NclassObs) == 1
                TransitionFlag(jj) = 0;
            else
                TransitionFlag(jj) = 1;
            end

            Nsamples_0 = length(find(LabelWinData{jj}==0));

            % Non-arm activity
            Nsamples_1 = length(find(LabelWinData{jj}==1));
            Nsamples_2 = length(find(LabelWinData{jj}==2));
            Nsamples_3 = length(find(LabelWinData{jj}==3));

            % Arm activity
            Nsamples_96 = length(find(LabelWinData{jj}==96));
            Nsamples_98 = length(find(LabelWinData{jj}==98));
            Nsamples_97 = length(find(LabelWinData{jj}==97));

            % Not accessable
            Nsamples_99 = length(find(LabelWinData{jj}==99));

            NsamplesClassesVect = [Nsamples_0 Nsamples_1 Nsamples_2 Nsamples_3 ...
                Nsamples_96 Nsamples_97 Nsamples_98 Nsamples_99];

            [~,indmax] = max(NsamplesClassesVect); % find the most prevalent tremor label in the window

            LabelWindow(jj) = NClassPossible(indmax);

            % Determine tremor label (0 or 1) by combining all tremor labels and
            % combining all non-tremor labels

            Nsamples_tremor = Nsamples_1 + Nsamples_2 + Nsamples_3 + Nsamples_97;

            Nsamples_nontremor = Nsamples_0 + Nsamples_96 + Nsamples_98;

            Nsamples_na = Nsamples_99;

            NsamplesClassesVect2 = [Nsamples_tremor Nsamples_nontremor Nsamples_na];

            [~,indmax2] = max(NsamplesClassesVect2); % find again the most prevalent tremor label

            NClassPossible2 = [1 0 99];

            TremorLabelWindow(jj) = NClassPossible2(indmax2);

            clear('NclassObs','NsamplesClassesVect','indmax','NsamplesClassesVect2','indmax2');

        end

        EventsData.RawLabel = LabelWindow; % Store the raw tremor label
        EventsData.TremorLabel = TremorLabelWindow; % Store the assigned tremor label (0 or 1)
        EventsData.TremorTransitionFlag = TransitionFlag; % Store the transition flag for the tremor label

end


%% Extract activity labels

switch Condition

    case 'premed' % For PD subjects

        field = 'premed';

    case 'postmed' % For PD subjects

        field = 'postmed';

    case 'pre' % For HC subjects

        field = 'pre';

    case 'post' % For HC subjects

        field = 'post';
end

% Taking the vector of starting events from a fixed reference
IndexTableToData    = TimeStampsActv.IndVector(1) + labels(current_id_labels).(field).Start;

% Taking the duration vector of the events
Duration            = labels(current_id_labels).(field).Duration;

% Taking the label of the events
LabelVector         = double(string(labels(current_id_labels).(field).Code));

% Take the number of events
[NTrialsTable,~]    = size(labels(current_id_labels).(field));

% Create vector with the same size of the relevant data interval - starting as NaN
SignalSynchLabel = NaN(length(SignalLogicalInterval),1);

IndVector    = [];
SlidingLabel = [];

for kk = 1:NTrialsTable % kk is the index for each event in the table

    % Take the indexes of a labelled event in the table
    IndVectorEvent = IndexTableToData(kk):(IndexTableToData(kk) + Duration(kk) - 1);
    IndVector      = [IndVector; IndVectorEvent'];

    % Put the label for each sample according to the event label. This builds a label signal
    % with a common reference for tremor or activity labels. This allows to
    % visualize both label signals in a synchronized way.

    SlidingLabel                     = [SlidingLabel; LabelVector(kk)*ones(length(IndVectorEvent),1)];
    SignalSynchLabel(IndVectorEvent) = LabelVector(kk)*ones(length(IndVectorEvent),1);
    clear('IndVectorEvent');

end

contpremedlabels = SignalSynchLabel(SignalLogicalInterval); % continuous labels in the correct interval
contpremedlabelsdown = downsample(contpremedlabels,ConfExtr.FsOriginal/ConfExtr.FsToAdjust); % downsample continous labels

WindowSizeTime = ConfExtr.WindowSizeTime;
NwindowPoints  = WindowSizeTime*ConfExtr.FsToAdjust;
shift          = round(ConfExtr.shift*NwindowPoints);

% Create windows
[LabelWinData,inipoint,endpoint,center] = SlidingWindowCell(double(string(contpremedlabelsdown)), ...
    NwindowPoints,shift);

[~,NLabelWindows] = size(LabelWinData);

NClassPossible = [1 2 3 4 5 6 7 8 9 10 11 12 13 99]; % Possible activity labels

for jj = 1:NLabelWindows

    NclassObs = unique(LabelWinData{jj});

    if length(NclassObs) == 1
        TransitionFlag(jj) = 0;
    else
        TransitionFlag(jj) = 1;
    end

    Nsamples_1 = length(find(LabelWinData{jj}==1));
    Nsamples_2 = length(find(LabelWinData{jj}==2));
    Nsamples_3 = length(find(LabelWinData{jj}==3));
    Nsamples_4 = length(find(LabelWinData{jj}==4));
    Nsamples_5 = length(find(floor(LabelWinData{jj})==5)); % Add the subgroups together
    Nsamples_6 = length(find(LabelWinData{jj}==6));
    Nsamples_7 = length(find(floor(LabelWinData{jj})==7)); % Add the subgroups together
    Nsamples_8 = length(find(LabelWinData{jj}==8));
    Nsamples_9 = length(find(LabelWinData{jj}==9));
    Nsamples_10 = length(find(LabelWinData{jj}==10));
    Nsamples_11 = length(find(LabelWinData{jj}==11));
    Nsamples_12 = length(find(LabelWinData{jj}==12));
    Nsamples_13 = length(find(LabelWinData{jj}==13));

    % Not accessable
    Nsamples_99 = length(find(LabelWinData{jj}==99));

    NsamplesClassesVect = [Nsamples_1 Nsamples_2 Nsamples_3 ...
        Nsamples_4 Nsamples_5 Nsamples_6 Nsamples_7 Nsamples_8 Nsamples_9...
        Nsamples_10 Nsamples_11 Nsamples_12 Nsamples_13 Nsamples_99];


    [~,indmax] = max(NsamplesClassesVect); % find the most prevalent activity label in the window

    LabelWindow(jj) = NClassPossible(indmax); % assign this label to the window

    clear('NclassObs','NsamplesClassesVect','indmax');

end

EventsData.ActvLabel = LabelWindow; % Store activity label
EventsData.ActvTransitionFlag = TransitionFlag; % Store transition flag for activity label

%% Preprocessing and feature extraction

GyroContinuous = phys(current_id_phys).(ind_arm).gyro(SignalLogicalInterval,2:4); % select the gyroscope data in the correct interval

% Decimate if wished
if ConfExtr.Decimate == 1
    for ss = 1:3
        % Downsample the tremor signal to 100 Hz
        GyrosContinuousDec(:,ss)  = decimate(GyroContinuous(:,ss),ConfExtr.FsOriginal/ConfExtr.FsToAdjust);
    end
end

% Extract windows
TypeOfSignal = 'Gy';
[DataWinGyro,inipoint,endpoint,center] = SlidingWindowCell(GyrosContinuousDec,NwindowPoints,shift);
[~,NDataWindows] = size(DataWinGyro);

% Extract features
for kk = 1:NDataWindows
    [PxxVectorGyro(kk)]   = SpectralExtract(ConfExtr,DataWinGyro{kk},ID);
end

% Store in the Output pre-precessing options
EventsData.Decimate     = ConfExtr.Decimate;

if ConfExtr.PerformExtract == 1
    EventsData.PxxVectorGyro  = PxxVectorGyro; % Store in the Output Features Extracted
else
    EventsData.PxxVectorGyro  = [];
end

end
