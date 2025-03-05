function [PxxVector] = SpectralExtract(Conf,DataWindow,ID)

% #########################################################################

% This function extracts spectral features

% Input
% Conf - Configuration Structure:

% Conf.Fs               - Sampling Rate
% Conf.WinMin           - Minimum Length of the Window in Seconds for Feature Extraction
% Conf.WindowSizeTime   - Window size in seconds for feature extraction
% Conf.numFilters       - Number of Filter for Mel Bank filters - MFCCs
% Conf.NumMelCoeff      - Number of MFCC Coeficients;
% Conf.pwelchwin        - Windos size in seconds for PSD estimation

% DataWindow            - Sensordata in window
% ID                    - Subject ID

% Output
% PxxVector - Structure of extracted features and stored information


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

Fs                       = Conf.FsToAdjust;     % sampling rate
PxxVector.ID             = ID;                  % Take the ID
WindowSizeTime           = Conf.WindowSizeTime; % Size of the window (time - s);
PxxVector.WindowSizeTime = WindowSizeTime;      % Store window size for output structure

%% Extract features
% For mel-frequency cepstral coefficients see San Segundo et al., 2016 - https://www.sciencedirect.com/science/article/pii/S016516841500331X

numFilters    = Conf.numFilters;  % Number of filters in the Mel Bank Filter
NumMelCoeff   = Conf.NumMelCoeff; % Number of Coeficcients in the Discrete Cosine Transform of the Output - usually 12 is fine
MaxFreqFilter = Conf.MaxFreqFilter; % Maximum Frequency to be considered in the Mel Bank Filter;
MFCCwin       = Conf.MFCCwin;     % Window size for MFCC computing
pwelchwin     = Conf.pwelchwin;   % Window size for PSD computing 

MelCepsCoeffTable = []; % Initializing Features Tables

if length(DataWindow(:,1)) < Conf.WinMin*Fs % Check if the window length is enough

    PxxVector.NumWindows   = NaN;
    PxxVector.MelCepsCoeff = NaN;
    PxxVector.FreqPeak     = NaN;
    PxxVector.ArmActvPower = NaN;
  
else
    
    % Calculate the PSD
    fresol = 0.25; % Frequency resolution in Hz
    nfft = Fs*1/fresol;
    noverlap = round(0.8*pwelchwin);
    [Pxx,freqlong] = pwelch(DataWindow,hann(pwelchwin*Fs),noverlap,nfft,Fs);
    
    % Extract the frequency of the peak between 1 and 25 Hz
    freqmax = 25;
    freqmin = 1;
    indcutfreq     = find(freqlong == freqmax);
    FreqVect       = freqlong(1:indcutfreq);

    PSDMatrix = Pxx(1:indcutfreq,:);
    PSDMatrix(:,4) = sum(PSDMatrix,2);
    [~,IndPeak] = max(PSDMatrix(FreqVect>=freqmin,4));
    PxxVector.FreqPeak = FreqVect(IndPeak)+freqmin;

    % Calculate the 0.5 - 3 Hz power
    InitialIndexBandForPower = find(FreqVect == 0.5);      % Initial index for bandpower estimation
    FinalIndexBandForPower   = find(FreqVect == (3-fresol)); % Final index for bandpower estimation: area will be estimated using approximation by the left
    PxxVector.ArmActvPower  = fresol*sum(PSDMatrix(InitialIndexBandForPower:FinalIndexBandForPower,4));    % Power in the band specified

    % Calculate the MFCCs
    ConfMFCC.Fs          =  Fs;
    ConfMFCC.numFilters  = numFilters;
    ConfMFCC.NumMelCoeff = NumMelCoeff;
    ConfMFCC.StrAxis     = 'Sum';
    ConfMFCC.MaxFreqFilter = MaxFreqFilter;
    ConfMFCC.MFCCwin = MFCCwin;

    [MelCepsCoeffAux] = MFCCExtract(ConfMFCC,DataWindow);
    MelCepsCoeffTable = vertcat(MelCepsCoeffTable,MelCepsCoeffAux);
    clear('MelCepsCoeffAux');
    PxxVector.MelCepsCoeff = MelCepsCoeffTable;
    PxxVector.NumWindows = 1;
    
end
