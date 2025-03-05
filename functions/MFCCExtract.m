function [TableOut] = MFCCExtract(Conf,Data)
% #########################################################################

% Input
% Conf - Structure
% Data - Matrix Data (Columns)

% Output
% TableOut - Table with Mel Frequency Coefficients

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

[Nsamples,Nsignals] = size(Data);

Fs          = Conf.Fs;
numFilters  = Conf.numFilters;
NumMelCoeff = Conf.NumMelCoeff;
StrAxis     = Conf.StrAxis;

% MyMFCC Coeffcients
filterbankStart = 0;
filterbankEnd   = Conf.MaxFreqFilter; % maximum Frequency

numBandEdges = numFilters + 2;

NFFT         = round(Conf.MFCCwin*Fs);        % take 2s of data - subwindows

fresol       = Fs/NFFT;     % frequency resolution if 0.5 Hz

filterBank   = zeros(numFilters,NFFT);

x = filterbankStart:0.01:filterbankEnd;

melscale = 64.875*log10( 1 + x./17.5);

xlinmel = linspace(melscale(1),melscale(end),numBandEdges);

bandEdges = 17.5*(10.^(xlinmel/64.875) - 1);

bandEdgesBins = round((bandEdges/Fs)*NFFT) + 1;

for ii = 1:numFilters
    filt = triang(bandEdgesBins(ii+2)-bandEdgesBins(ii));
    leftPad = bandEdgesBins(ii);
    rightPad = NFFT - numel(filt) - leftPad;
    filterBank(ii,:) = [zeros(1,leftPad),filt',zeros(1,rightPad)];
    filterBank(ii,:) = filterBank(ii,:)/(fresol*sum(filterBank(ii,:)));
end

overlap = round(0.8*NFFT); % ovelap between each subwindow

% Extract MFCC for each axis
switch StrAxis
    case 'X'
        Signal = Data(:,1);
        [S1,f,t] = stft(Signal,Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        S = abs(S1);
    case 'Y'
        Signal = Data(:,2);
        [S2,f,t] = stft(Signal,Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        S = abs(S2);
    case 'Z'
        Signal = Data(:,3);
        [S3,f,t] = stft(Signal,Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        S = abs(S3);
    case 'Sum'
        [S1,f,t] = stft(Data(:,1),Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        [S2,f,t] = stft(Data(:,2),Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        [S3,f,t] = stft(Data(:,3),Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        S = abs(S1) + abs(S2) + abs(S3);
end

for pp = 1:NumMelCoeff
    CepsName{1,pp} = ['MFCC',num2str(pp)];
end

VarName = CepsName;

Spec      = filterBank*S;

CepsCoeff = cepstralCoefficients(Spec,'NumCoeffs',NumMelCoeff);

Ceps = mean(CepsCoeff);

CepsVariables = Ceps;

TableOut  = array2table(CepsVariables,'VariableNames',VarName);