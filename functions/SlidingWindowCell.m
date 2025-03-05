function [DataWin,inipoint,endpoint,center] = SlidingWindowCell(Data,NwindowPoints,shift)

% #########################################################################
% Input
% Data: Data matrix - Nsamples x Nvariables
% NwindowPoints     - number of samples of the window
% shift             - number of samples related to the shift for obtaining
%                     the next window
% Output
% DataWin  - Cell with matrix of data indexed by the number of windows
% inipoint - Vector - window index start
% endpoint - Vector - window index end
% center   - Vector - window index center (median)

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

% Windowing
[Nsamples,Nvariables] = size(Data);
Npw = NwindowPoints;

FlagEnd = 1; % Flag for while 
             
kk = 1;      % window index

while FlagEnd == 1
    
    inipoint(kk) = (shift*(kk-1) + 1);
    endpoint(kk) = inipoint(kk) + Npw - 1;
    center(kk)   = ceil(median(inipoint(kk):endpoint(kk)));
    
    if endpoint(kk) >=  Nsamples
        
        endpoint(kk) = Nsamples;
        DataWin{kk}  = Data(inipoint(kk):end,:);
        center(kk)   = ceil(median(inipoint(kk):endpoint(kk)));
    
        break
    end
    
    DataWin{kk}   = Data(inipoint(kk):endpoint(kk),:);
    center(kk)    = floor(median(inipoint(kk):endpoint(kk)));
    kk = kk + 1;
end
