function [TimeStamps] = GetTimeLogIntervals(Conf,phys,labels) 

% #########################################################################
% Return a structure with time and indexes associated a given
% condition (premed or postmed) and a given label (tremor or activity)

% Input - Conf: Structure

%         Conf.current_id_labels - patient id on the provided label structure;
%         Conf.KindofLabel       - tremor or actv (activity);
%         Conf.Condition         - premed or postmed;
%         Conf.current_id_phys   - patient id on the provided data structure;
%         Conf.ind_arm           - arm of interest;
%         phys                   - provided Data recoded structure;
%         labels                 - provided Label structure;

% Output - TimeStamps: Structure

%          TimeStamps.Start - Time that labelled data starts;
%          TimeStamps.End   - Time that labelled data ends;
%          TimeStamps.LogicalInterval - Logical variable indicating the
%          presence (1) or not (0) of relevant label data;
%          TimeStamps.Time      - Time for relevant label interval
%          TimeStamps.IndVector - Index vector for relevant label interval

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

current_id_labels = Conf.current_id_labels;
KindofLabel       = Conf.KindofLabel;
Condition         = Conf.Condition;
current_id_phys   = Conf.current_id_phys;
ind_arm           = Conf.ind_arm;

switch KindofLabel
    
    case 'tremor' % define the kind of label
        
        switch Condition
            
            case 'premed' % define the condition of interest
                
                fieldstart = 'premed_tremorstart';
                fieldend   = 'premed_tremorend';
                
            case 'postmed'
                
                fieldstart = 'postmed_tremorstart';
                fieldend   = 'postmed_tremorend';
                
        end
        
    case 'actv'
        
        switch Condition
            
            case 'premed'
                
                fieldstart = 'premedstart';
                fieldend   = 'premedend';
                
            case 'postmed'
                
                fieldstart = 'postmedstart';
                fieldend   = 'postmedend';
                
            case 'pre'

                fieldstart = 'prestart';
                fieldend  = 'preend';

            case 'post'

                fieldstart = 'poststart';
                fieldend = 'postend';

        end
end

TimeStamps.Start =  labels(current_id_labels).(fieldstart); % interval of interest start
TimeStamps.End   =  labels(current_id_labels).(fieldend);   % interval of interest end
% Take interval of interest
TimeStamps.LogicalInterval = (phys(current_id_phys).(ind_arm).accel(:, 1) > TimeStamps.Start & phys(current_id_phys).(ind_arm).accel(:, 1) < TimeStamps.End);
TimeStamps.Time      = phys(current_id_phys).(ind_arm).accel(TimeStamps.LogicalInterval, 1);
TimeStamps.IndVector = find(TimeStamps.LogicalInterval == 1);
