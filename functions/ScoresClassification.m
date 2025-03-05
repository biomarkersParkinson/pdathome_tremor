function [TestPerform] = ScoresClassification(ScoresTrain,YTrain,ScoresTest,YTest)

% #########################################################################

% Assess the training and testing performance based on predicted values
% (ScoresTrain and ScoresTest) and true values (YTrain and YTest)

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

[XpfTrain,YpfTrain,TpfTrain,AUCpfTrain] = perfcurve(YTrain,...
    ScoresTrain,'1','XVals',0.01:0.01:1);

for kk = 1:length(XpfTrain)
    v1 = [XpfTrain(kk) YpfTrain(kk)];
    v2 = [0 1];
    distopt(kk) = norm(v2-v1);
end

[~,IndOptTrain] = min(distopt);

Threshold = TpfTrain(IndOptTrain);

YhatTest = double(ScoresTest > Threshold);

TestPerform.Prediction  = YhatTest;

TestPerform.TrueLabels   = YTest;

TestPerform.Scores      = ScoresTest;

[XpfTest,YpfTest,TpfTest,AUCpfTest] = perfcurve(YTest,...
    ScoresTest,'1','XVals',0.01:0.01:1);

indtestvector = find(TpfTest <= Threshold);

indtest = indtestvector(1);

AccTest = sum(YhatTest == YTest)/length(YhatTest);

CTest = confusionmat(YTest,YhatTest);

TestPerform.AUC = AUCpfTest;

TestPerform.Thr = Threshold;

% For performance evaluation notice the correct positions
% in the Confusion Matrix for the positive Class

TruePositive = CTest(2,2); FalseNegative = CTest(2,1);

TrueNegative = CTest(1,1); FalsePositive = CTest(1,2);

TestPerform.Specificity = TrueNegative / (TrueNegative + FalsePositive); 

TestPerform.Sensitivity = TruePositive / (TruePositive + FalseNegative);

TestPerform.Precision   = TruePositive / (TruePositive + FalsePositive);

TestPerform.F1score     = 2*TruePositive / (2*TruePositive + FalsePositive + FalseNegative);

TestPerform.BalanAcc    = (TestPerform.Specificity + TestPerform.Sensitivity) / 2;

SpecificityVector = 1-XpfTest;

ind95vector = find(SpecificityVector <= 0.951);

ind95 = ind95vector(1);

TestPerform.Specificity95 = SpecificityVector(ind95);

TestPerform.SensitivityToSpecificity95 = YpfTest(ind95);

TestPerform.ThrSpec95  = TpfTest(ind95);

TestPerform.Acc       = AccTest;

TestPerform.Confusion = CTest; % Notice the positions in Confusion Matrix

TestPerform.NtrialsClass0Negative = sum(CTest(1,:));

TestPerform.NtrialsClass1Positive = sum(CTest(2,:));

TestPerform.ROC_X_Axis_FPR = XpfTest;

TestPerform.ROC_Y_Axis_TPR = YpfTest;

end
