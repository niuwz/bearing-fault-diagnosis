load('data/Fault_Diag_Data.mat')
for i=1:9
writetable(TrainData{i}.data,['data/',TrainData{i}.label,'.csv'])
end