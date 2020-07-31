%run all separately first
clear;
costFunction.fitObjectiveFunction_human;
validations_all
total

sprintf('Last term is the fourth term in J_{ZMP}: %2.1f',total(end)*100)
sprintf('Next best term is dyn condition with total: %2.1f',total(4)*100)

validations_all(end,:)*100;
validations_all(4,:)*100;

%fit LIPM cost function weights across all patients:
clear;
costFunction.fitObjectiveFunction;
sprintf('mean weights: %0.4f, %0.4f, %0.4f, %0.4f',mean(w,1))
sprintf('Percent of Correctly Predicted Preferences: %2.1f, %2.1f, %2.1f, %2.1f, %2.1f, %2.1f',cell2mat(validations)*100)
sprintf('Total percentage correct: %2.1f',val_avg*100)

%fit the LIPM cost function weights on 5 patients and test on the 6th.
clear;
costFunction.fitObjectiveFunction_leavout;
sprintf('mean weights: %0.4f, %0.4f, %0.4f, %0.4f',mean(w,1))
sprintf('Percent of Correctly Predicted Preferences: %2.1f, %2.1f, %2.1f, %2.1f, %2.1f, %2.1f',cell2mat(validations)*100)
sprintf('Total percentage correct: %2.1f',val_avg*100)

%run combined analysis:
clear;
costFunction.fitObjectiveFunction_Custom;

sprintf('mean weights: %0.4f, %0.4f, %0.4f, %0.4f, %0.4f',mean(w,1))
sprintf('Percent of Correctly Predicted Preferences: %2.1f, %2.1f, %2.1f, %2.1f, %2.1f, %2.1f',validations_all*100)
sprintf('Total percentage correct: %2.1f',total*100)


