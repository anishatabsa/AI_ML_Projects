% Prolog rules generated from decision tree
% |--- OverTime <= 0.50
% |   |--- TotalWorkingYears <= 2.50
% |   |   |--- Department <= 0.50
% |   |   |   |--- Education <= 1.50
% |   |   |   |   |--- class: 0
% |   |   |   |--- Education >  1.50
% |   |   |   |   |--- class: 1
% |   |   |--- Department >  0.50
% |   |   |   |--- MaritalStatus <= 1.50
% |   |   |   |   |--- class: 0
% |   |   |   |--- MaritalStatus >  1.50
% |   |   |   |   |--- class: 0
% |   |--- TotalWorkingYears >  2.50
% |   |   |--- NumCompaniesWorked <= 4.50
% |   |   |   |--- YearsAtCompany <= 38.50
% |   |   |   |   |--- class: 0
% |   |   |   |--- YearsAtCompany >  38.50
% |   |   |   |   |--- class: 1
% |   |   |--- NumCompaniesWorked >  4.50
% |   |   |   |--- Age <= 33.50
% |   |   |   |   |--- class: 0
% |   |   |   |--- Age >  33.50
% |   |   |   |   |--- class: 0
% |--- OverTime >  0.50
% |   |--- MonthlyIncome <= 2475.00
% |   |   |--- DistanceFromHome <= 15.50
% |   |   |   |--- YearsAtCompany <= 3.50
% |   |   |   |   |--- class: 1
% |   |   |   |--- YearsAtCompany >  3.50
% |   |   |   |   |--- class: 0
% |   |   |--- DistanceFromHome >  15.50
% |   |   |   |--- class: 1
% |   |--- MonthlyIncome >  2475.00
% |   |   |--- MaritalStatus <= 1.50
% |   |   |   |--- MonthlyIncome <= 3779.00
% |   |   |   |   |--- class: 0
% |   |   |   |--- MonthlyIncome >  3779.00
% |   |   |   |   |--- class: 0
% |   |   |--- MaritalStatus >  1.50
% |   |   |   |--- JobRole <= 6.50
% |   |   |   |   |--- class: 0
% |   |   |   |--- JobRole >  6.50
% |   |   |   |   |--- class: 1
% 
% Implement actual rules based on the above logic
