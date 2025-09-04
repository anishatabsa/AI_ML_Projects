% Employee Attrition Prediction Rules
% Generated from Decision Tree

% Facts for categorical mappings
department(0, 'HR').
department(1, 'R&D').
department(2, 'Sales').

job_role(0, 'Analyst').
job_role(1, 'Engineer').
job_role(2, 'Manager').
job_role(3, 'Sales Rep').
job_role(4, 'Technician').

marital_status(0, 'Divorced').
marital_status(1, 'Married').
marital_status(2, 'Single').

overtime(0, 'No').
overtime(1, 'Yes').

% Main prediction rules

attrition_rule_1(Age, Department, Distance, Education, JobRole, Satisfaction, 
         Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    true, TotalWorkingYears =< 1.50, |   Age =< 33.50, |   |   OverTime =< 0.50, |   |   |   HourlyRate =< 58.50.

predict_attrition(yes, Age, Department, Distance, Education, JobRole, Satisfaction, 
                 Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    attrition_rule_1(Age, Department, Distance, Education, JobRole, Satisfaction, 
              Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

attrition_rule_2(Age, Department, Distance, Education, JobRole, Satisfaction, 
         Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    true, TotalWorkingYears =< 1.50, |   Age =< 33.50, |   |   OverTime =< 0.50, |   |   |   HourlyRate >  58.50, |   |   |   |   HourlyRate =< 79.50.

predict_attrition(no, Age, Department, Distance, Education, JobRole, Satisfaction, 
                 Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    attrition_rule_2(Age, Department, Distance, Education, JobRole, Satisfaction, 
              Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

attrition_rule_3(Age, Department, Distance, Education, JobRole, Satisfaction, 
         Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    true, TotalWorkingYears =< 1.50, |   Age =< 33.50, |   |   OverTime =< 0.50, |   |   |   HourlyRate >  58.50, |   |   |   |   HourlyRate >  79.50.

predict_attrition(yes, Age, Department, Distance, Education, JobRole, Satisfaction, 
                 Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    attrition_rule_3(Age, Department, Distance, Education, JobRole, Satisfaction, 
              Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

attrition_rule_4(Age, Department, Distance, Education, JobRole, Satisfaction, 
         Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    true, TotalWorkingYears =< 1.50, |   Age =< 33.50, |   |   OverTime >  0.50, |   |   |   MonthlyRate =< 18347.50.

predict_attrition(yes, Age, Department, Distance, Education, JobRole, Satisfaction, 
                 Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    attrition_rule_4(Age, Department, Distance, Education, JobRole, Satisfaction, 
              Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

attrition_rule_5(Age, Department, Distance, Education, JobRole, Satisfaction, 
         Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    true, TotalWorkingYears =< 1.50, |   Age =< 33.50, |   |   OverTime >  0.50, |   |   |   MonthlyRate >  18347.50.

predict_attrition(yes, Age, Department, Distance, Education, JobRole, Satisfaction, 
                 Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    attrition_rule_5(Age, Department, Distance, Education, JobRole, Satisfaction, 
              Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

attrition_rule_6(Age, Department, Distance, Education, JobRole, Satisfaction, 
         Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    true, TotalWorkingYears =< 1.50, |   Age >  33.50.

predict_attrition(no, Age, Department, Distance, Education, JobRole, Satisfaction, 
                 Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    attrition_rule_6(Age, Department, Distance, Education, JobRole, Satisfaction, 
              Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

attrition_rule_7(Age, Department, Distance, Education, JobRole, Satisfaction, 
         Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    true, TotalWorkingYears >  1.50, |   OverTime =< 0.50, |   |   WorkLifeBalance =< 1.50, |   |   |   DailyRate =< 679.00, |   |   |   |   MonthlyIncome =< 4259.50.

predict_attrition(yes, Age, Department, Distance, Education, JobRole, Satisfaction, 
                 Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    attrition_rule_7(Age, Department, Distance, Education, JobRole, Satisfaction, 
              Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

attrition_rule_8(Age, Department, Distance, Education, JobRole, Satisfaction, 
         Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    true, TotalWorkingYears >  1.50, |   OverTime =< 0.50, |   |   WorkLifeBalance =< 1.50, |   |   |   DailyRate =< 679.00, |   |   |   |   MonthlyIncome >  4259.50.

predict_attrition(no, Age, Department, Distance, Education, JobRole, Satisfaction, 
                 Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    attrition_rule_8(Age, Department, Distance, Education, JobRole, Satisfaction, 
              Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

attrition_rule_9(Age, Department, Distance, Education, JobRole, Satisfaction, 
         Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    true, TotalWorkingYears >  1.50, |   OverTime =< 0.50, |   |   WorkLifeBalance =< 1.50, |   |   |   DailyRate >  679.00, |   |   |   |   DistanceFromHome =< 7.50.

predict_attrition(no, Age, Department, Distance, Education, JobRole, Satisfaction, 
                 Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    attrition_rule_9(Age, Department, Distance, Education, JobRole, Satisfaction, 
              Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

attrition_rule_10(Age, Department, Distance, Education, JobRole, Satisfaction, 
         Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    true, TotalWorkingYears >  1.50, |   OverTime =< 0.50, |   |   WorkLifeBalance =< 1.50, |   |   |   DailyRate >  679.00, |   |   |   |   DistanceFromHome >  7.50.

predict_attrition(no, Age, Department, Distance, Education, JobRole, Satisfaction, 
                 Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    attrition_rule_10(Age, Department, Distance, Education, JobRole, Satisfaction, 
              Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

% Main prediction predicate
employee_will_leave(Employee) :-
    employee(Employee, Age, Department, Distance, Education, JobRole, Satisfaction, 
             Marital, Income, Companies, Overtime, WorkYears, CompanyYears),
    predict_attrition(yes, Age, Department, Distance, Education, JobRole, Satisfaction, 
                     Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

employee_will_stay(Employee) :-
    employee(Employee, Age, Department, Distance, Education, JobRole, Satisfaction, 
             Marital, Income, Companies, Overtime, WorkYears, CompanyYears),
    predict_attrition(no, Age, Department, Distance, Education, JobRole, Satisfaction, 
                     Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

% Sample employee facts (you can add more)
employee(john, 35, 1, 10, 3, 1, 4, 1, 75000, 2, 1, 15, 5).
employee(sarah, 28, 0, 25, 4, 0, 2, 2, 45000, 4, 1, 8, 2).

% Query examples:
% ?- employee_will_leave(john).
% ?- employee_will_stay(sarah).
% ?- predict_attrition(Result, 30, 2, 15, 3, 2, 3, 1, 60000, 3, 0, 10, 4).
