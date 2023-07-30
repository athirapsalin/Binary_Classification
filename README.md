Task 1
This function calculates the hour difference between two time stamps.
hour_difference(start, end, weekday_cal)

How to use
Takes 3 input
           start_timestamp = "%Y/%m/%d %H:%M" 
           end_timestamp = "%Y/%m/%d %H:%M"
            weekday_cal = Bool
            
If weekday_cal is set True it will calculate the time difference in hours on only weekdays between the window 9.00 to 5.00.
If weekday_cal is set False it will calculate the hour difference between the timestamps in hours
example input: hour_difference("2023/02/18 12:04","2023/02/18 19:00", weekday_cal =True)
expected output: 5
example input: hour_difference("2023/02/18 12:04","2023/02/18 19:00", weekday_cal =False)
expected output: 7

Task 2
This is a binary classification task utilising two input datasets.
Datasets -task1_1.csv, task1_2.csv

main.py

It calls the src.py file and imports the classes PREPROCESSING and Model_Selection from it
prints the metrics defined in Model_Selection as ouput.

How to use
to run:- python3 main.py 

src.py - 

 imported all the packages
 
 defined two class PREPROCESSING and Model_Selection
 PREPROCESSING has methods given below.
        self.handle_duplicates_complete()
        self.handle_duplicates_ID()
        self.merge_data()
        self.impute_missing_values()
        self.handle_imbalance(oversampling_ratio)
        self.one_hot_encode()
        self.log_transform(log_list)
        self.scale_numerical_features()
        self.remove_correlated(threshold)
        self.drop_feature_importance( feature_imp_threshold)
        self.optimal_feature()
        self.target_map()
        
PREPROCESSING return self.df_independent_feature, self.df_label

Model_Selection has methods
        self.outlier_detection_ocsvm()
        self.grid_search()
        self.best_model_score()
        self.thresholding()
        self.perform_threshold()
                
class helps in writing clean, modular, and maintainable code. 

task2.py.ipynb

This notebook has all the packages imported and has the class PREPROCESSING and Model_Selection
How to use- variables inside the methods which are defined for example self.variable_name can be called utilized outside the class

The notebook generates all the plots which are used to solve the problem statement.


