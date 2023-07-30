from src import PREPROCESSING, Model_Selection
PP = PREPROCESSING('Task1_1.csv', 'Task1_2.csv')
df_independent_feature, df_label = PP.process(threshold= 0.8 )
MS = Model_Selection(df_independent_feature, df_label)
metrics = MS.process()
print(metrics)