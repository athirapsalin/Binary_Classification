import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.svm import OneClassSVM
from numpy import quantile, where, random
from sklearn.manifold import TSNE
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score, recall_score
from sklearn.metrics import roc_auc_score


class PREPROCESSING:
    
    def __init__(self, task1_data_path, task2_data_path):
        """
        
        This class give information about the datasets.
        
        """
        self.df_task1 = pd.read_csv(task1_data_path, sep=';')
        self.df_task2 = pd.read_csv(task2_data_path, sep=';')
    
    def handle_duplicates_complete(self):
        """
        This method handles the duplicates in the datasets
        
        """
        self.df_task1 = self.df_task1.drop_duplicates()
        self.df_task2 = self.df_task2.drop_duplicates()
    
    def handle_duplicates_ID(self):
        """
        This method handles the duplicates in 'ID' before 
        joing the dataset
        
        """
        
        self.df_task1 = self.df_task1.drop_duplicates("ID")
        self.df_task2 = self.df_task2.drop_duplicates("ID")
        
    def merge_data(self):
        
        """
        Merge the datasets on 'ID'
        
        """
        self.df_merged = pd.merge(self.df_task1,self.df_task2,on='ID')
        self.df_original =self.df_merged.copy()
        
    def handle_missing_values(self):
        """
        Remove the column 'ZIK' with majority values missing
        
        """
        del self.df_merged['ZIK']
        
        
    def impute_missing_values(self):
        """
        
        Impute the missing values in the numerical columns with mean and 
        categorical column with mode
        
        """
        self.num_vars = self.df_merged.select_dtypes(include='number').columns.tolist()
        num_imputer = SimpleImputer(strategy='mean')
        self.df_merged[self.num_vars] = num_imputer.fit_transform(self.df_merged[self.num_vars])
        self.cat_vars = self.df_merged.select_dtypes(include='object').columns.tolist()
        cat_imputer = SimpleImputer(strategy='most_frequent')
        self.df_merged[self.cat_vars] = cat_imputer.fit_transform(self.df_merged[self.cat_vars])
        
    def handle_imbalance(self, oversampling_ratio):
        """
        Handle the imbalance in the target by oversampling
        """
        self.minority_class = self.df_merged[self.df_merged['Type'] == 'n']
        self.majority_class = self.df_merged[self.df_merged['Type'] == 'y']
        self.df_balanced = pd.concat([self.majority_class] + [self.minority_class]*int(oversampling_ratio))

    def one_hot_encode(self):
        """
        One hot encode the categorical features 
        Retain the target variable unchanged
        
        """
        
        self.df_one_hot_encoded = pd.get_dummies(self.df_balanced, columns=self.cat_vars[:-1])*1
        self.df_before_transform = self.df_one_hot_encoded.copy()
    def scale_numerical_features(self):
        """
        Scale the numerical variables 
        
        """
        self.df_num = self.df_one_hot_encoded[self.num_vars[1:]]
        scaler = MinMaxScaler()
        self.scaled_data = scaler.fit_transform(self.df_num)
        self.df_one_hot_encoded[self.num_vars[1:]] = pd.DataFrame(self.scaled_data, columns=self.num_vars[1:])
        #self.df_before_transform = self.df_one_hot_encoded.copy()
    def log_transform(self,log_list):
        """
        log transform done to remove skewness
        
        """
        for col in log_list:
             self.df_one_hot_encoded[col] = np.log(self.df_one_hot_encoded[col])
        self.df_one_hot_encoded = self.df_one_hot_encoded.replace([np.inf, -np.inf],0)
    def remove_correlated(self, threshold):
        """
        Consider columns other than 'ID' and 'Type'
        Remove the correlated feature using a threshold 
        
        """
        
        self.df_dropped = self.df_one_hot_encoded.drop(['ID', 'Type'], axis=1)
        self.df_corr = self.df_dropped.corr(method='pearson', min_periods=1)
        self.df_not_correlated = ~(self.df_corr.mask(np.tril(np.ones([len(self.df_corr)]*2, dtype=bool))).abs() > threshold).any()
        self.un_corr_idx = self.df_not_correlated.loc[self.df_not_correlated[self.df_not_correlated.index] == True].index
        self.df_uncorr = self.df_dropped[self.un_corr_idx]
        self.df_uncorr['ID']=self.df_one_hot_encoded['ID']
        self.df_uncorr['Type']=self.df_one_hot_encoded['Type']
        
    def drop_feature_importance(self , feature_imp_threshold):
        """
        Drop the feature with less importance
        
        """
        
        self.X = self.df_uncorr[self.df_uncorr.columns[:-2]]
        self.y = self.df_uncorr[['Type']]
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(self.X, self.y)
        self.feature_imp = clf.feature_importances_  
        self.model = SelectFromModel(clf, prefit=True, threshold= feature_imp_threshold)
        self.X_new = self.model.transform(self.X)
        
    def optimal_feature(self): 
        """
        Retain the features with importance
        """
        self.selected_feature_indices = self.model.get_support()
        self.selected_feature_names = self.df_uncorr.columns[:-2][self.selected_feature_indices]
        self.df_independent_feature = pd.DataFrame(data=self.X_new, columns=self.selected_feature_names)
                       
    def target_map(self):
        """
        Map the target column 'Type' from ['n', 'y'] to ['0', '1']
        """
        self.y['Type'] = self.y.Type.replace(to_replace=['n', 'y'], value=[0, 1])
        self.df_label = self.y
        
    def process(self, oversampling_ratio = 1, log_list=['CDx','POUG','TRE'], threshold = 0.2, feature_imp_threshold = 5e-3):
        """
        Process all the methods inside the class
        """
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
        return self.df_independent_feature, self.df_label
    
class Model_Selection:
    def __init__(self, df_independent_feature, df_label):
        """
        Take the two input df_independent_features and df_label. 
        """
        self.df_independent_feature = df_independent_feature
        self.df_label = df_label
        pass
    
    def outlier_detection_ocsvm(self):
        """
        Reduce the independent features to two dimensions.
        Perform one class SVM, detect the outliers and remove the outliers        
        """
        self.x_reduced = TSNE(n_components=2, random_state=0).fit_transform(self.df_independent_feature)
        svm = OneClassSVM( nu=0.005, gamma=1e-04)
        svm.fit(self.x_reduced)
        self.x_predicted = svm.predict(self.x_reduced)
        self.df_independent_no_out=self.df_independent_feature[self.x_predicted==1]
        self.df_label_no_out=self.df_label[self.x_predicted==1]  
        
    def grid_search(self):
        """
        Split the independent features and label to train and test set.
        Consider Logistic regression, Random Forest and SVM model for binary classificatio.
        Perform 5 fold cross validation and grid search to find the best model 
        and the corresponding hyperparameters
        """
        
        X_train, X_test, y_train, y_test = train_test_split(self.df_independent_no_out, self.df_label_no_out, test_size=0.2, random_state=42)
        logreg_param_grid = {'C': [0.1, 1, 10],'class_weight': [ 'balanced']}
        rf_param_grid = {'min_samples_split': [10, 20, 30],'n_estimators': [ 100, 200], 'max_depth': [ 4, 5, 6, 7],'class_weight': [ 'balanced']}
        svm_param_grid = {'probability':[True],'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'],'class_weight': [ 'balanced']}
        logreg = LogisticRegression()
        rf = RandomForestClassifier()
        svm = SVC(probability=True)
        gb = GradientBoostingClassifier()
        models = {'Logistic Regression': (logreg, logreg_param_grid),
                  'Random Forest': (rf, rf_param_grid),
                  'SVM': (svm, svm_param_grid)}

        self.model_results = {}
        self.roc_curves = {}
        for model_name, (model, param_grid) in models.items():
            print(f"Performing grid search for {model_name}")
            self.model_results[model_name] = {}

            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict_proba(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test['Type'].values, y_pred.T[1], pos_label=1)
            test_accuracy = metrics.auc(fpr, tpr)
            
            
            y_pred_train = best_model.predict_proba(X_train)
            fpr, tpr, thresholds = metrics.roc_curve(y_train['Type'].values, y_pred_train.T[1], pos_label=1)
            train_accuracy = metrics.auc(fpr, tpr)

            self.model_results[model_name]["best_model"] = best_model
            self.model_results[model_name]["params"] = grid_search.best_params_
            self.model_results[model_name]["test_accuracy"] = test_accuracy
            self.model_results[model_name]["train_accuracy"] = train_accuracy
            self.model_results[model_name]["train_accuracy"] = pd.DataFrame(grid_search.cv_results_)   
            self.roc_curves[model_name] = {'fpr': fpr, 'tpr': tpr}
    
    def best_model_score(self):
        """
        Consider the best model after the grid search and calculate evaluation metrics.
        - accuracy
        - precision
        -recall
        - F1
        - ROC curve
        Find the threshold value
        
        Perform thresholding and calculate the evaluation metrics.
        
        """
        X_train, X_test, y_train, y_test = train_test_split(self.df_independent_no_out, self.df_label_no_out, test_size=0.2, random_state=42)
        
        best_model = self.model_results['Random Forest']['best_model']
        
        
        best_model.fit(X_train, y_train)

        self.y_true =y_test.copy()
        y_pred = best_model.predict(X_test)
        y_pred_tr = best_model.predict(X_train)
        
        
        self.cm_train = confusion_matrix(y_train, y_pred_tr)
        self.cm_test = confusion_matrix(y_test, y_pred)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        
        out_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [accuracy, precision, recall, f1]

        self.metrics_df = pd.DataFrame({'Metrics': out_metrics, 'Values': values})
        
        y_pred = best_model.predict_proba(X_test)
        self.y_pred_prob =y_pred.copy()
        self.fpr_ts, self.tpr_ts, thresholds = metrics.roc_curve(y_test['Type'].values, y_pred.T[1], pos_label=1)
        self.roc_auc_test = metrics.auc(self.fpr_ts, self.tpr_ts)
        
        self.metrics_df.loc[len(self.metrics_df)] = ['AUC', self.roc_auc_test]
        
        y_pred_train = best_model.predict_proba(X_train)
        self.fpr_tr, self.tpr_tr, thresholds = metrics.roc_curve(y_train['Type'].values, y_pred_train.T[1], pos_label=1)
        self.roc_auc_train = metrics.auc(self.fpr_tr, self.tpr_tr)
        
    def thresholding(self):
        """
        Find the threshold value from the evaluation metrics scores
        """
        thresholds = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        y_true = self.y_true
        y_pred_prob = self.y_pred_prob.T[1]
        for threshold in np.arange(0.1, 0.5, 0.01):
            y_pred = (y_pred_prob >= threshold)*1
            
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)

            thresholds.append(threshold)
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            self.threshold_df = pd.DataFrame({'Threshold': thresholds, 'F1': f1_scores, 'Precison': precision_scores, 'Recall': recall_scores})
    def perform_threshold(self):
        """
        Perform thresholding and find the new evaluation metric scores
        """
        threshold = 0.4  
        y_pred = (self.y_pred_prob.T[1] >= threshold)*1 #astype(int)
        y_true = self.y_true
        
        metrics = {
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1 Score': f1_score(y_true, y_pred),
            'Accuracy': accuracy_score(y_true, y_pred)
        }

        
        self.thresholded_metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
        
        
        pass
    
    def process(self):
        self.outlier_detection_ocsvm()
        self.grid_search()
        self.best_model_score()
        self.thresholding()
        self.perform_threshold()
        
        
            
    


















































        
              
 