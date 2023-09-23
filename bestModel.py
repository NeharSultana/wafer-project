from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class BestTuner:

    def __init__(self):
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def get_best_randomForest_params(self,x_train,y_train):
        self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'], "max_depth": range(2, 4, 1), "max_features": ['sqrt', 'log2']}
        self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5, verbose=3)
        self.grid.fit(x_train, y_train)
        n_estimator = self.grid.best_params_['n_estimators']
        criterion = self.grid.best_params_['criterion']
        max_depth = self.grid.best_params_['max_depth']
        max_features = self.grid.best_params_['max_features']
        model=RandomForestClassifier(n_estimators = n_estimator,
                                     criterion = criterion,
                                     max_depth =max_depth,
                                     max_features = max_features)
        model.fit(x_train, y_train)
        return model

    def get_best_xgboost_params(self,x_train,y_train):
        y_train = y_train.replace(-1, 0)
        y_train = y_train.replace(1, 1)
        self.param_grid = {'learning_rate': [0.5, 0.1, 0.01, 0.001], 'max_depth': [3, 5, 10, 20], 'n_estimators': [10, 50, 100, 200]}
        self.grid = GridSearchCV(estimator=self.xgb, param_grid=self.param_grid, cv=5, verbose=3)
        self.grid.fit(x_train, y_train)
        learning_rate = self.grid.best_params_['learning_rate']
        max_depth = self.grid.best_params_['max_depth']
        n_estimators = self.grid.best_params_['n_estimators']
        model=XGBClassifier(n_estimators = n_estimators,
                            max_depth =max_depth,
                            learning_rate = learning_rate)
        model.fit(x_train, y_train)
        return model

    def getBestModel(self, x_train, x_test, y_train, y_test):
        self.randomForestModel = self.get_best_randomForest_params(x_train, y_train)
        self.randForest_predict = self.randomForestModel.predict(x_test)
        if (len(y_test.unique() == 1)):
            self.randForest_score = accuracy_score(self.randForest_predict, y_test)
            print(f'Accuracy for RF: {self.randForest_score}')
        else:
            self.randForest_score = roc_auc_score(y_test, self.randForest_predict)
            print(f'AUC for RF: {self.randForest_score}')

        self.xgboostModel = self.get_best_xgboost_params(x_train, y_train)
        self.xgboost_predict = self.xgboostModel.predict(x_test)
        if len(y_test.unique()) == 1:
            self.xgboost_score = accuracy_score(self.xgboost_predict, y_test)
            print(f'Accuracy for XGBOOST: {self.xgboost_score}')
        else:
            self.xgboost_score = roc_auc_score(y_test, self.xgboostModel.predict_proba(x_test)[:, 1])
            print(f'AUC for XGBOOST: {self.xgboost_score}')

        # Calculate precision, recall, and F1-score for random forest model or average='micro'
        rf_precision, rf_recall, rf_f1score, _ = precision_recall_fscore_support(y_test, self.randForest_predict, average='micro')
        print(f'Random Forest: Precision: {rf_precision}, Recall: {rf_recall}, F1-score: {rf_f1score}')

        # Assuming y_test and self.randForest_predict contain the true labels and predicted labels, respectively
        confusion_rf = confusion_matrix(y_test, self.randForest_predict)
        print("Confusion Matrix for Random Forest:")
        print(confusion_rf)

        # Assuming the confusion matrix is stored in the variable 'confusion_matrix'
        sns.heatmap(confusion_rf, annot=True, cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        # Assuming y_test and self.xgboost_predict contain the true labels and predicted labels, respectively
        confusion_xgb = confusion_matrix(y_test, self.xgboost_predict)
        print("Confusion Matrix for XGBoost:")
        print(confusion_xgb)

        # Assuming the confusion matrix is stored in the variable 'confusion_matrix'
        sns.heatmap(confusion_xgb, annot=True, cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        # Assuming y_test and self.randForest_predict contain the true labels and predicted labels, respectively, for Random Forest
        fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, self.randForest_predict)

        # Assuming y_test and self.xgboost_predict contain the true labels and predicted labels, respectively, for XGBoost
        fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, self.xgboost_predict)

        # Plotting the ROC curve for Random Forest
        plt.plot(fpr_rf, tpr_rf, label='Random Forest')

        # Plotting the ROC curve for XGBoost
        plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')

        # Plotting the diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

        if len(y_test.unique()) == 2:
            rf_error_rate = 1.0 - self.randForest_score
            print(f'Error rate for RF: {rf_error_rate}')

            xgb_error_rate = 1.0 - self.xgboost_score
            print(f'Error rate for XGBoost: {xgb_error_rate}')

        if (self.randForest_score > self.xgboost_score):
            return self.randomForestModel, "randomForest"
        else:
            return self.xgboostModel, "xgboost"