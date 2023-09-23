from logger import App_Logger
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from kneed import KneeLocator
import bestModel
import os
import pickle
import json

class Model_Preprocessing:
    def __init__(self,perform):
        self.log_writer = App_Logger()
        self.task=perform
        self.final_csv = "Final_csv/"
        self.wafer_data=None
        if self.task == "train":
            self.log_file = "Logs/train_log.txt"
        else:
            self.log_file = "Logs/prediction_log.txt"

    def trainModel(self):
        self.file_log = open(self.log_file, "a+")
        self.log_writer.log(self.file_log, "Training Started !!")
        self.log_writer.log(self.file_log, "Reading the complete data...")
        self.wafer_data = pd.read_csv(self.final_csv+"train_Input.csv")
        self.wafer_data = self.removeColums(self.wafer_data)
        X,y=self.removeLabel(self.wafer_data)
        X_data = self.removeNullValues(X)
        X_data = self.RemoveColwith0Dev(X_data)
        cluster = self.createElbow(X_data)
        X_data=self.create_clusters(X_data, cluster)
        X_data['Label']=y
        list_of_clusters = X_data['cluster'].unique()
        self.model_creation(X_data,list_of_clusters)

    def predict(self):
        self.file_log = open(self.log_file, "a+")
        self.log_writer.log(self.file_log, "Prediction Started !!")
        self.log_writer.log(self.file_log, "Reading the complete data...")
        self.wafer_data = pd.read_csv(self.final_csv + "predict_Input.csv")
        wafer_names = list(self.wafer_data['Wafer'])
        self.wafer_data = self.removeColums(self.wafer_data)
        X_data = self.removeNullValues(self.wafer_data)
        X_data = self.RemoveColwith0Dev(X_data)
        kcluster_model = self.load_model("Kmeans.pkl")
        predict_clust = kcluster_model.predict(X_data)
        X_data['cluster'] = predict_clust
        list_of_clusters = X_data['cluster'].unique()
        return self.model_prediction(X_data, list_of_clusters,wafer_names)

    def model_prediction(self, X_data,list_of_clusters,wafer_names):
        path="Prediction_Output_File/Predictions.csv"
        for i in list_of_clusters:
            cluster_data = X_data[X_data['cluster'] == i]
            wafer_names = wafer_names
            cluster_data = cluster_data.drop(['cluster'], axis=1)
            model = self.load_model("randomForest"+str(i)+".pkl")
            predict = list(model.predict(cluster_data))
            result = pd.DataFrame(list(zip(wafer_names, predict)), columns=['Wafer', 'Prediction'])
            if not os.path.exists("Prediction_Output_File"):
                os.makedirs("Prediction_Output_File")
            result.to_csv(path, header=True, mode='a+')

        return path, result.head().to_json(orient="records")

    def load_model(self, filename):
        path = "./model"
        model = pickle.load(open(path+f"/{filename}", "rb"))
        return model

    def model_creation(self,X_data,list_of_clusters):
        print(f'List of clusters: {list_of_clusters}')
        for i in list_of_clusters:
            cluster_data = X_data[X_data['cluster'] == i]
            cluster_features = cluster_data.drop(['Label', 'cluster'], axis=1)
            cluster_label = cluster_data['Label']
            x_train,x_test,y_train,y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)
            model = bestModel.BestTuner()
            model,model_name = model.getBestModel(x_train,x_test,y_train,y_test)
            self.saveModel(model,model_name+str(i))

    def saveModel(self,bestModel,model_name):
        path = "./model"
        if not os.path.exists(path):
            os.makedirs(path)
        file = open(path+f"/{model_name}.pkl", "wb")
        pickle.dump(bestModel,file)

    def create_clusters(self,X_data,cluster):
        kmeans = KMeans(n_clusters=cluster, init='k-means++', random_state=42)
        y_predict = kmeans.fit_predict(X_data)
        self.saveModel(kmeans, "Kmeans")
        X_data['cluster'] = y_predict
        return X_data

    def createElbow(self,X_data):
        wcss = []
        for i in range(1,10):
            kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
            kmeans.fit(X_data)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1,10),wcss)
        plt.xlabel('Number of clusters')
        plt.ylabel('Total within-cluster sum of squares (WSS)')
        plt.savefig("elbow_cluster")
        kn = KneeLocator(range(1, 10), wcss, curve='convex', direction='decreasing')
        return kn.knee

    def RemoveColwith0Dev(self,X):
        dataStats=X.describe()
        col_names = dataStats.columns
        remove_col=[]
        for col in col_names:
            if dataStats[col]['std'] == 0:
                remove_col.append(col)
        if (len(remove_col) > 0 ):
            return X.drop(remove_col,axis=1)
        return X

    def removeNullValues(self,X):
        if (any(X.isna().sum() > 0)):
            return self.imputMissingValues(X)
        return X

    def imputMissingValues(self, X):
        impute=KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
        new_X=impute.fit_transform(X)
        return pd.DataFrame(data=new_X,columns=X.columns).astype(np.float64)

    def removeLabel(self, wafer_data):
        x=wafer_data.iloc[:,wafer_data.columns!='Output']
        y=wafer_data.iloc[:,wafer_data.columns=='Output']
        return x,y

    def removeColums(self,wafer_data):
        self.file_log = open(self.log_file, "a+")
        self.log_writer.log(self.file_log, "Removing Wafer Column")
        return wafer_data.iloc[:, wafer_data.columns != 'Wafer']