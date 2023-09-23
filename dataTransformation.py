from os import listdir
import pandas as pd
from logger import App_Logger
from flask import Response

class DataTransformation:

    def __init__(self,perform):
        self.log_writer = App_Logger()
        if perform == "train":
            self.log_file = "Logs/train_log.txt"
            self.good_data_path = "Clean_batch_files/Training/good/"
        else:
            self.log_file = "Logs/prediction_log.txt"
            self.good_data_path = "Clean_batch_files/Prediction/good/"

    def replace_missing_with_null(self):
        try:
            logFile = open(self.log_file, "a+")
            self.log_writer.log(logFile, "Data Transformation by updating the null values!!")
            for file in [fl for fl in listdir( self.good_data_path)]:
                csv_df = pd.read_csv(self.good_data_path+file)
                csv_df.fillna("NULL", inplace=False)
                csv_df['Wafer'] = csv_df['Wafer'].str[6:]
                csv_df.to_csv(self.good_data_path+file,index=None, header=True)
            self.log_writer.log(logFile, "Data Transformation completed successfully!!")
            logFile.close()
        except ValueError:
            logFile.close()
            return Response("Error Occurred! %s" % ValueError)