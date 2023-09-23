import json
import shutil
from logger import App_Logger
from flask import Response
import re
from os import listdir
import os
import pandas as pd

class DataValidation:
    def __init__(self, path, perform):
        self.path = path
        self.log_writer = App_Logger()
        if perform == "train":
            self.schema_path = 'schema/schema_train.json'
            self.log_file = "Logs/train_log.txt"
            self.clean_path = "Clean_batch_files/Training/"
            self.path = "Batch_files/Training_Batch_Files/"
        else:
            self.schema_path = 'schema/schema_prediction.json'
            self.log_file = "Logs/prediction_log.txt"
            self.clean_path = "Clean_batch_files/Prediction/"

    def validate_schema(self):
        try:
            logFile = open(self.log_file, "a+")
            self.log_writer.log(logFile, "Validating Expected Schema")
            with open(self.schema_path, 'r') as f:
                data_info = json.load(f)
                f.close()
            name_pattern = data_info["SampleFileName"]
            LengthOfDateStampInFile = data_info["LengthOfDateStampInFile"]
            LengthOfTimeStampInFile = data_info['LengthOfTimeStampInFile']
            NumberofColumns = data_info['NumberofColumns']
            ColName = data_info['ColName']
            self.log_writer.log(logFile,f"File name pattern::{name_pattern}"
                                         f"||Date length::{LengthOfDateStampInFile}"
                                         f"||Time length::{LengthOfTimeStampInFile}"
                                         f"||No of Columns::{NumberofColumns}"
                                        f"||Column names::{ColName}")
            self.log_writer.log(logFile,"Schema Validated Successfully")
            logFile.close()
        except ValueError:
            logFile.close()
            return Response("Error Occurred! %s" % ValueError)
        return name_pattern,LengthOfDateStampInFile,LengthOfTimeStampInFile,NumberofColumns,ColName

    def validateNamePattern(self,name_pattern,LengthOfDateStampInFile,LengthOfTimeStampInFile):
        regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"
        self.cleanDirectory()
        good = 0
        bad = 0
        try:
            logFile = open(self.log_file, "a+")
            self.log_writer.log(logFile,"Validating the file names")
            for f in [f for f in listdir(self.path)]:
                split1 = re.split('.csv',f)
                split2 = re.split('_',split1[0])
                if (re.match(regex, f) and split2[0] == "wafer" and len(split2[1]) == LengthOfDateStampInFile and len(split2[2]) == LengthOfTimeStampInFile):
                    good+=1
                    self.log_writer.log(logFile, self.moveToFolder("good", f, self.path, False))
                else:
                    bad+=1
                    self.log_writer.log(logFile,"Invalid file!! "+self.moveToFolder("bad", f, self.path, False))
            self.log_writer.log(logFile,f"Total file:{good+bad}::Valid :{good}::Invalid :{bad}")
            self.log_writer.log(logFile, "File name validation successfully completed !!")
            logFile.close()

        except ValueError:
            logFile.close()
            return Response("Error Occurred! %s" % ValueError)

    def validateNumberofColumns(self,NumberofColumns, ColName):
        good = 0
        bad = 0
        try:
            logFile = open(self.log_file, "a+")
            self.log_writer.log(logFile, "Validating the number of columns and there names!!")
            good_file_path = self.clean_path+"good/"
            for f in listdir(good_file_path):
                csv_data= pd.read_csv(good_file_path+f)
                csv_data.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                csv_data.rename(columns={"Good/Bad": "Output"}, inplace=True)
                csv_data.to_csv(good_file_path + f, index=None, header=True)
                csv_data = pd.read_csv(good_file_path + f)
                if (csv_data.shape[1] == NumberofColumns and [*ColName] == csv_data.columns.tolist()):
                    good+=1
                else:
                    bad += 1
                    self.log_writer.log(logFile, "Invalid number of columns!! " + self.moveToFolder("bad",f,good_file_path, True))
            self.log_writer.log(logFile, f"Total files:{good + bad}::Valid :{good}::Invalid :{bad}")
            self.log_writer.log(logFile, "No. of columns validated successfully !!")
            logFile.close()
        except ValueError:
            logFile.close()
            return Response("Error Occurred! %s" % ValueError)

    def validateMissingValuesInWholeColumn(self):
        try:
            logFile = open(self.log_file, "a+")
            self.log_writer.log(logFile, "Validating missing values in columns !!")
            good_file_path = self.clean_path+"good/"
            total=0
            bad=0
            for f in listdir(good_file_path):
                total += 1
                csv_data = pd.read_csv(good_file_path + f)
                for column in csv_data:
                    if (csv_data[column].count() == 0):
                        bad += 1
                        self.log_writer.log(logFile,"Has missing column!! " + self.moveToFolder("bad", f, good_file_path, True))
                        break
            self.log_writer.log(logFile, f"Total files:{total}::Valid :{total-bad}::Invalid :{bad}")
            self.log_writer.log(logFile, "Missing values updated successfully !!")
            logFile.close()
        except ValueError:
            logFile.close()
            return Response("Error Occurred! %s" % ValueError)

    def moveToFolder(self, type, file, path, deleteFromSource):
        source =path + "/" + file
        destination = self.createFolder(self.clean_path,type)
        shutil.copy(source, destination)
        if (deleteFromSource):
            os.remove(path + "/" + file)
        return f"{file} moved to {destination}"

    def createFolder(self, path,type):
        if not os.path.exists(path+type):
            os.makedirs(path+type)
        return path+type

    def cleanDirectory(self):
        if os.path.exists(self.clean_path):
            shutil.rmtree(self.clean_path)