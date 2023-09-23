import json
import shutil
from logger import App_Logger
from flask import Response
import re
from os import listdir
import os
import pandas as pd


class Predicted_data_validation:
    def __init__(self, path):
        self.path = path
        self.schema_path = 'schema/schema_prediction.json'
        self.log_writer = App_Logger()
        self.log_file = "Logs/prediction_log.txt"
        
    def validateSchema(self):
        try:
            self.log_writer.log(open(self.log_file,"a+"), "STEP 1 :: Validating Expected Schema")
            with open(self.schema_path, 'r') as f:
                data_info = json.load(f)
                f.close()
            name_pattern = data_info["SampleFileName"]
            LengthOfDateStampInFile = data_info["LengthOfDateStampInFile"]
            LengthOfTimeStampInFile = data_info['LengthOfTimeStampInFile']
            NumberofColumns = data_info['NumberofColumns']
            ColName = data_info['ColName']

            file=open("Logs/prediction_log.txt", "a+")
            self.log_writer.log(file,f"File name pattern::{name_pattern}"
                                         f"||Date length::{LengthOfDateStampInFile}"
                                         f"||Time length::{LengthOfTimeStampInFile}"
                                         f"||No of Columns::{NumberofColumns}"
                                        f"||Column names::{ColName}"
                                    )
        except ValueError:
            return Response("Error Occurred! %s" % ValueError)

        return name_pattern,LengthOfDateStampInFile,LengthOfTimeStampInFile,NumberofColumns,ColName

    def validateNamePattern(self,name_pattern,LengthOfDateStampInFile,LengthOfTimeStampInFile):
        regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"
        self.cleanDirectory(self.path)
        good = 0
        bad = 0
        try:
            file = open("Logs/prediction_log.txt", "a+")
            self.log_writer.log(file,"STEP 2 :: Validating the file names")
            for f in [f for f in listdir(self.path)]:
                split1 = re.split('.csv',f)
                split2 = re.split('_',split1[0])
                if (re.match(regex, f) and split2[0] == "wafer" and len(split2[1]) == LengthOfDateStampInFile and len(split2[2]) == LengthOfTimeStampInFile):
                    good+=1
                    self.log_writer.log(file, self.moveToFolder("good", f, self.path, False))
                else:
                    bad+=1
                    self.log_writer.log(file,"Invalid file name!! "+self.moveToFolder("bad", f, self.path, False))
            self.log_writer.log(file,f"Total files:{good+bad}::Valid name:{good}::Invalid name:{bad}")
            file.close()

        except ValueError:
            file.close()
            return Response("Error Occurred! %s" % ValueError)

    def validateNumberofColumns(self,NumberofColumns, ColName):
        good = 0
        bad = 0
        try:
            file = open("Logs/prediction_log.txt", "a+")
            self.log_writer.log(file, "STEP 3 :: Validating the number of columns and there names")
            good_file_path = "Predict_Validated_file/good/"
            for f in listdir(good_file_path):
                csv_data= pd.read_csv(good_file_path+f)
                csv_data.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                csv_data.to_csv(good_file_path + f, index=None, header=True)
                if (csv_data.shape[1] == NumberofColumns and [*ColName] == csv_data.columns.tolist()):
                    good+=1
                else:
                    bad += 1
                    self.log_writer.log(file, "Invalid number of columns!! " + self.moveToFolder("bad",f,good_file_path, True))
            self.log_writer.log(file, f"Total files:{good + bad}::Valid name:{good}::Invalid name:{bad}")
            file.close()
        except ValueError:
            file.close()
            return Response("Error Occurred! %s" % ValueError)


    def validateMissingValuesInWholeColumn(self):
        try:
            file = open("Logs/prediction_log.txt", "a+")
            self.log_writer.log(file, "STEP 4 :: Validating missing values in columns")
            good_file_path = "Predict_Validated_file/good/"
            for f in listdir(good_file_path):
                csv_data = pd.read_csv(good_file_path + f)
                for column in csv_data:
                    if (csv_data[column].count() == 0):
                        self.log_writer.log(file,"Has missing column!! " + self.moveToFolder("bad", f, good_file_path, True))
                        break
            file.close()
        except ValueError:
            file.close()
            return Response("Error Occurred! %s" % ValueError)



    def moveToFolder(self, type, file, path, deleteFromSource):
        source =path + "/" + file
        destination = self.createFolder(type,file)
        shutil.copy(source, destination)
        if (deleteFromSource):
            os.remove(path + "/" + file)
        return f"{file} moved to {destination}"

    def createFolder(self,type,file):
        newpath = r'Predict_Validated_file/'
        if not os.path.exists(newpath+type):
            os.makedirs(newpath+type)
        return newpath+type



    def cleanDirectory(self, path):
        newpath = r'Predict_Validated_file/'
        if os.path.exists(newpath):
            shutil.rmtree(newpath)
