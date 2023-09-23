from logger import App_Logger
from flask import Response
from Predict.predicted_data_validation import Predicted_data_validation
from Predict.DataTransformation_Prediction.dataTransformationPrediction import dataTransformation
from Predict.DbTableCreation_Prediction.dbTableCreationPrediction import dBOperation

class pred_validation:
    def __init__(self, path):
        self.raw_data = Predicted_data_validation(path)
        self.log_writer = App_Logger()
        self.file_object = open("Logs/prediction_log.txt", "a+")
        self.dataTransformation = dataTransformation()
        self.dBOperation = dBOperation()

    def prediction_validation(self):
        try:
            name_pattern, LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns, ColName =self.raw_data.validateSchema()
            self.raw_data.validateNamePattern(name_pattern,LengthOfDateStampInFile,LengthOfTimeStampInFile)
            self.raw_data.validateNumberofColumns(NumberofColumns, ColName)
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.file_object, "Raw Data Validation Complete Successfully!!")
            self.log_writer.log(self.file_object, "Starting Data Transformation!!")
            self.dataTransformation.replaceMissingWithNull()
            self.log_writer.log(self.file_object, "Data Transformation completed Successfully!!")
            self.log_writer.log(self.file_object, "Creating Data base for the Valid predicted data!!")
            self.dBOperation.createTableDb('Prediction', ColName)
            self.log_writer.log(self.file_object, "Database with table PREDICTION_DATA created successfully!!")
            self.log_writer.log(self.file_object, "Inserting the prediction data into table PREDICTION_DATA")
            self.dBOperation.insertPredictionData('Prediction')
            self.log_writer.log(self.file_object, "Inserting of data completed successfully!!")

        except ValueError:
            return Response("Error Occurred! %s" % ValueError)
        return True
