from logger import App_Logger
from flask import Response
from dataValidation import DataValidation
from dataTransformation import DataTransformation
from dataDatabaseOperations import DB_Operation

class Data_Preprocessing:
    def __init__(self, path, perform):
        self.raw_data = DataValidation(path,perform)
        self.dataTransformation = DataTransformation(perform)
        self.dBOperation = DB_Operation(perform)
        self.log_writer = App_Logger()

    def preprocess(self):
        try:
            name_pattern, LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns, ColName =self.raw_data.validate_schema()
            self.raw_data.validateNamePattern(name_pattern,LengthOfDateStampInFile,LengthOfTimeStampInFile)
            self.raw_data.validateNumberofColumns(NumberofColumns, ColName)
            self.raw_data.validateMissingValuesInWholeColumn()
            self.dataTransformation.replace_missing_with_null()
            self.dBOperation.create_table_db(ColName)
            self.dBOperation.insert_data_db()
            self.dBOperation.get_db_data_to_csv()
        except ValueError:
            return Response("Error Occurred! %s" % ValueError)
        return True