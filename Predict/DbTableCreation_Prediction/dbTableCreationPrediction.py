from os import listdir
import pandas as pd
from flask import Response
from logger import App_Logger
import sqlite3

class dBOperation:

    def __init__(self):
        self.goodFilePath = "Predict_Validated_file/good/"
        self.log_file = "Logs/prediction_log.txt"
        self.dbPath = "Db_Connection/"
        self.log_writer = App_Logger()

    def createTableDb(self, databaseName,colNames):
        try:
            file_log = open(self.log_file, "a+")
            self.log_writer.log(file_log, "Connecting DB")
            conn = sqlite3.connect(self.dbPath+databaseName+".db")
            conn.execute("DROP TABLE IF EXISTS PREDICTION_DATA;")

            for key, dtype in colNames.items():
                try:
                    conn.execute(f"ALTER TABLE PREDICTION_DATA ADD '{key}' '{dtype}'")
                except:
                    conn.execute(f"CREATE TABLE PREDICTION_DATA ({key} {dtype})")
            conn.commit()
            conn.close()
            file_log.close()
        except ConnectionError:
            self.log_writer.log(file_log, "Error while connecting to database: %s" % ConnectionError)
            file_log.close()
            raise ConnectionError
        except ValueError:
            file_log.close()
            return Response("Error Occurred! %s" % ValueError)

    def insertPredictionData(self,databaseName):
        try:
            file_log = open(self.log_file, "a+")
            self.log_writer.log(file_log, "Connecting DB")
            conn = sqlite3.connect(self.dbPath+databaseName+".db")
            for f in listdir(self.goodFilePath):
                csv_df = pd.read_csv(self.goodFilePath+f)
                csv_df.to_sql("PREDICTION_DATA", conn,if_exists='replace',index=False)
                self.log_writer.log(file_log, f"{f} inserted successfully!!")
            conn.commit()
            conn.close()
        except ConnectionError:
            self.log_writer.log(file_log, "Error while connecting to database: %s" % ConnectionError)
            file_log.close()
            raise ConnectionError
