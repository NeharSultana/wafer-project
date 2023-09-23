from os import listdir
import pandas as pd
from flask import Response
from logger import App_Logger
import sqlite3
import os
import csv

class DB_Operation:

    def __init__(self,perform):
        self.dbPath = "DB_Connection/"
        self.log_writer = App_Logger()
        self.final_csv = "Final_csv/"
        self.task = perform
        if self.task == "train":
            self.log_file = "Logs/train_log.txt"
            self.good_data_path = "Clean_batch_files/Training/good/"
            self.databaseName = "TRAIN"
            self.table_name="TRAIN_DATA"
        else:
            self.log_file = "Logs/prediction_log.txt"
            self.good_data_path = "Clean_batch_files/Prediction/good/"
            self.databaseName = "PREDICTION"
            self.table_name = "PREDICTION_DATA"

    def create_table_db(self,colNames):
        file_log = open(self.log_file, "a+")
        self.log_writer.log(file_log, f"Creating table {self.table_name} in Database {self.databaseName} !!")
        try:
            self.log_writer.log(file_log, "Connecting to DB")
            conn = sqlite3.connect(self.dbPath+self.databaseName+".db")
            conn.execute(f"DROP TABLE IF EXISTS {self.table_name};")
            for key, dtype in colNames.items():
                try:
                    conn.execute(f"ALTER TABLE {self.table_name} ADD '{key}' '{dtype}'")
                except:
                    conn.execute(f"CREATE TABLE {self.table_name} ({key} {dtype})")
            conn.commit()
            conn.close()
            self.log_writer.log(file_log, "Table creation Successful !!")
            file_log.close()
        except ConnectionError:
            self.log_writer.log(file_log, "Error while connecting to database: %s" % ConnectionError)
            file_log.close()
            raise ConnectionError
        except ValueError:
            file_log.close()
            return Response("Error Occurred! %s" % ValueError)

    def insert_data_db(self):
        file_log = open(self.log_file, "a+")
        self.log_writer.log(file_log, f"Inserting data into {self.table_name} in Database {self.databaseName} !!")
        try:
            self.log_writer.log(file_log, "Connecting to DB")
            conn = sqlite3.connect(self.dbPath+self.databaseName+".db")
            for f in listdir(self.good_data_path):
                csv_df = pd.read_csv(self.good_data_path+f)
                csv_df.to_sql(self.table_name, conn,if_exists='append',index=False)
                self.log_writer.log(file_log, f"{f} inserted successfully!!")
            conn.commit()
            conn.close()
            self.log_writer.log(file_log, "Inserted data Successfully !!")
            file_log.close()
        except ConnectionError:
            self.log_writer.log(file_log, "Error while connecting to database: %s" % ConnectionError)
            file_log.close()
            raise ConnectionError

    def get_db_data_to_csv(self):
        file_log = open(self.log_file, "a+")
        self.log_writer.log(file_log, "Pulling the data from database!!")
        try:
            self.log_writer.log(file_log, "Connecting to DB")
            conn = sqlite3.connect(self.dbPath + self.databaseName + ".db")
            sqlQuery = f"SELECT * FROM {self.table_name}"
            cursor = conn.cursor()
            result = cursor.execute(sqlQuery).fetchall()
            if not os.path.isdir(self.final_csv):
                os.makedirs(self.final_csv)
            headers = [i[0] for i in cursor.description]
            csvFile = csv.writer(open(self.final_csv+self.task+"_Input.csv", 'w', newline=''), delimiter=',', lineterminator='\r\n', quoting=csv.QUOTE_ALL, escapechar='\\')
            csvFile.writerow(headers)
            csvFile.writerows(result)
            self.log_writer.log(file_log, f"Transferred the data to csv file {self.final_csv+self.task+'_Input.csv'} !!")
            file_log.close()
        except ConnectionError:
            self.log_writer.log(file_log, "Error while connecting to database: %s" % ConnectionError)
            file_log.close()
            raise ConnectionError