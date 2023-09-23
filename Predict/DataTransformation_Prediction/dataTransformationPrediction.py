from os import listdir
import pandas as pd
from logger import App_Logger


class dataTransformation:

    def __init__(self):
        self.goodDataPath = "Predict_Validated_file/good/"
        self.log_file = "Logs/prediction_log.txt"
        self.log_writer = App_Logger()

    def replaceMissingWithNull(self):
        for file in [goodFile for goodFile in listdir( self.goodDataPath)]:
            csv_df = pd.read_csv(self.goodDataPath+file)
            csv_df.fillna("NULL", inplace=False)
            csv_df['Wafer'] = csv_df['Wafer'].str[6:]
            csv_df.to_csv(self.goodDataPath+file,index=None, header=True)
