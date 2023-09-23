from datetime import datetime


class App_Logger:
    def __init__(self):
        pass

    def log(self, file, message):
        self.current = datetime.now()
        self.date = self.current.date()
        self.time = self.current.time()
        print(f"{self.date} :: {self.time} :::: {message}")
        file.write(f"\n{self.date} :: {self.time} :::: {message}")


