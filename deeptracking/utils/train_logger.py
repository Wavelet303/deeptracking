from logging import getLoggerClass, addLevelName, setLoggerClass, NOTSET

# level just above info
SLACK = 21
DATA = 22


class TrainLogger(getLoggerClass()):
    def __init__(self, name, level=NOTSET):
        super().__init__(name, level)

        addLevelName(SLACK, "SLACK")
        addLevelName(DATA, "DATA")

    def slack(self, msg, *args, **kwargs):
        if self.isEnabledFor(SLACK):
            print("send to slack!!")
            self._log(SLACK, msg, args, **kwargs)

    def data(self, msg, *args, **kwargs):
        if self.isEnabledFor(DATA):
            print("Data : {}, {}".format(msg, *args))
            self._log(DATA, msg, None, None)