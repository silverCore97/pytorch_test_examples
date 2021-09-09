import time


class Timer():
    def __init__(self):
        self.__start=time.time()
        self.__duration = 0

    def start(self):
        self.__start=time.time()

    def stop(self):
        self.__duration = time.time() - self.__start

    def duration(self):
        return self.__duration
