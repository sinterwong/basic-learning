from multiprocessing import Process, Value
import signal
import time


def worker():
    time.sleep(20)


if __name__ == '__main__':
    p = Process(target=worker)
    p.start()
    time.sleep(1)
    print(p.pid)
    p.terminate()
