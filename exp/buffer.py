from multiprocessing import Queue

def create_queue(maxsize=100):
    return Queue(maxsize=maxsize)