from multiprocessing import Queue

#this buffer is created for trainer to extract roll out information 
def create_queue(maxsize=100):
    return Queue(maxsize=maxsize)