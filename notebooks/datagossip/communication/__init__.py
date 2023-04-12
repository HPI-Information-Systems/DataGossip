class Communicator:
    def __setstate__(self, state):
        print("setstate", state)
        