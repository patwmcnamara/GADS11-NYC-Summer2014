#Creating a class
class Car():
    def __init__(self, model='Ford'):
    #intitiating the features of the car class. The values given are the default values
        self.model = model
        self.running = False
    def start(self):
    #start is a function of the car and below are the steps to be performed when the start function is called
        if self.running != True:
            print 'The car started!'
            self.running = True
        else:
            print 'The car is already running!'
    def stop(self):    
    #start is a function of the car and below are the steps to be performed when the start function is called    
        if self.running == True:
            print 'The car stopped!'
            self.running = False
        else:
            print 'The car was not running!'