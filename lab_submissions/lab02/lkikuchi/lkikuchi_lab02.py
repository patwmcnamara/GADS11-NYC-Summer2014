class Car():
    def __init__(self, model='Ford', wheels=4):
        self.model = model
        # set the number of wheels
        self.wheels = wheels
        self.running = False
    def start(self):
        if self.running != True:
            print 'The car started!'
            self.running = True
        else:
            print 'The car is already running!'
    def stop(self):
        if self.running == True:
            print 'The car stopped!'
            self.running = False
        else:
            print 'The car was not running!'
    def set_wheels(self,num=4):
        if num < 0:
            print 'The car can\'t have negative wheels!'
        else:
            self.wheels = num
            print "The car now has %d wheels!" % num
