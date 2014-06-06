class Car():
    def __init__(self, model='Ford', wheel=4):
        self.model = model
        self.running = False
        self.wheel= wheel
	if self.wheel < 0:
	    self.wheel = 0
	else:
	    print "You can't have negative wheels on a car"
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

ford = Car(wheel=-7)
print "Ford wheel: " + str(ford.wheel)
nissan = Car(model = 'Nissan')
ford.running
ford.start()
ford.running
nissan.running
nissan.stop()
