__author__ = 'reshamashaikh'

# ------------------------------------------
# General Assembly:  Data Science
# Python v:          2.7.6
# Lab:               2 (Thurs, 6/5/14)
# Topic:             Classes
# ------------------------------------------


# value of classes:  reusability, encapsulation (easy to pass on - inheritance)

print "-"*50

# Patrick's comment
# You notify the user when they set wheels less than 0, 
# but don't make sure the wheels are actually non-negative

class Car():
    def __init__(self, model='Ford',wheels=4):
        self.model = model
        self.running = False
        self.wheels = wheels
        if self.wheels < 0:
            self.wheels = 0
            print "It is not possible to have less than 0 wheels!"
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


chevy = Car(model = 'Chevy')
chevy.running
chevy.start()
chevy.stop()



print "-"*50

chevy = Car(model = 'Chevy', wheels=2)
chevy.running
chevy.start()
chevy.stop()


print "-"*50

chevy = Car(model = 'Chevy', wheels=-1)
chevy.running
chevy.start()
chevy.stop()



exit()
