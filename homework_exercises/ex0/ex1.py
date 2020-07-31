# Name: Ofir Cohen
# ID: 312255847
# Date: 03/11/2019

class Clock(object):
    def __init__(self, hours=0, minutes=0, seconds=0):
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
    
    def tick(self):
        self.seconds += 1
        if self.seconds == 60:
            self.minutes += 1
            self.seconds = 0
        
        if self.minutes == 60:
            self.hours += 1
            self.minutes = 0
        
        if self.hours == 24:
            self.hours = 0
            self.minutes = 0
            self.seconds = 0
    
    def __str__(self):
        return "{}: {}: {}".format(self.hours, self.minutes, self.seconds)


class Calender(object):
    months_len = (31,28,31,30,31,30,31,31,30,31,30,31)
    
    def __init__(self, day=1, month=1, year=1900):
        self.day = day
        self.month = month
        self.year = year
    
    def advance(self):
        self.day += 1
        if self.day == self.months_len[self.month-1] + 1:
            self.month += 1
            self.day = 1
        
        if self.month == len(self.months_len) + 1:
            self.year += 1
            self.month = 1
            self.day = 1
        
    def __str__(self):
        return "{}/{}/{}".format(self.day, self.month, self.year)


class CalenderClock(Clock, Calender):
    def __init__(self, day, month, year, hours=0, minutes=0, seconds=0):
        Calender.__init__(self, day, month, year)
        Clock.__init__(self, hours, minutes, seconds)
        
    def __str__(self):
        return "{}, {}".format(Calender.__str__(self), Clock.__str__(self))

if __name__ == "__main__":
    
    x = CalenderClock(24,12,57)
    print(x)
    
    for i in range(1000):
        x.tick()
        
    for i in range(1000):
        x.advance()
        
    print(x)