import datetime

class DateIter () : 
    def __init__ (self, start, end) :
        self.start = start
        self.end = end
    
    def __iter__ (self) : 
        return self

    def __next__ (self) : 
        curr = self.start
        if self.start.date != self.end.date :
            self.start = self.start + 1
            return curr
        else : 
            raise StopIteration()

    def __len__ (self) : 
        return self.end - self.start

    def toList(self):
        return [x for x in self]

class Date () : 

    MONTHS = ['Jan', 'Feb', 'Mar', 
            'Apr', 'May', 'Jun', 
            'Jul', 'Aug', 'Sep', 
            'Oct', 'Nov', 'Dec']
    
    def __init__ (self, date) : 
        self.date = date
        d, m  = date.split(' ')
        d = int(d)
        self.day = d
        self.month = self.MONTHS.index(m) + 1

    def __add__ (self, n) : 
        td = datetime.timedelta(days=n)
        newDate = datetime.date(2020, self.month, self.day) + td
        month = self.MONTHS[newDate.month - 1]
        day = newDate.day
        return Date(f'{day} {month}')
    
    def __sub__ (self, that) :
        if isinstance(that, Date) : 
            d1 = datetime.date(2020, self.month, self.day)
            d2 = datetime.date(2020, that.month, that.day)
            return (d1 - d2).days
        else : 
            d1 = datetime.date(2020, self.month, self.day)
            td = datetime.timedelta(days=-that)
            d2 = d1 + td
            month = self.MONTHS[d2.month - 1]
            day = d2.day
            return Date(f'{day} {month}')

    def __lt__ (self, that) : 
        return (self.month, self.day) < (that.month, that.day)
        
    def __gt__ (self, that) : 
        return (self.month, self.day) > (that.month, that.day)
    
    def __le__ (self, that) : 
        return (self.month, self.day) <= (that.month, that.day)
        
    def __ge__ (self, that) : 
        return (self.month, self.day) >= (that.month, that.day)

