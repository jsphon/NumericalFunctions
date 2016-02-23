#import time
from datetime import datetime

class Timer:    
    
    def __init__(self,title=None):
        self.title=title
        
    def __enter__(self):
        #if self.title:
        #    print( 'Beginning {0}'.format( self.title ) )
        #self.start = time.clock()
        self.start = datetime.utcnow()
        return self

    def __exit__(self, *args):
        #self.end = time.clock()
        #self.interval = self.end - self.start
        self.end = datetime.utcnow()
        self.interval = ( self.end - self.start ).total_seconds()
        if self.title:
            print( '{1} took {0:0.4f} seconds'.format( self.interval, self.title ) )
        else:
            pass#
            #print( 'Timer took {0:0.4f} seconds'.format( self.interval ) )
            
class AccumulatedTimer( object ):

    def __init__(self):
        self.interval=0.0
        self.call_count=0

    def __enter__(self):
        self.start = time.time()
        return self        

    def __exit__(self,a,b,c):
        self.end = time.time()
        self.interval+=self.end-self.start
        self.call_count+=1
    
    def __repr__(self):
        return '%0.4f seconds : %i calls'%(self.interval,self.call_count)