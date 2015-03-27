import time

class Timer:    
    
    def __init__(self,title=None):
        self.title=title
        
    def __enter__(self):
        if self.title:
            print( 'Beginning {0}'.format( self.title ) )
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        if self.title:
            print( '{1} took {0:0.3f} seconds'.format( self.interval, self.title ) )
        else:
            print( 'Timer took {0:0.3f} seconds'.format( self.interval ) )