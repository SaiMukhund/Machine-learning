import numpy as np

class perceptron():

    def __init__(self,learning_rate=0.001,n_iters=100,random_state=24):
        self.learning_rate=learning_rate
        self.n_iters=n_iters
        self.random_state=random_state

    def fit(self,X,y):
        """
        X:np.arrays 2d dimensional 
        y:np.arrays 1d dimensional 
        """
        np.random.seed(self.random_state)
        n=X.shape[1]
        m=X.shape[0]

        self.w_=np.random.rand(n)
        self.b_=np.random.rand()

        for _ in range(self.n_iters):
            for xi,yi in zip(X,y):
                y_pred=self.predict(xi)
                error=yi-y_pred
                self.w_+=(error*xi)*self.learning_rate
                self.b_+=error*self.learning_rate

    
    def predict(self,X):
        z=np.dot(X,self.w_)+self.b_
        # if z>=0:
        #     return 1
        # else:
        #     return 0
        return np.where(z>=0,1,0)
    
if __name__=="__main__":

    x=np.array([
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [-1],
        [-2],[-3],[-4],[-5],[-6]
    ])
    y=np.array([1,1,1,1,1,1,1,0,0,0,0,0,0])

    model=perceptron()
    model.fit(x,y)
    print(model.b_,model.w_)
        
