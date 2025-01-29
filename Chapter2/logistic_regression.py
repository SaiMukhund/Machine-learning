import numpy as np

class logisticRegression():
    
    def __init__(self,learning_rate=0.0001,n_iters=100,random_state=24):
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
            p_pred=self.predict(X)
            error=y-p_pred
            self.w_+=self.learning_rate*(np.matmul(error,X))/m
            self.b_+=self.learning_rate*(error.mean())
        
    def net_output(self,X):
        z=np.dot(X,self.w_)+self.b_
        p=1/(1+np.exp(-z))
        return p
    
    def predict(self,X):
        p=self.net_output(X)
        return np.where(p>=0.5,1,0)
    

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

    model=logisticRegression(n_iters=100)
    model.fit(x,y)
    print(model.b_,model.w_,model.predict(6))
