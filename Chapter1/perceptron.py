import numpy as np 

class Perceptron():

    def __init__(self,learning_rate=0.01,n_iters=100,random_state=24):
        self.learning_rate=learning_rate
        self.n_iters=n_iters
        self.random_state=random_state

    def fit(self,X,y):
        np.random.seed(self.random_state)
        n_features=X.shape[1]
        m=X.shape[0]
        self.w_=np.random.rand(n_features)
        self.b_=0.0

        for _ in range(self.n_iters):
            for xi,yi in zip(X,y):
                y_pred=self.predict(xi)
                error=(yi-y_pred)
                self.w_+=self.learning_rate*(error*xi)
                self.b_+=self.learning_rate*(error)

    def predict(self,X):
        z=np.dot(X,self.w_)+self.b_
        return np.where(z>=0,1,0)
    
if __name__=="__main__":
    x=np.array([[0],[1],[2],[3],[4],[5],[6],[-1],[-2],[-3],[-4],[-5],[-6]])
    y=np.array([1,1,1,1,1,1,1,0,0,0,0,0])
    model=Perceptron()
    model.fit(x,y)
    print(model.b_,model.w_)