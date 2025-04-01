from sklearn.metrics import mean_squared_error,r2_score
from .train import train_model
import matplotlib.pyplot as plt


def evalute_model():
    model,x_test,y_test=train_model()
    y_pred=model.predict(x_test)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    
    print(f"mean_squre_error{mse}")
    print(f"R^2 error {r2}")
    plt.scatter(x_test,y_test , color='blue')
    plt.plot(x_test , y_pred,color="red", linewidth=2)
    plt.xlabel('Hours Studied')
    plt.ylabel('Score')
    plt.title('Hours vs Score')
    plt.show()

if __name__ == "__main__":
    evalute_model()    