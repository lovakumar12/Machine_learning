import pandas as pd
from sklearn.model_selection import train_test_split
from .model import create_model
import yaml


with open("config/config.yaml" , "r") as file:
    config=yaml.safe_load(file)
    
    
def train_model():
    df=pd.read_csv(config['data']['preprocesed_path'])
    X=df["Hours"]
    y=df['scores']
    x_train,x_test,y_train,y_test= train_test_split(X,y ,train_size=config['model']['test_size'], random_state=config["model"]['random_state'])
    model=create_model()
    model.fit(x_train,y_train)
    return model ,x_test,y_test

if __name__=="__main__":
    train_model()
    
