import pandas as pd
import yaml


with open("config/config.yaml" , "r")  as file:
    config=yaml.safe_load(file)
    print(config)

def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    ## preprocess steps
    return df



if __name__=="__main__":
    df=load_data(config['data']['raw_path'])
    print(df)
    preprocess_df=preprocess_data(df)
    preprocess_df.to_csv(config['data']['processed_path'],index=False)
    
    
    