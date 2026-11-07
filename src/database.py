import pandas as pd
from pathlib import Path



def get_client_signature(id):
    data_path = Path(r"F:\03 - Projects\8_Signature_Verification_V2\database\clients.csv")
    df = pd.read_csv(data_path, index_col=False)
    df = df.dropna(how='all')
    df['NationalID'] = df['NationalID'].astype(int)
    
    if id in df['NationalID'].values:
        print(f'{id} Found in the database')
        return df[df['NationalID'] == id]['SignatureImagePath']
    else:
        print("Not Found")
        return None