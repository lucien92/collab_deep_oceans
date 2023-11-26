import pandas as pd
from geoenrich.dataloader import *
from geoenrich.satellite import *
from geoenrich.enrichment import *
from geoenrich.exports import *




if __name__ == "__main__":
    df = pd.read_csv('plankton_data/planktons_med.csv')
    n = df.shape[0]
    for i in range((n//10000) + 1):
        print("Collating ", i*10000, " to ", (i+1)*10000)
        try:
            collate_npy(ds_ref = 'plankton_med', data_path = './npy', slice = (i*10000, (i+1)*10000))
        except Exception as e:
            print("Error with : ", e)
            pass



