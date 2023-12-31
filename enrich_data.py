import pandas as pd
from geoenrich.dataloader import *
from geoenrich.satellite import *
from geoenrich.enrichment import *
from geoenrich.exports import *


# taxons = [
#     "Dinophysis acuminata",
#     "Karenia mikimotoi",
#     "Chaetoceros",
#     "Dinophysis", 
#     "Alexandrium minutum",
#     "Pseudo-nitzschia"
# ]


if __name__ == "__main__":
    df = pd.read_csv('plankton_data/planktons_med.csv')

    # Import those occurrnces :
    geodf = import_occurrences_csv(
        path = 'plankton_data/planktons_med.csv', 
        id_col = 'index', 
        date_col = 'datetime', 
        lat_col = 'lat',
        lon_col = 'lon',
        )

    # Create the enrichment file
    create_enrichment_file(geodf, "plankton_med")

    variables = [
        'sst',
        'nh4_med',
        'no3_med',
        'po4_med',
        'o2_med',
        'chl_med',
        'thetao_med',
        'so_med',
        ]

    for variable in variables:

        print("")
        print("Enriching with ", variable)
        print("")
        var_id = variable
        geo_buff = 115
        time_buff = (0, 0)
        n = df.shape[0]

        # try:
        #     enrich(dataset_ref = 'plankton_med', var_id = var_id, geo_buff = geo_buff, time_buff = time_buff, slice = (0, 1000))
        # except Exception as e:
        #     print("Error with", variable, " : ", e)
        #     pass

        i = n//10000
        enrich(dataset_ref = 'plankton_med', var_id = var_id, geo_buff = geo_buff, time_buff = time_buff, slice = (i*10000, (i+1)*10000))

        # for i in range(n//10000):
        #     if variable != variables[0] or i >= 5: # Skip what we already downloaded
        #         print("Enriching ", i*10000, " to ", (i+1)*10000)
        #         try:
        #             enrich(dataset_ref = 'plankton_med', var_id = var_id, geo_buff = geo_buff, time_buff = time_buff, slice = (i*10000, (i+1)*10000))
        #         except Exception as e:
        #             print("Error with", variable, " : ", e)
        #             pass



