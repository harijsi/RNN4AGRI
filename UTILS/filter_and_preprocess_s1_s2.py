import pandas as pd
from tqdm import tqdm

# Define list of classes
class_list = [1, 2, 3, 11, 12, 15, 16]
processed_dir = 'processed_datasets/'
filename = f'Filtered_S1_S2_formatted_{str(class_list)}.csv'
subset_size = 500

# Define input csv
df = pd.read_csv('S1ft_S2full_formatted_Class_ID.csv')
df2 = df.loc[df['Class_ID'].isin(class_list)]

counter_dict = {f"{l}":0 for l in class_list}


def filter_dataframe(dataframe):
    target_df = pd.DataFrame()
    # Filter 500 samples from each class
    for i in tqdm(range(0, len(dataframe.index))):
        row = dataframe.iloc[[i]]
        class_id = row.iloc[0]['Class_ID']

        for k, v in counter_dict.items():
            if v < subset_size:
                if class_id == int(k):
                    target_df = target_df.append(row)
                    counter_dict[k] += 1

    target_df.to_csv(f'Filtered_S1_S2_formatted_{str(class_list)}.csv', index=False)


def process_dataframe(file_name):
    dataframe = pd.read_csv(file_name)
    dataframe = dataframe.fillna(value=0, axis=0)
    dataframe['Class_ID'] = dataframe['Class_ID'].astype(int)

    class2idx = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            8: 7,
            9: 8,
            11: 9,
            12: 10,
            13: 11,
            14: 12,
            15: 13,
            16: 14,
            17: 15,
            18: 16,
            19: 17,
            20: 18
    }

    # full list of class2idx
    # class2idx = {
    #     1: 0,
    #     2: 1,
    #     3: 2,
    #     4: 3,
    #     5: 4,
    #     6: 5,
    #     7: 6,
    #     8: 7,
    #     9: 8,
    #     11: 9,
    #     12: 10,
    #     13: 11,
    #     14: 12,
    #     15: 13,
    #     16: 14,
    #     17: 15,
    #     18: 16,
    #     19: 17,
    #     20: 18
    # }

    idx2class = {v: k for k, v in class2idx.items()}
    dataframe['Class_ID'].replace(class2idx, inplace=True)
    result_df = pd.DataFrame()

    fields = ['ASC_VV_BCK_MEAN', 'ASC_VH_BCK_MEAN', 'ASC_RATIO_BCK_MEAN',
              'DESC_VV_BCK_MEAN', 'DESC_VH_BCK_MEAN', 'DESC_RATIO_BCK_MEAN',
              'VH_COHE_MEAN_t', 'VV_COHE_MEAN_t', 'dev_b3', 'dev_b4', 'dev_b5',
              'dev_b6', 'dev_b7', 'dev_b8', 'dev_b8a', 'dev_b11', 'dev_ndvi',
              'dev_ndwi', 'dev_brightness', 'mean_b3', 'mean_b4', 'mean_b5',
              'mean_b6', 'mean_b7', 'mean_b8', 'mean_b8a', 'mean_b11', 'mean_ndvi',
              'mean_ndwi', 'mean_brightness']

    for i in tqdm(range(0, len(dataframe.index))):
        df_row = dataframe.iloc[[i]].T

        # Field list to be populated with stats

        df_row[fields] = None
        date_list = []

        # Populate date list
        for row in df_row.iterrows():
            if row[0][0:2] == 'XX':
                date = row[0][4:10]
                if date not in df_row.index.values:
                    df_row = df_row.append(pd.Series(name=date, dtype=float))
                    res = df_row.append(pd.Series(name=date, dtype=float))
                    date_list.append(date)

        # Create and populate columns
        for row in df_row.iterrows():

            # Exclude Field ID and Class ID
            if row[0][0:2] == 'XX':

                # Look for data only from MEAN columns - can be used to filter parameters
                if 'MEAN' in row[0] or 'mean' in row[0] or 'dev_' in row[0]:
                    date = row[0][4:10]
                    for b in fields:
                        if b in row[0]:
                            df_row.at[date, b] = float(row[1][i])

        rowData = df_row.loc[date_list, :]
        rowData = rowData.drop(i, axis=1)

        Class_ID = int(df_row.iloc[1][i])
        NewID = int(df_row.iloc[0][i])

        rowData['Field_ID'] = NewID
        rowData['Class_ID'] = Class_ID
        result_df = result_df.append(rowData)

    result_df.to_csv(processed_dir+f'Processed_S1_S2_overfit_{str(class_list)}].csv')


filter_dataframe(df2)
process_dataframe(filename)