import pyarrow as pa
import pandas as pd
import geopandas as gpd
import re
from datetime import date, datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import numpy as np

# Import both .ipc tables and the CropType shapefile table
work_dir = "L4A_MDB_270721/"

for file in os.listdir(work_dir):
    if "OPT_MAIN" in file and file.endswith("ipc"):
        opt_main_df = pa.ipc.open_file(work_dir+file).read_pandas()
    elif "OPT_RE" in file and file.endswith("ipc"):
        opt_re_df = pa.ipc.open_file(work_dir+file).read_pandas()
    elif "SAR_MAIN" in file and file.endswith("ipc"):
        sar_main_df = pa.ipc.open_file(work_dir+file).read_pandas()
    elif "SAR_RE" in file and file.endswith("ipc"):
        sar_re_df = pa.ipc.open_file(work_dir+file).read_pandas()
    elif "CropType" in file and file.endswith("shp"):
        table = gpd.read_file(work_dir+file)
        l4a_table = pd.DataFrame(table)
        l4a_table['CT_decl'] = l4a_table['CT_decl'].fillna('999')

print("Data imported")

# These functions parse the column names of the database and return sensor, metric and date
s2_regex = re.compile(r'(?P<year>\d{4})_(?P<month>\d\d)_(?P<day>\d\d)_s2_(?P<metric>[a-z0-9_]+)')
s1_regex = re.compile(r'(?P<year>\d{4})(?P<week>\d{2})_(?P<orbit>(\w{3}|\w{4}))_(?P<metric>(\w{2}|\w{5})_(\w{4}|\w{3}))_')


def parse_s2(inp):
    matches = s2_regex.search(inp)
    if matches:
        metric = matches.group('metric')
        day = matches.group('day')
        month = matches.group('month')
        year = matches.group('year')
        return {'metric':metric, 'datetime': date(year=int(year), month=int(month), day=int(day))}


def parse_s1(inp):
    matches = s1_regex.search(inp)
    if matches:
        metric = matches.group('metric')
        year = matches.group('year')
        week = matches.group('week')
        orbit = matches.group('orbit')
        return {'metric': metric, 'orbit': orbit, 'datetime' : date.fromisocalendar(year=int(year), week=int(week), day=7)}


# This function fills upsampled timeseries NaN values with the mean from the previous and next observation
# CAUTION: modifies input table
def fill_na_mean(inp):
    for col in inp.columns:
        for i in range(0, len(inp[col])-1):
            if pd.isnull(inp[col][i]):
                inp.at[i, col] = np.mean([inp[col][i-1], inp[col][i+1]])


# This function counts observations in the S1 dataset
def count_dates_s1(inp):
    date_list = []
    for c in inp.filter(regex='MEAN').columns:
        dat = parse_s1(c)['datetime']
        if dat not in date_list:
            date_list.append(dat)
    return date_list


# This section reformats mdb timeseries field by field to [date x metric] format
# This section omits NewIDs that are not processed with 20m resolution data
opt_val = opt_re_df['NewID'].values
sar_val = sar_re_df['NewID'].values
newids = list(set(opt_val).intersection(sar_val))
sar_dates = count_dates_s1(sar_main_df)


def process_data(newid):
    # This line checks if any of the newids are already processed
    if str(newid) not in result_dataframe['NewID'].values:
        try:
            # Create dataframes for each mdb + result dataframe
            s2_main = opt_main_df.loc[opt_main_df['NewID'] == newid].filter(regex='mean')
            s2_re = opt_re_df.loc[opt_main_df['NewID'] == newid].filter(regex='mean')
            sar_df = sar_main_df.loc[sar_main_df['NewID'] == newid].filter(regex='MEAN')
            r_df = pd.DataFrame()

            for c in s2_main.columns:
                props = parse_s2(c)
                metric = props['metric']
                p_date = props['datetime']
                if p_date not in r_df.index:
                    data = pd.DataFrame({metric: s2_main[c].values}, index=[p_date])
                    r_df = r_df.append(data)
                else:
                    r_df.at[p_date, metric] = s2_main[c].values

            for c in s2_re.columns:
                props = parse_s2(c)
                metric = props['metric']
                p_date = props['datetime']
                if p_date not in r_df.index:
                    data = pd.DataFrame({metric: s2_re[c].values}, index=[p_date])
                    r_df = r_df.append(data)
                else:
                    r_df.at[p_date, metric] = s2_re[c].values

            # Creates a DateTime field with type of Pandas datetime index - mandatory for resampling
            r_df['DateTime'] = pd.DatetimeIndex(r_df.index)

            # This gives you a DF with added upsampled rows, but the new rows are NaN
            # Define the desired upsample step in the resample() function
            # Field count should match SAR data
            # CAUTION: BEFORE RESAMPLING OPTICAL DATA TIMESERIES, EXAMINE BOTH OPTICAL AND SAR DATASETS
            # IN ORDER TO ADJUST THE resample() FUNCTION TO MATCH THE COUNT OF RECORDS IN BOTH DATASETS.
            # DUE TO DIFFERENCES IN DATA ACQUISITION, OPT AND SAR DATASET TIME STEPS DO NOT MATCH.
            r_df = r_df.set_index('DateTime').resample('8.5D').mean().reset_index()

            # PRIMARY APPROACH: This function takes the resampled timeseries and fills NaN with mean of previous and next value
            fill_na_mean(r_df)

            # This section does feature extraction and reformat for SAR data + adds the data to the s2 dataframe
            for c in sar_df.columns:
                props = parse_s1(c)
                metric = props['metric']
                p_date = props['datetime']
                idx = sar_dates.index(p_date)
                r_df.at[idx, metric] = sar_df[c].values

            # Some SAR acquisition dates lack COHE metrics, so we fill them with the previous available value
            r_df = r_df.fillna(method='pad')

            # Finally, adds newid field to each collection of rows that represent one field
            r_df['NewID'] = str(newid)
            r_df['Class_ID'] = int(table.loc[table['NewID'] == newid]['CT_decl'].values)
            return r_df
        except ValueError:
            pass
    else:
        pass


# having concat in the loop makes a copy of the dataframe every iteration, so it substantially slows down the processing
# calculating the results in a list comp and only then concatenating them together makes for a much faster workflow
results = [process_data(newid) for newid in tqdm(newids)]
result_dataframe = pd.concat(results)

# Define output dataset / can also be CSV with pd.to_csv()
result_dataframe.to_pickle('MDB270721_cached.pkl')