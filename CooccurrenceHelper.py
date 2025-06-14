import os
import pandas as pd
from datetime import datetime
import numpy as np
import torch
import time
from sklearn.metrics import pairwise_distances
import csv

class CooccurrenceHelper():
    def __init__ (self,
                  spatial_conf,
                  temporal_conf,
                  target_species=[],
                  distance_function=lambda x1, x2, y1, y2: ((x1-x2)**2 + (y1-y2)**2)**0.5,
                  output_dir='./workspace/species_data'):
        
        super().__init__()

        self.spatial_conf = spatial_conf
        self.temporal_conf = temporal_conf
        
        self.target_species = target_species
        self.x_start = spatial_conf.x_start
        self.y_start = spatial_conf.y_start
        self.x_end = spatial_conf.x_end
        self.y_end = spatial_conf.y_end
        
        # for 2 dimension
        self.cooccurrence_xy_limit = 2 * spatial_conf.out_res
        
        # for 1 dimension
        self.cooccurrence_day_limit = temporal_conf.cooccurrence_day_limit        
        
        # convert to datetime format
        self.date_start = datetime.strptime(temporal_conf.date_start, '%Y-%m-%d')
        self.date_end = datetime.strptime(temporal_conf.date_end, '%Y-%m-%d')
        
        # convert date to 'days after day_first' (int)
        self.day_first = 0
        self.day_last = (self.date_end - self.date_start).days
        
        # set output directory and create folders
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.occurrence_dir = os.path.join(self.output_dir, 'occurrence_data')
        if not os.path.exists(self.occurrence_dir):
            os.makedirs(self.occurrence_dir)

        self.cooccurrence_dir = os.path.join(self.output_dir, 'cooccurrence_data')
        if not os.path.exists(self.cooccurrence_dir):
            os.makedirs(self.cooccurrence_dir)

        self.get_distance = distance_function
        
        # whether to embedding with virtual species
        
    def filter_occurrence(self, gbif_occurrence_csv, nrows=None):

        self.gbif_occurrence_csv = gbif_occurrence_csv
        
        # read raw data directly download from gbif
        self.species_raw = pd.read_csv(self.gbif_occurrence_csv, 
                                       sep = '\t', 
                                       usecols = [
                                           'species',
                                           'decimalLatitude',
                                           'decimalLongitude',
                                           'day',
                                           'month',
                                           'year', 
                                           'coordinateUncertaintyInMeters', 
                                           'coordinatePrecision'],
                                       nrows=nrows,
                                       quoting=csv.QUOTE_NONE
                                      )
        
        # drop na values
        species_filter = self.species_raw.dropna(subset = ['species', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year'], how = 'any').reset_index(drop = True)

        # filter species 
        # change species name
        species_filter['species'] = species_filter.apply(lambda x: f"{x['species'].split(' ')[0]}_{x['species'].split(' ')[1]}", axis = 1)
        if len(self.target_species) != 0:
            species_filter = species_filter[species_filter['species'].isin(self.target_species)].reset_index(drop = True)

        # add column 'date' with values of 'year-month-day'
        species_filter['date'] = species_filter['year'].astype(int).astype(str) + "-" + species_filter['month'].astype(int).astype(str) + "-" + species_filter['day'].astype(int).astype(str)
        species_filter['day'] = species_filter['day'].astype(int)

        # add column 'daysincebegin' with values of days after the begin of the first date
        # number of days since date_start
        species_filter['daysincebegin'] = species_filter.apply(lambda x: (datetime.strptime(x.date, '%Y-%m-%d') - self.date_start).days, axis = 1)

        # filter records by date
        species_filter = species_filter[(species_filter['daysincebegin'].values >= self.day_first) & (species_filter['daysincebegin'].values <= self.day_last)].reset_index(drop = True)
        
        species_filter = species_filter.query('(decimalLatitude >= @self.y_start) & (decimalLatitude < @self.y_end) & (decimalLongitude >= @self.x_start) & (decimalLongitude < @self.x_end)').reset_index(drop=True)

        # filter records by coordinate uncertainty (1000m)
        species_filter = species_filter[~(species_filter.coordinateUncertaintyInMeters > 1000)].reset_index(drop=True)

        # filter records by coordinate precision (0.008333333)
        species_filter = species_filter[~(species_filter.coordinatePrecision > 0.008333333)].reset_index(drop=True)

        # remove duplicate records by date and time
        cols_key = ["species", "decimalLatitude", "decimalLongitude", "year", "month", "day"]
        species_filter = species_filter.drop_duplicates(subset=cols_key, keep="first").reset_index(drop=True)

        # save filtered csv
        species_filter.to_csv(os.path.join(self.occurrence_dir, 'species_occurrence_filter.csv'), index = None)
        print(f"File: {os.path.join(self.occurrence_dir, 'species_occurrence_filter.csv')} saved.")
        
        self.species_filter = species_filter


    def aggregate_cooccurrence_units(self, sp_filter_from=None, 
                                     cooccurrence_day_mul=1, cooccurrence_xy_mul=1):
        start_time = time.time()
        def res_divider(a, b):
            digit_parts = str(b).split('.')
            assert(len(digit_parts) == 2)
            num_digits_after_decimal = min(len(digit_parts[1]), 6)
            return round(a / b, num_digits_after_decimal)

        if sp_filter_from is not None:
            self.species_filter = pd.read_csv(sp_filter_from)

        sp_filter = self.species_filter.copy()            
        
        self.data_unit = dict()
        
        self.cooccurrence_day_mul = cooccurrence_day_mul
        self.cooccurrence_xy_mul = cooccurrence_xy_mul
        
        sp_filter['daysincebeginUnit'] = ((sp_filter.daysincebegin - sp_filter.daysincebegin.min()) // (self.cooccurrence_day_limit * self.cooccurrence_day_mul)).astype(int)
        sp_filter['decimalLongitudeUnit'] = res_divider(sp_filter.decimalLongitude - self.x_start, self.spatial_conf.out_res * self.cooccurrence_xy_mul).astype(int)
        sp_filter['decimalLatitudeUnit'] = res_divider(sp_filter.decimalLatitude - self.y_start, self.spatial_conf.out_res * self.cooccurrence_xy_mul).astype(int)

        def coocurr_agg(df, data_unit):
            t, x, y = df.name
            if t not in data_unit:
                data_unit[t] = dict()
            if round(x, 2) not in data_unit[t]:
                data_unit[t][x] = dict()
            self.data_unit[t][x][y] = df
#             print("Aggregating...", t,x,y, end='\r')

        # order: t, x, y
        sp_filter.groupby(['daysincebeginUnit', 'decimalLongitudeUnit', 'decimalLatitudeUnit']).apply(coocurr_agg, self.data_unit)
        print(f'Aggregating data costs {time.time() - start_time} seconds.')
        
    def count_cooccurrence_mod(self, cooccurrence_counts_file='cooccurrence.csv'):
        start_time = time.time()
        data_unit = self.data_unit
        primary_indices = {}
        cooccur_counts_df = None
        for t in data_unit:
            len_t = max(data_unit)
            for x in data_unit[t]:
                len_x = max(data_unit[t])
                for y in data_unit[t][x]:
                    len_y = max(data_unit[t][x])
                    print(f'Counting... {t}/{len_t}, {x}/{len_x}, {y}/{len_y}', end='\r')

                    obsrvs1 = data_unit[t][x][y]

                    neighbor_units = np.array(np.meshgrid([0, 1], [0, 1], [0, 1]), dtype=int).T.reshape(-1, 3)

                    for dt, dx, dy in neighbor_units:
                        neighbor_t = t + dt
                        if neighbor_t not in data_unit:
                            continue

                        neighbor_x = x + dx
                        if neighbor_x not in data_unit[neighbor_t]:
                            continue

                        neighbor_y = y + dy
                        if neighbor_y not in data_unit[neighbor_t][neighbor_x]:
                            continue

                        obsrvs2 = data_unit[neighbor_t][neighbor_x][neighbor_y]

                        day_unit_str = f'd{t}-{neighbor_t}'
                        spatial_unit_x = f'x{x}-{neighbor_x}' #str(cooccur_satisfied_vals[i][4]*1000 + cooccur_satisfied_vals[i][5])
                        spatial_unit_y = f'y{y}-{neighbor_y}' #str(cooccur_satisfied_vals[i][1]*1000 + cooccur_satisfied_vals[i][2])
                        data_unit_str = f'{day_unit_str}_{spatial_unit_x}_{spatial_unit_y}'

                        if data_unit_str in primary_indices:
                            continue

                        primary_indices[data_unit_str] = True

                        pwd_spatial_satisfied = pairwise_distances(obsrvs1[['decimalLatitude', 'decimalLongitude']], obsrvs2[['decimalLatitude', 'decimalLongitude']]) < self.cooccurrence_xy_mul * self.cooccurrence_xy_limit
                        pwd_temporal_satisfied = pairwise_distances(obsrvs1[['daysincebegin']], obsrvs2[['daysincebegin']]) < self.cooccurrence_day_mul * self.cooccurrence_day_limit
                        pwd_satisfied = pwd_spatial_satisfied & pwd_temporal_satisfied
                        satisfied_idx = np.where(pwd_satisfied)
                        cooccur_satisfied_vals = np.stack([
                            obsrvs1.iloc[satisfied_idx[0]].species.values, 
                            obsrvs2.iloc[satisfied_idx[1]].species.values
                        ], axis=1)
                        cooccur_satisfied_vals = cooccur_satisfied_vals[cooccur_satisfied_vals[:, 0]!=cooccur_satisfied_vals[:, 1]]
                        if cooccur_satisfied_vals.shape[0] > 0:
                            cooccur_satisfied_vals.sort(axis=1)
                            cooccur_local_unique = pd.DataFrame(cooccur_satisfied_vals, columns=['sp1', 'sp2'])
                            cooccur_local_unique_counts = cooccur_local_unique.groupby(['sp1', 'sp2']).size().to_frame('counts').reset_index()
                            if cooccur_counts_df is None:
                                cooccur_counts_df = cooccur_local_unique_counts
                            else:
                                cooccur_counts_df = pd.concat([cooccur_counts_df, cooccur_local_unique_counts])

            if cooccur_counts_df is not None:
                cooccur_counts_df = cooccur_counts_df.groupby(['sp1', 'sp2']).sum().reset_index()
                
        # save the cooccurrence csv
        sp_list = self.species_filter.species.unique()
        sp_list.sort()
        sp_combs_df = pd.DataFrame(np.array(np.meshgrid(sp_list, sp_list)).T.reshape(-1, 2), columns=['sp1', 'sp2'])
        sp_combs_df['counts'] = 0
        
        cooccur_counts_df = pd.concat([cooccur_counts_df, sp_combs_df])
        cooccur_counts_df = cooccur_counts_df.groupby(['sp1', 'sp2']).head(1)
        
        cooccur_counts_df.to_csv(os.path.join(self.cooccurrence_dir, cooccurrence_counts_file), sep='\t', index=False)
        print(f'File: {os.path.join(self.cooccurrence_dir, cooccurrence_counts_file)} saved.')
        print(f'Counting cooccurrence costs {time.time() - start_time} seconds.')

