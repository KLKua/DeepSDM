# ==================================================================================
# This script includes a PlotUtlis class and a Beta regression implementation.
# ==================================================================================

import os, traceback
import yaml
import rasterio
import json
import pandas as pd
from types import SimpleNamespace
import numpy as np
import cv2
import matplotlib as mpl
from joblib import Parallel, delayed
import h5py
import feather
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import mahalanobis
from scipy.stats import spearmanr, ttest_rel, kruskal
import glob
from rasterio.transform import xy
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
import itertools
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

class PlotUtlis():
    """
    The PlotUtlis class provides utility functions for:
     - Loading model configurations
     - Handling raster data
     - Loading and processing species occurrence data
     - Performing dimensionality reduction on environmental data
     - Generating and storing results (e.g., performance indicators, attention maps)
     - Calculating niche space results and related metrics
     - Handling cluster-based operations on niche spaces
    """

    def __init__(self, run_id, exp_id):
        # Set the size for the niche raster
        self.niche_rst_size = 50

        # Define various paths for configurations and predictions
        self.conf_path = os.path.join('mlruns', exp_id, run_id, 'artifacts', 'conf')
        self.predicts_path = os.path.join('predicts', run_id)
        self.predicts_maxent_path = os.path.join('predicts_maxent', run_id)
        self.deepsdm_h5_path = os.path.join(self.predicts_path, 'h5', '[SPECIES]', '[SPECIES].h5')
        self.maxent_h5_path = os.path.join(self.predicts_maxent_path, 'h5', 'all', '[SPECIES]', '[SPECIES].h5')
        self.attention_h5_path = os.path.join(self.predicts_path, 'attention', '[SPECIES]', '[SPECIES]_[DATE]_attention.h5')
        self.traitdataset_taxon_path = os.path.join('raw', 'dwca-trait_454-v1.68', 'taxon.txt')
        self.traitdataset_mesurement_path = os.path.join('raw', 'dwca-trait_454-v1.68', 'measurementorfacts.txt')
        self.performance_indicator_multithreshold = os.path.join(self.predicts_maxent_path, 'model_performance_diffthreshold*.csv')
        self.performance_indicator_singlethreshold = os.path.join(self.predicts_maxent_path, 'model_performance_constantthreshold*.csv')

        # Load DeepSDM configurations
        self.DeepSDM_conf_path = os.path.join(self.conf_path, 'DeepSDM_conf.yaml')
        with open(self.DeepSDM_conf_path, 'r') as f:
            self.DeepSDM_conf = SimpleNamespace(**yaml.load(f, Loader=yaml.FullLoader))

        # Load the extent binary map (area of interest in a raster)
        with rasterio.open(os.path.join(self.conf_path, 'extent_binary.tif'), 'r') as f:
            self.transform = f.transform
            self.height, self.width = f.shape
            self.lon_min, self.lat_max = rasterio.transform.xy(self.transform, 0, 0)
            self.lon_max, self.lat_min = rasterio.transform.xy(self.transform, self.height - 1, self.width - 1)
            self.extent_binary = f.read(1)
        self.extent_binary_extent = [self.lon_min, self.lon_max, self.lat_min, self.lat_max]

        # Load species occurrence points
        with open(os.path.join(self.conf_path, 'species_information.json'), 'r') as f:
            self.sp_info = json.load(f)
        with open(os.path.join(self.conf_path, 'cooccurrence_vector.json')) as f:
            self.coocc_vector = json.load(f)
        with open(os.path.join(self.conf_path, 'env_information.json')) as f:
            self.env_info = json.load(f)

        # Define lists of species and dates
        self.species_list_train = sorted(self.DeepSDM_conf.training_conf['species_list_train'])
        self.species_list_predict = sorted(self.DeepSDM_conf.training_conf['species_list_predict'])
        self.species_list_all = sorted(list(self.coocc_vector.keys()))
        self.date_list_predict = self.DeepSDM_conf.training_conf['date_list_predict']
        self.date_list_train = self.DeepSDM_conf.training_conf['date_list_train']

        # Load species occurrence DataFrame
        self.species_occ = pd.read_csv(self.DeepSDM_conf.cooccurrence_conf['sp_filter_from'])

        # Load elevation data for reference
        elev_path = os.path.join(self.env_info['dir_base'],
                                 self.env_info['info']['elev'][self.date_list_train[0]]['tif_span_avg'])
        with rasterio.open(elev_path, 'r') as f:
            self.elev_rst = f.read(1)

        # Load co-occurrence counts
        self.coocc_counts = pd.read_csv('./workspace/species_data/cooccurrence_data/cooccurrence.csv', sep='\t')

        # Define environment variables and handle naming changes for clarity
        self.env_list = self.DeepSDM_conf.training_conf['env_list']
        env_list_change = {
            'clt': 'Cloud area fraction',
            'hurs': 'Relative humidity',
            'pr': 'Precipitation',
            'rsds': 'Shortwave radiation',
            'sfcWind': 'Wind speed',
            'tas': 'Temperature',
            'EVI': 'EVI',
            'landcover_PC01': 'LandcoverPC1',
            'landcover_PC02': 'LandcoverPC2',
            'landcover_PC03': 'LandcoverPC3',
            'landcover_PC04': 'LandcoverPC4',
        }
        self.env_list_detail = [env_list_change[i] for i in self.env_list]

        # Define indices for plotting principal components
        self.x_pca = 1
        self.y_pca = 2

        # Define default color list
        self.color_list = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]
        
        # Define subfolder paths for plots
        self.plot_path = os.path.join('plots', run_id)
        self.plot_path_embedding_dimension_reduction = os.path.join(self.plot_path, 'Fig2_embedding_dimension_reduction')
        self.plot_path_embedding_correlation = os.path.join(self.plot_path, 'Fig2_embedding_correlation')
        self.plot_path_attention = os.path.join(self.plot_path, 'Fig3_attention')
        self.plot_path_nichespace = os.path.join(self.plot_path, 'Fig4_nichespace')
        self.plot_path_nichespace_clustering = os.path.join(self.plot_path, 'Fig5_nichespace_clustering')
        self.plot_path_cph = os.path.join(self.plot_path, 'Fig6_cph')
        self.plot_path_cph_subplots = os.path.join(self.plot_path_cph, 'subplots')
        self.plot_path_suppl = os.path.join(self.plot_path, 'FigSuppl')

        # Define output file paths
        self.avg_elev_path = os.path.join(self.plot_path_embedding_dimension_reduction, 'avg_elevation.csv')
        self.df_attention_path = os.path.join(self.plot_path_attention, 'df_attention.csv')
        self.env_pca_loadings_path = os.path.join(self.plot_path_nichespace, 'env_pca_loadings.csv')
        self.df_env_corr_path = os.path.join(self.plot_path_nichespace, 'env_correlation.csv')
        self.df_env_pca_path = os.path.join(self.plot_path_nichespace, 'df_env_pca.feather')
        self.pc_info_path = os.path.join(self.plot_path_nichespace, 'pc_info.yaml')
        self.bin_info_path = os.path.join(self.plot_path_nichespace, 'bin_info.yaml')
        self.extent_info_path = os.path.join(self.plot_path_nichespace, 'extent_info.yaml')
        self.plot_path_df_species = os.path.join(self.plot_path_nichespace, 'df_species', '[SPECIES].feather')
        self.plot_path_nichespace_h5 = os.path.join(self.plot_path_nichespace, 'h5', '[SPECIES].h5')
        self.plot_path_nichespace_png_sp = os.path.join(self.plot_path_nichespace, 'png', '[SPECIES]', '[SPECIES]_nichespace_[SUFFIX].png')
        self.df_grid_path = os.path.join(self.plot_path_nichespace, 'df_grid.feather')
        self.df_spearman_path = os.path.join(self.plot_path_nichespace, 'df_spearman.csv')
        self.cluster_labels_path = os.path.join(self.plot_path_nichespace_clustering, 'cluster_labels.yaml')
        self.cluster_avg_nichespace_path = os.path.join(self.plot_path_nichespace_clustering, 'cluster_avg_nichespace.yaml')
        self.df_nichespace_center_coordinate_path = os.path.join(self.plot_path_nichespace_clustering, 'nichespecies_center_coordinate.csv')
        self.df_spearman_ecogeo_path = os.path.join(self.plot_path_cph, 'df_spearman_ecogeo.csv')
        self.niche_beta_params_path = os.path.join(self.plot_path_cph, 'niche_beta_params.json')
        self.geographical_beta_params_path = os.path.join(self.plot_path_cph, 'geographical_beta_params.json')
        self.beta_cluster_result_path = os.path.join(self.plot_path_cph, '[CENTER_TYPE]_cluster_[CLUSTER]_beta_result.txt')
        self.niche_beta_cluster_params_path = os.path.join(self.plot_path_cph, 'niche_beta_cluster_params.json')
        self.geographical_beta_cluster_params_path = os.path.join(self.plot_path_cph, 'geographical_beta_cluster_params.json')
        self.label_to_color_path = os.path.join(self.plot_path_nichespace_clustering, 'label_to_color.json')
        self.species_cluster_env_values_stats_path = os.path.join(self.plot_path_suppl, "cluster_env_values_stats.json")
        
        # Load any existing files if they are present
        self.load_existing_files()

    # For Fig2
    def calculate_avg_elev(self):
        """
        Calculate average elevation for each species by querying
        the raster at the occurrence points.
        """
        species_avg_elevation = {}

        for i, sp in enumerate(self.species_list_all):
            # Filter occurrence records for current species
            occ_sp = self.species_occ[self.species_occ.species == sp]
            rows, cols = rasterio.transform.rowcol(
                self.transform,
                occ_sp['decimalLongitude'].values,
                occ_sp['decimalLatitude'].values
            )
            elevations = self.elev_rst[rows, cols]

            valid_elevations = np.where(elevations == -9999, np.nan, elevations)
            if np.any(~np.isnan(valid_elevations)):
                species_avg_elevation[sp] = np.nanmean(valid_elevations)
            else:
                species_avg_elevation[sp] = None

        avg_elevation_df = pd.DataFrame.from_dict(
            species_avg_elevation, orient='index', columns=['AverageElevation']
        )
        avg_elevation_df.reset_index(inplace=True)
        avg_elevation_df.rename(columns={'index': 'Species'}, inplace=True)

        return avg_elevation_df

    # For Fig2
    def set_cooccurrence_counts(self, species_list):
        """
        Filter co-occurrence data based on:
         - removing sp1 == sp2
         - removing zero 'counts'
         - only keeping species in the provided list
        """
        coocc_counts = self.coocc_counts.copy()
        coocc_counts = coocc_counts[(coocc_counts.sp1 != coocc_counts.sp2)].reset_index(drop=True)
        coocc_counts = coocc_counts[coocc_counts.counts != 0].reset_index(drop=True)
        coocc_counts = coocc_counts[(coocc_counts.sp1.isin(species_list)) & (coocc_counts.sp2.isin(species_list))].reset_index(drop=True)
        return coocc_counts

    # For Fig3
    def create_attention_df(self):
        """
        Create a DataFrame of the accumulated attention values
        for all species and environmental variables across predicted dates.
        """

        def process_species(species, env_list, date_list_predict):
            sp_attention = np.zeros([len(env_list), self.height, self.width], dtype=np.float32)
            for date in date_list_predict:
                with h5py.File(self.attention_h5_path.replace('[SPECIES]', species).replace('[DATE]', date), 'r') as hf:
                    for i_env, env in enumerate(env_list):
                        sp_attention[i_env] += hf[env][:]
            sp_attention[sp_attention < 0] = np.nan
            species_attention = np.nanmean(sp_attention, axis=(1, 2)) / len(date_list_predict)
            return [species] + species_attention.tolist()

        results = Parallel(n_jobs=-1)(
            delayed(process_species)(species, self.env_list, self.date_list_predict)
            for species in self.species_list_predict
        )

        df_attention = pd.DataFrame(results, columns=['Species'] + self.env_list).set_index('Species')
        return df_attention

    # For Fig3
    def get_species_habitat(self, df_attention_order):
        """
        Retrieve habitat-related information from the trait dataset for each species
        and map them to a final list for plotting/analysis.
        """
        taxon_df = pd.read_csv(self.traitdataset_taxon_path, delimiter='\t')
        measurement_df = pd.read_csv(self.traitdataset_mesurement_path, delimiter='\t')
        habitat_related_df = measurement_df[
            measurement_df['measurementType'].str.contains('Habitat', na=False, case=False)
        ]
        merged_habitat_df = habitat_related_df.merge(taxon_df, left_on='id', right_on='taxonID', how='left')
        merged_habitat_df['measurementValue'] = merged_habitat_df['measurementValue'].astype(float)
        best_habitat = merged_habitat_df.loc[merged_habitat_df.groupby('taxonID')['measurementValue'].idxmax()]
        species_habitat_df = best_habitat[['scientificName', 'measurementType']].copy()

        name_to_change = {
            'Porzana fusca': 'Zapornia fusca',
            'Gallirallus striatus': 'Lewinia striata',
            'Stachyridopsis ruficeps': 'Cyanoderma ruficeps',
            'Pomatorhinus erythrocnemis': 'Erythrogenys erythrocnemis',
            'Alcippe brunnea': 'Schoeniparus brunneus',
            'Ictinaetus malayensis': 'Ictinaetus malaiensis',
            'Poecile varius': 'Sittiparus castaneoventris',
            'Parus holsti': 'Machlolophus holsti',
            'Turdus poliocephalus': 'Turdus niveiceps',
            'Garrulax poecilorhynchus': 'Pterorhinus poecilorhynchus',
            'Garrulax ruficeps': 'Pterorhinus ruficeps',
            'Glaucidium brodiei': 'Taenioptynx brodiei',
            'Tarsiger indicus': 'Tarsiger formosanus'
        }

        df_attention_sp_name = list(df_attention_order.index)
        for i, name in enumerate(df_attention_sp_name):
            if name in name_to_change:
                df_attention_sp_name[i] = name_to_change[name]

        habitat = [
            species_habitat_df.measurementType[species_habitat_df.scientificName == sp].values.tolist()[0]
            for sp in df_attention_sp_name
        ]
        return habitat

    # For Fig4
    def set_df_env(self):
        """
        Create a DataFrame of normalized environmental variables for all training dates
        within the valid extent of interest.
        """
        df_env = pd.DataFrame()
        for date in self.date_list_train:
            df_date = pd.DataFrame()
            for env in self.env_list:
                env_path = os.path.join(self.env_info['dir_base'], self.env_info['info'][env][date]['tif_span_avg'])
                with rasterio.open(env_path, 'r') as f:
                    env_value = (f.read(1))[self.extent_binary == 1].flatten()
                if env not in self.DeepSDM_conf.training_conf['non_normalize_env_list']:
                    env_value = (env_value - self.env_info['info'][env]['mean']) / self.env_info['info'][env]['sd']
                df_date[env] = env_value
            df_env = pd.concat([df_env, df_date], ignore_index=True)
        return df_env

    # For Fig4
    def set_df_env_pca(self, pca):
        """
        Apply PCA transformation to environmental data, for each date,
        and store combined results in a single DataFrame.
        """
        df_env_pca = []
        for date in self.date_list_train:
            env_data = []
            original_col_names = []
            final_col_names = []

            for env in self.env_list:
                env_path = os.path.join(self.env_info['dir_base'], self.env_info['info'][env][date]['tif_span_avg'])
                with rasterio.open(env_path, 'r') as f:
                    raster_data = f.read(1)[self.extent_binary == 1].flatten()

                if env not in self.DeepSDM_conf.training_conf['non_normalize_env_list']:
                    raster_data = (raster_data - self.env_info['info'][env]['mean']) / self.env_info['info'][env]['sd']

                env_data.append(raster_data)
                original_col_names.append(f'{env}')
                final_col_names.append(f'{env}_{date}')

            df_month = pd.DataFrame(np.array(env_data).T, columns=original_col_names)
            df_pca = pd.DataFrame(
                data=pca.transform(df_month),
                columns=[f'PC{i+1:02d}_{date}' for i in range(len(pca.components_))]
            )
            df_month.columns = final_col_names
            df_env_pca.append(pd.concat([df_month, df_pca], axis=1))

        df_env_pca = pd.concat(df_env_pca, axis=1, ignore_index=False)
        feather.write_dataframe(df_env_pca, self.df_env_pca_path)
        return df_env_pca

    # For Fig4
    def set_df_species(self):
        """
        For each species, read raster predictions from
        DeepSDM and MaxEnt models and occurrence data.
        Create and store in a feather file.
        """
        create_folder(os.path.dirname(self.plot_path_df_species))

        def process_species(species):
            series_dict = {}

            maxent_h5_species_path = self.maxent_h5_path.replace('[SPECIES]', species)
            # if os.path.exists(maxent_h5_species_path):
            #     with h5py.File(maxent_h5_species_path, 'r') as hf:
            #         maxent_all_all_value = (hf['all'][:])[self.extent_binary == 1].flatten()
            #         series_dict[f'maxent_all_all_{species}'] = maxent_all_all_value

            for date in self.date_list_predict:
                deepsdm_h5_species_path = self.deepsdm_h5_path.replace('[SPECIES]', species)
                if os.path.exists(deepsdm_h5_species_path):
                    # print(f'{deepsdm_h5_species_path} exists. ')
                    with h5py.File(deepsdm_h5_species_path, 'r') as hf:
                        deepsdm_all_month_value = (hf[date][:])[self.extent_binary == 1].flatten()
                        series_dict[f'deepsdm_all_month_{species}_{date}'] = deepsdm_all_month_value
                        # print(series_dict)
                else:
                    print(f'{deepsdm_h5_species_path} does not exist. ')
                if os.path.exists(maxent_h5_species_path):
                    with h5py.File(maxent_h5_species_path, 'r') as hf:
                        maxent_all_month_value = (hf[date][:])[self.extent_binary == 1].flatten()
                        series_dict[f'maxent_all_month_{species}_{date}'] = maxent_all_month_value

                occ_path = os.path.join(self.sp_info['dir_base'], self.sp_info['file_name'][species]['h5file_name'])
                if os.path.exists(occ_path):
                    with h5py.File(occ_path, 'r') as hf:
                        occ_value = (hf[date][:][self.extent_binary == 1]).flatten()
                        series_dict[f'occ_{species}_{date}'] = occ_value.astype(int)

            df_species = pd.DataFrame(series_dict)
            output_path = self.plot_path_df_species.replace('[SPECIES]', species)
            feather.write_dataframe(df_species, output_path)

        workers = min(8, (os.cpu_count() or 4))
        n_ok = n_err = 0
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_species, sp): sp for sp in self.species_list_predict}
            for fut in as_completed(futures):
                sp = futures[fut]
                try:
                    fut.result()
                    print(f"[ok] {sp}")
                    n_ok += 1
                except Exception as e:
                    n_err += 1
                    print(f"[error] {sp}: {e}")
                    traceback.print_exc()
        
        print(f"summary: {n_ok} ok, {n_err} error")

    # For Fig4
    def set_pc_bin_extent_info(self, df_env_pca):
        """
        Calculate min/max PC values across all dates, create bin edges,
        and define final extent for subsequent usage.
        """
        pc_info = dict(
            **{
                f'PC{(n_pc+1):02d}_max': df_env_pca.loc[:, df_env_pca.columns.str.startswith(f'PC{(n_pc+1):02d}')].max().max().tolist()
                for n_pc in range(len(self.env_list))
            },
            **{
                f'PC{(n_pc+1):02d}_min': df_env_pca.loc[:, df_env_pca.columns.str.startswith(f'PC{(n_pc+1):02d}')].min().min().tolist()
                for n_pc in range(len(self.env_list))
            }
        )

        bin_info = {
            f'PC{(n_pc+1):02d}_bins': np.linspace(
                pc_info[f'PC{(n_pc+1):02d}_min'],
                pc_info[f'PC{(n_pc+1):02d}_max'],
                num=self.niche_rst_size + 1
            )
            for n_pc in range(len(self.env_list))
        }
        bin_info_list = {key: bin_info[key].tolist() for key in bin_info}

        extent_info = dict(
            **{
                f'PC{(n_pc+1):02d}_extent_max':
                ((bin_info[f'PC{(n_pc+1):02d}_bins'][1:] + bin_info[f'PC{(n_pc+1):02d}_bins'][:-1]) / 2)[-1].tolist()
                for n_pc in range(len(self.env_list))
            },
            **{
                f'PC{(n_pc+1):02d}_extent_min':
                ((bin_info[f'PC{(n_pc+1):02d}_bins'][1:] + bin_info[f'PC{(n_pc+1):02d}_bins'][:-1]) / 2)[0].tolist()
                for n_pc in range(len(self.env_list))
            }
        )

        with open(self.pc_info_path, 'w') as yaml_file:
            yaml.dump(pc_info, yaml_file)
        with open(self.bin_info_path, 'w') as yaml_file:
            yaml.dump(bin_info_list, yaml_file)
        with open(self.extent_info_path, 'w') as yaml_file:
            yaml.dump(extent_info, yaml_file)

        return pc_info, bin_info_list, extent_info

    # For Fig4
    def set_df_grid(self, df_env_pca, bin_info):
        """
        Generate binned PC values for each pixel in the extent,
        resulting in a grid-based DataFrame.
        """
        df_grid = pd.DataFrame()
        for n_pc in range(len(self.env_list)):
            df_env_pca_pc = df_env_pca.filter(regex=f'^PC{(n_pc+1):02d}')

            def apply_cut(column):
                return pd.cut(column,
                              bins=bin_info[f'PC{(n_pc+1):02d}_bins'],
                              labels=False,
                              include_lowest=True)

            grid = df_env_pca_pc.apply(apply_cut)
            grid.columns = [f'{col}_grid' for col in grid.columns]
            df_grid = pd.concat([df_grid, grid], axis=1)
        return df_grid

    # For Fig4
    def create_species_nichespace(self):
        """
        Calculate niche-space grids for each species using 
        DeepSDM and MaxEnt predictions. Writes output
        grids to HDF5 and saves PNG representations.
        """
        create_folder(os.path.dirname(self.plot_path_nichespace_h5))
        df_grid = feather.read_dataframe(self.df_grid_path)

        for species in self.species_list_predict:
            create_folder(os.path.dirname(self.plot_path_nichespace_png_sp.replace('[SPECIES]', species)))
            df_species = feather.read_dataframe(self.plot_path_df_species.replace('[SPECIES]', species))
            df_species = pd.concat([df_species, df_grid], axis=1)

            month_sp_date = [(species, d) for d in self.date_list_predict]
            grid_deepsdm_all_month_sum = np.zeros((self.niche_rst_size, self.niche_rst_size))
            grid_deepsdm_all_month_count = np.zeros((self.niche_rst_size, self.niche_rst_size))
            grid_deepsdm_all_month_max = np.zeros((self.niche_rst_size, self.niche_rst_size))
            grid_maxent_all_month_sum = np.zeros((self.niche_rst_size, self.niche_rst_size))
            grid_maxent_all_month_count = np.zeros((self.niche_rst_size, self.niche_rst_size))
            grid_maxent_all_month_max = np.zeros((self.niche_rst_size, self.niche_rst_size))

            for (sp, d) in month_sp_date:
                grouped = df_species.groupby([
                    f'PC{self.x_pca:02d}_{d}_grid',
                    f'PC{self.y_pca:02d}_{d}_grid'
                ])
                grid_deepsdm_all_month_sum_d = np.zeros((self.niche_rst_size, self.niche_rst_size))
                grid_deepsdm_all_month_count_d = np.zeros((self.niche_rst_size, self.niche_rst_size))
                grid_deepsdm_all_month_max_d = np.zeros((self.niche_rst_size, self.niche_rst_size))

                deepsdm_key = f'deepsdm_all_month_{sp}_{d}'
                try:
                    max_values_deepsdm_all_month = grouped[deepsdm_key].max()
                    sum_values_deepsdm_all_month = grouped[deepsdm_key].sum()
                    count_values_deepsdm_all_month = grouped[deepsdm_key].count()
                    for (x, y), value in max_values_deepsdm_all_month.items():
                        grid_deepsdm_all_month_max_d[self.niche_rst_size-1-y, x] = value
                    for (x, y), value in count_values_deepsdm_all_month.items():
                        grid_deepsdm_all_month_count_d[self.niche_rst_size-1-y, x] = value
                    for (x, y), value in sum_values_deepsdm_all_month.items():
                        grid_deepsdm_all_month_sum_d[self.niche_rst_size-1-y, x] = value
                except KeyError:
                    pass

                grid_deepsdm_all_month_max = np.nanmax([grid_deepsdm_all_month_max,
                                                        grid_deepsdm_all_month_max_d], axis=0)
                grid_deepsdm_all_month_count = grid_deepsdm_all_month_count + grid_deepsdm_all_month_count_d
                grid_deepsdm_all_month_sum = grid_deepsdm_all_month_sum + grid_deepsdm_all_month_sum_d

                grid_maxent_all_month_sum_d = np.zeros((self.niche_rst_size, self.niche_rst_size))
                grid_maxent_all_month_count_d = np.zeros((self.niche_rst_size, self.niche_rst_size))
                grid_maxent_all_month_max_d = np.zeros((self.niche_rst_size, self.niche_rst_size))

                maxent_key = f'maxent_all_month_{sp}_{d}'
                try:
                    max_values_maxent_all_month = grouped[maxent_key].max()
                    sum_values_maxent_all_month = grouped[maxent_key].sum()
                    count_values_maxent_all_month = grouped[maxent_key].count()
                    for (x, y), value in max_values_maxent_all_month.items():
                        grid_maxent_all_month_max_d[self.niche_rst_size-1-y, x] = value
                    for (x, y), value in count_values_maxent_all_month.items():
                        grid_maxent_all_month_count_d[self.niche_rst_size-1-y, x] = value
                    for (x, y), value in sum_values_maxent_all_month.items():
                        grid_maxent_all_month_sum_d[self.niche_rst_size-1-y, x] = value
                except KeyError:
                    pass

                grid_maxent_all_month_max = np.nanmax([grid_maxent_all_month_max,
                                                       grid_maxent_all_month_max_d], axis=0)
                grid_maxent_all_month_count = grid_maxent_all_month_count + grid_maxent_all_month_count_d
                grid_maxent_all_month_sum = grid_maxent_all_month_sum + grid_maxent_all_month_sum_d

            grid_maxent_all_month_count = np.where(grid_maxent_all_month_count == 0, 1, grid_maxent_all_month_count)
            grid_deepsdm_all_month_count = np.where(grid_deepsdm_all_month_count == 0, 1, grid_deepsdm_all_month_count)

            grid_deepsdm_all_month_mean = grid_deepsdm_all_month_sum / grid_deepsdm_all_month_count
            grid_maxent_all_month_mean = grid_maxent_all_month_sum / grid_maxent_all_month_count

            logpng_info = {
                'deepsdm_all_month_max': grid_deepsdm_all_month_max,
                'deepsdm_all_month_mean': grid_deepsdm_all_month_mean,
                'deepsdm_all_month_sum': grid_deepsdm_all_month_sum,
                'maxent_all_month_max': grid_maxent_all_month_max,
                'maxent_all_month_mean': grid_maxent_all_month_mean,
                'maxent_all_month_sum': grid_maxent_all_month_sum
            }

            for key, data in logpng_info.items():
                plot_output = self.plot_path_nichespace_png_sp.replace('[SPECIES]', species).replace('[SUFFIX]', key)
                cv2.imwrite(plot_output, png_operation(data))

            logh5_info = {
                'deepsdm_all_month_max': grid_deepsdm_all_month_max,
                'deepsdm_all_month_mean': grid_deepsdm_all_month_mean,
                'deepsdm_all_month_sum': grid_deepsdm_all_month_sum,
                'maxent_all_month_max': grid_maxent_all_month_max,
                'maxent_all_month_mean': grid_maxent_all_month_mean,
                'maxent_all_month_sum': grid_maxent_all_month_sum
            }
            for key, data in logh5_info.items():
                with h5py.File(self.plot_path_nichespace_h5.replace('[SPECIES]', species), 'a') as hf:
                    if key in hf:
                        del hf[key]
                    hf.create_dataset(key, data=data)
        print(f'all species nichespace pngs are saved in {self.plot_path_nichespace_png_sp}')
        print(f'all species nichespace rasters are saved in {self.plot_path_nichespace_h5}')

    # For Fig4
    def calculate_spearman(self, extent_info):
        """
        Robust Spearman correlation between Mahalanobis distance from niche centroid
        and predicted suitability for each species and raster, with safeguards to
        prevent warnings due to empty inputs, division by zero, or singular covariance.
        """
        import numpy as np
        import pandas as pd
        import h5py
        from scipy.stats import spearmanr
        from numpy.linalg import pinv
    
        # --- helper functions ---
        def _grid_centers(extent, n):
            # Compute the grid center coordinates
            x0, x1, y0, y1 = extent
            cw = (x1 - x0) / n
            ch = (y1 - y0) / n
            xs = x0 + (np.arange(n) + 0.5) * cw
            ys = y1 - (np.arange(n) + 0.5) * ch
            X, Y = np.meshgrid(xs, y1 - (np.arange(n) + 0.5) * ch)  # from top to bottom
            return X, Y
    
        def _weighted_centroid(values, X, Y, eps=1e-12):
            # Compute a weighted centroid, skip grids with all zeros or NaNs
            mask = (values > 0) & np.isfinite(values)
            if not np.any(mask):
                return np.nan, np.nan
            v = values[mask]
            x = X[mask]
            y = Y[mask]
            wsum = v.sum()
            if wsum <= eps:
                return np.nan, np.nan
            return (np.sum(v * x) / wsum, np.sum(v * y) / wsum)
    
        def _mahalanobis_distances(cx, cy, Xv, Yv, eps=1e-9):
            # Compute Mahalanobis distance for valid coordinates
            coords = np.column_stack([Xv, Yv])
            if coords.shape[0] < 3:
                return None
            if (np.ptp(coords[:, 0]) == 0) or (np.ptp(coords[:, 1]) == 0):
                return None
            cov = np.cov(coords, rowvar=False)
            if not np.all(np.isfinite(cov)):
                return None
            cov = cov + eps * np.eye(2)
            VI = pinv(cov)
            deltas = coords - np.array([cx, cy])[None, :]
            md2 = np.einsum('ij,jk,ik->i', deltas, VI, deltas)
            md2 = np.maximum(md2, 0.0)
            return np.sqrt(md2)
    
        # --- prepare grid and coordinates ---
        extent = [
            extent_info[f'PC{self.x_pca:02d}_extent_min'],
            extent_info[f'PC{self.x_pca:02d}_extent_max'],
            extent_info[f'PC{self.y_pca:02d}_extent_min'],
            extent_info[f'PC{self.y_pca:02d}_extent_max'],
        ]
        Xc, Yc = _grid_centers(extent, self.niche_rst_size)
        Xc_flat = Xc.ravel()
        Yc_flat = Yc.ravel()
    
        results = []
    
        # --- iterate through species ---
        for species in self.species_list_predict:
            with h5py.File(self.plot_path_nichespace_h5.replace('[SPECIES]', species), 'r') as hf:
                rasters = {
                    'deepsdm_max': hf['deepsdm_all_month_max'][:],
                    'deepsdm_mean': hf['deepsdm_all_month_mean'][:],
                    'deepsdm_sum': hf['deepsdm_all_month_sum'][:],
                    'maxent_max': hf['maxent_all_month_max'][:],
                    'maxent_mean': hf['maxent_all_month_mean'][:],
                    'maxent_sum': hf['maxent_all_month_sum'][:],
                }
    
            # Compute weighted centroids for each raster
            centroids = {}
            for name, grid in rasters.items():
                cx, cy = _weighted_centroid(grid, Xc, Yc)
                centroids[name] = (cx, cy)
    
            # Compute Mahalanobis distances and Spearman correlation
            for name, grid in rasters.items():
                cx, cy = centroids[name]
                if not np.isfinite(cx) or not np.isfinite(cy):
                    results.append([name, species, np.nan, np.nan])
                    continue
    
                values = grid.ravel()
                valid_mask = (values > 0) & np.isfinite(values)
                if valid_mask.sum() < 3:
                    results.append([name, species, np.nan, np.nan])
                    continue
    
                Xv = Xc_flat[valid_mask]
                Yv = Yc_flat[valid_mask]
                vv = values[valid_mask]
    
                dists = _mahalanobis_distances(cx, cy, Xv, Yv)
                if dists is None:
                    results.append([name, species, np.nan, np.nan])
                    continue
    
                # Skip constant input arrays to avoid ConstantInputWarning
                if (np.ptp(dists) == 0) or (np.ptp(vv) == 0):
                    results.append([name, species, np.nan, np.nan])
                    continue
    
                rho, p = spearmanr(dists, vv)
                if not (np.isfinite(rho) and np.isfinite(p)):
                    rho, p = np.nan, np.nan
                results.append([name, species, rho, p])
    
        df_spearman = pd.DataFrame(results, columns=['model', 'species', 'rho', 'p'])
        df_spearman.to_csv(self.df_spearman_path, index=False)
        print(f'All results of Spearman rho are saved in {self.df_spearman_path}.')
        return df_spearman


    # Fig4
    def calculate_rho_niche_coocccounts(self, species_exclude = []):
        """
        Compute the correlation between co-occurrence counts of two species
        and the cosine similarity of their niche space. 
        Only uses species in the prediction list with non-zero co-occurrence.
        """

        def preload_nichespace_files(coocc_counts):
            nichespace_cache = {}
            unique_species = set(coocc_counts['sp1']).union(set(coocc_counts['sp2']))
            for species in unique_species:
                with h5py.File(self.plot_path_nichespace_h5.replace('[SPECIES]', species), 'r') as hf:
                    for model, suffix in [('deepsdm', 'max'), ('maxent', 'max')]:
                        rst = hf[f'{model}_all_month_{suffix}'][:]
                        nichespace_cache[(species, model)] = rst[rst > 0]
            return nichespace_cache

        def process_row(data, nichespace_cache):
            try:
                niche_space_deepsdm = [nichespace_cache[(sp, 'deepsdm')] for sp in [data.sp1, data.sp2]]
                niche_space_maxent = [nichespace_cache[(sp, 'maxent')] for sp in [data.sp1, data.sp2]]
                if any(ns is None for ns in niche_space_deepsdm) or any(ns is None for ns in niche_space_maxent):
                    return None
                cosine_deepsdm = cosine_similarity_manual(niche_space_deepsdm[0], niche_space_deepsdm[1])
                cosine_maxent = cosine_similarity_manual(niche_space_maxent[0], niche_space_maxent[1])
                return data.counts, cosine_deepsdm, cosine_maxent
            except Exception:
                return None

        if len(species_exclude) != 0:
            species_list = set(self.species_list_predict) - set(species_exclude)
        else:
            species_list = self.species_list_predict
            
        coocc_counts_filter = self.coocc_counts[
            (self.coocc_counts.sp1.isin(species_list)) &
            (self.coocc_counts.sp2.isin(species_list)) &
            (self.coocc_counts.sp1 != self.coocc_counts.sp2) &
            (self.coocc_counts.counts != 0)
        ].reset_index(drop=True)

        nichespace_cache = preload_nichespace_files(coocc_counts_filter)
        results = Parallel(n_jobs=-1)(
            delayed(process_row)(data, nichespace_cache) for _, data in coocc_counts_filter.iterrows()
        )
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            return None
        else:
            counts, deepsdm_cosine, maxent_cosine = zip(*valid_results)
            rho_cosine_deepsdm, p_cosine_deepsdm = spearmanr(np.array(deepsdm_cosine), np.array(counts))
            rho_cosine_maxent, p_cosine_maxent = spearmanr(np.array(maxent_cosine), np.array(counts))
            return rho_cosine_deepsdm, p_cosine_deepsdm, rho_cosine_maxent, p_cosine_maxent

    # For Fig4
    def merge_performance_indicator(self):
        """
        Merge performance indicators from single and multi-threshold
        evaluations into a single DataFrame.
        """
        files = glob.glob(self.performance_indicator_singlethreshold)
        indicator_singlethreshold = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

        files = glob.glob(self.performance_indicator_multithreshold)
        indicator_multithreshold = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

        indicator_multithreshold[['species', 'date']] = indicator_multithreshold['spdate'].str.rsplit('_', n=1, expand=True)
        indicator_multithreshold['date'] = pd.to_datetime(indicator_multithreshold['date'])
        indicator_singlethreshold['date'] = pd.to_datetime(indicator_singlethreshold['date'])

        columns_to_merge = [
            'species', 'date',
            'maxent_all_month_val', 'maxent_all_month_train', 'maxent_all_month_all',
            'deepsdm_all_month_val', 'deepsdm_all_month_train', 'deepsdm_all_month_all'
        ]
        indicator_multithreshold_subset = indicator_multithreshold[columns_to_merge]

        indicator_merged = indicator_singlethreshold.merge(indicator_multithreshold_subset, on=['species', 'date'], how='left')

        indicator_merged.rename(columns={
            'deepsdm_all_month_val': 'DeepSDM_AUC',
            'maxent_all_month_val': 'MaxEnt_AUC',
            'deepsdm_val_TSS': 'DeepSDM_TSS',
            'maxent_val_TSS': 'MaxEnt_TSS',
            'deepsdm_val_kappa': 'DeepSDM_Kappa',
            'maxent_val_kappa': 'MaxEnt_Kappa',
            'deepsdm_val_f1': 'DeepSDM_F1',
            'maxent_val_f1': 'MaxEnt_F1'
        }, inplace=True)

        return indicator_merged

    # For Fig5
    def get_deepsdm_nichespace(self, species_exclude = [], suffix='max'):
        """
        Retrieve the DeepSDM niche space from the stored HDF5 files for
        all species, returning both flattened arrays of non-zero values and
        2D (non-flattened) arrays.
        """
        if len(species_exclude) > 0:
            self.species_list_exclude = [sp for sp in self.species_list_predict if sp not in species_exclude]
        else:
            self.species_list_exclude = self.species_list_predict
        nichespace_deepsdm_all = []
        nichespace_deepsdm_all_nonflatten = []
        for species in self.species_list_exclude:
            h5_path = self.plot_path_nichespace_h5.replace('[SPECIES]', species)
            with h5py.File(h5_path, 'r') as hf:
                deepsdm_dataset_name = f'deepsdm_all_month_{suffix}'
                if deepsdm_dataset_name in hf.keys():
                    nichespace_deepsdm = hf[deepsdm_dataset_name][:].copy()
                    nichespace_deepsdm = nichespace_deepsdm / nichespace_deepsdm.sum().sum()
                    i_nonzeros = np.where(nichespace_deepsdm.flatten() > 0)
                    nichespace_deepsdm_all.append(nichespace_deepsdm.flatten()[i_nonzeros])
                    nichespace_deepsdm_all_nonflatten.append(nichespace_deepsdm)
                else:
                    pass

        nichespace_deepsdm_all = np.vstack(nichespace_deepsdm_all)
        nichespace_deepsdm_all_nonflatten = np.stack(nichespace_deepsdm_all_nonflatten, axis=0)
        return nichespace_deepsdm_all, nichespace_deepsdm_all_nonflatten

    # For Fig5
    def get_all_cluster_average_nichespace(self, cluster_labels, nichespace_deepsdm_all_nonflatten):
        """
        For each cluster, compute an average niche space across species within that cluster.
        """
        unique_clusters = sorted(set(cluster_labels))
        all_cluster_avg_nichespace = []
        for target_cluster in unique_clusters:
            species_in_cluster = [sp for sp, label in zip(self.species_list_predict, cluster_labels) if label == target_cluster]
            niche_sum = None
            valid_species_count = 0
            for sp in species_in_cluster:
                i_sp = self.species_list_predict.index(sp)
                niche_image = nichespace_deepsdm_all_nonflatten[i_sp]
                if niche_sum is None:
                    niche_sum = niche_image
                else:
                    niche_sum += niche_image
                valid_species_count += 1

            if valid_species_count > 0 and niche_sum is not None:
                niche_avg = niche_sum / valid_species_count
                all_cluster_avg_nichespace.append((target_cluster, niche_avg))
            else:
                continue

        return all_cluster_avg_nichespace

    # For Fig5
    def calculate_deepsdm_nichespace_center(self, nichespace_deepsdm_all_nonflatten, cluster_labels):
        """
        Calculate the weighted niche-space center for each species in the cluster.
        """
        center_allspecies = []
        for i_sp in range(nichespace_deepsdm_all_nonflatten.shape[0]):
            nichespace_species = nichespace_deepsdm_all_nonflatten[i_sp]

            coordinates_values = {'center_x': [], 'center_y': [], 'value_deepsdm': []}
            for i in range(self.niche_rst_size):
                for j in range(self.niche_rst_size):
                    coordinates_values['center_x'].append(
                        self.nichespace_extent[0] +
                        j * self.nichespace_cell_width +
                        self.nichespace_cell_width / 2
                    )
                    coordinates_values['center_y'].append(
                        self.nichespace_extent[3] -
                        i * self.nichespace_cell_height -
                        self.nichespace_cell_height / 2
                    )
                    coordinates_values['value_deepsdm'].append(nichespace_species[i, j])

            df_cor = pd.DataFrame(coordinates_values).query('value_deepsdm > 0').reset_index(drop=True)
            center_x = np.multiply(np.array(df_cor['center_x']), np.array(df_cor['value_deepsdm'])).sum() / np.array(df_cor['value_deepsdm']).sum()
            center_y = np.multiply(np.array(df_cor['center_y']), np.array(df_cor['value_deepsdm'])).sum() / np.array(df_cor['value_deepsdm']).sum()
            center = np.array([center_x, center_y])
            center_allspecies.append(center)

        df_center = pd.DataFrame(np.vstack(center_allspecies), index=self.species_list_exclude, columns=['PC01', 'PC02'])
        df_center['cluster'] = cluster_labels
        self.df_nichespace_center_coordinate = df_center
        return df_center

    # For Fig5
    def calculate_nichecenter_cluster_lda(self, SEED = 42):
        
        L = self.env_pca_loadings[['PC01', 'PC02']].to_numpy(float)
        mu = self.df_nichespace_center_coordinate[['PC01', 'PC02']].to_numpy(float)
        labels = self.df_nichespace_center_coordinate['cluster'].astype(str).to_numpy()
        species = self.df_nichespace_center_coordinate.index.astype(str).to_numpy()
        
        # fit & compute
        Imp, Uhat, w, L2 = lda_pcspace_importance(mu, labels, self.env_pca_loadings, use_cosine_only=False)
        order = np.argsort(-Imp)
        res_pc = pd.DataFrame({
            'variable': self.env_pca_loadings.index.values[order],
            'importance_pc': Imp[order],
            'loading_PC1': self.env_pca_loadings.iloc[order]['PC01'].values,
            'loading_PC2': self.env_pca_loadings.iloc[order]['PC02'].values
        })
        Uhat_df = pd.DataFrame(Uhat, index=['PC01', 'PC02'],
                       columns=[f'LD{i+1}' for i in range(Uhat.shape[1])])
        # Projection of environmental factors on LDA space: 
        P = (L2 @ Uhat)  # (p, q)
        for j in range(P.shape[1]):
            res_pc[f'proj_on_LD{j+1}'] = P[order, j]
        
        # Stratified K-fold cross-validation (external evaluation)
        counts = pd.Series(labels).value_counts(); min_count = int(counts.min())
        K = max(2, min(5, min_count))
        cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
        scores = cross_val_score(LinearDiscriminantAnalysis(solver='svd'), mu, labels, cv=cv)
        print(f'Stratified {K}-fold CV accuracy: {scores.mean():.3f} (± {scores.std():.3f})')

        return res_pc, Uhat_df, w, scores
        
    # For Fig5
    def calculate_correlation_center_maxvariance(self, df_center):
        """
        Find the max-variance direction of cluster centers (PC01, PC02) and compute
        cosine similarity between each environmental factor's loading vector and that direction.
        Cosine similarity removes the effect of vector magnitude.
        """
        # 1) Covariance & eigen (use eigh for symmetric)
        cov_matrix = np.cov(df_center[['PC01', 'PC02']].T)
        evals, evecs = np.linalg.eigh(cov_matrix)          # eigenvectors are orthonormal
        max_idx = np.argmax(evals)
        dir_vec = evecs[:, max_idx]                         # already unit-length with eigh, but we can ensure:
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
    
        # 2) Environmental factor loadings (each row = a factor's vector in PC space)
        env = self.env_pca_loadings[['PC01', 'PC02']].values  # shape: (n_factors, 2)
        row_norms = np.linalg.norm(env, axis=1, keepdims=True)
        # avoid divide-by-zero
        safe_norms = np.where(row_norms == 0, 1.0, row_norms)
        env_unit = env / safe_norms
    
        # 3) Cosine similarity (magnitude-invariant)
        cos_sim = env_unit @ dir_vec                         # shape: (n_factors,)
        cos_sim_abs = np.abs(cos_sim)
    
        # 4) 也可提供純投影（含量級）供參考
        projection = env @ dir_vec                           # 受向量大小影響
    
        # 5) 結果表
        out = pd.DataFrame({
            "Environmental Factor": self.env_pca_loadings.index,
            "Cosine with Max-Var Direction": cos_sim,        # 範圍 [-1, 1]，不受向量長度影響
            "Abs(Cosine)": cos_sim_abs,
            "Signed Projection": projection                  # 受向量長度影響（若你也想看「貢獻量」）
        }).sort_values("Abs(Cosine)", ascending=False).reset_index(drop=True)
    
        # 方向斜率（僅供繪圖/直覺解讀）
        slope = dir_vec[1] / dir_vec[0] if dir_vec[0] != 0 else np.inf
        return out, slope

    # For Fig5
    def get_cluster_env_values(self, cluster_labels, env_plot):
        """
        For each cluster, gather environment values where species presence occurs.
        """
        env_value_cluster_all = []
        for cluster in np.unique(cluster_labels):
            species_list_cluster = np.array(self.species_list_exclude)[np.array(cluster_labels) == cluster].tolist()
            env_value_cluster = []
            for species in species_list_cluster:
                df_species = feather.read_dataframe(self.plot_path_df_species.replace('[SPECIES]', species))
                occ_value = df_species[sorted([key for key in df_species.keys() if key.startswith('occ')])].values.flatten()
                env_value = self.df_env_pca[sorted([key for key in self.df_env_pca.keys() if key.startswith(env_plot)])].values.flatten()
                mask = ~np.isnan(occ_value) & ~np.isnan(env_value)
                occ_value = occ_value[mask]
                env_value = env_value[mask]
                i_threshold = np.where(occ_value == 1)[0]
                env_value = env_value[i_threshold]
                env_value_cluster.append(env_value)
            env_value_cluster = np.concatenate(env_value_cluster)
            env_value_cluster_all.append((cluster, env_value_cluster))
        return env_value_cluster_all
        
    # For Fig5
    def align_label_to_color(self, k_optimal, cluster_labels):
        """
        Set the consistant combination between labels of clusters and colors
        """
        unique_labels = np.unique(cluster_labels)
        assert len(unique_labels) == k_optimal
        label_to_color = {int(lbl): self.color_list[i] for i, lbl in enumerate(unique_labels)}
        with open(self.label_to_color_path, 'w') as f:
            json.dump(label_to_color, f)
        return unique_labels, label_to_color
        
    # For Fig6
    def calculate_cph_spearman_beta(self):
        """
        For each species, calculates:
         1) Spearman correlation in geographical space
         2) Spearman correlation in niche space (Mahalanobis distance)
         3) Beta regression parameters (mu link and phi link) for each center type
        """
        ecological_fit_info = {}
        geographical_fit_info = {}
        df_spearman_ecogeo = pd.DataFrame({'center_type': [], 'species': [], 'rho': [], 'p': []})
        epsilon = 1e-8

        for species in self.species_list_predict:
            img_sum = None
            with h5py.File(self.deepsdm_h5_path.replace('[SPECIES]', species), 'r') as hf:
                for date in self.date_list_predict:
                    if img_sum is None:
                        img_sum = hf[date][:].copy()
                    else:
                        img_sum = np.maximum(img_sum, hf[date][:].copy())

            x_indices = np.where(self.extent_binary == 1)[1]
            y_indices = np.where(self.extent_binary == 1)[0]
            lon_lat_pairs = [xy(self.transform, y, x) for y, x in zip(y_indices, x_indices)]
            lons, lats = zip(*lon_lat_pairs)
            lons = np.array(lons)
            lats = np.array(lats)
            valid_mask = ~np.isnan(img_sum)
            total = img_sum[valid_mask].sum()
            y_weighted = (np.where(valid_mask)[0] * img_sum[valid_mask]).sum() / total
            x_weighted = (np.where(valid_mask)[1] * img_sum[valid_mask]).sum() / total
            lon_center, lat_center = rasterio.transform.xy(self.transform, y_weighted, x_weighted, offset='center')
            distances_all = haversine(lats, lons, lat_center, lon_center)
            cell_values_all = img_sum[valid_mask].flatten()
            rho_geo, p_geo = spearmanr(distances_all, cell_values_all)
            df_spearman_ecogeo.loc[len(df_spearman_ecogeo)] = ['Geographical_center', species, rho_geo, p_geo]

            with h5py.File(self.plot_path_nichespace_h5.replace('[SPECIES]', species), 'r') as hf:
                nichespace_deepsdm = hf['deepsdm_all_month_max'][:]

            coordinates_values = {'center_x': [], 'center_y': [], 'value_deepsdm': []}
            for i in range(self.niche_rst_size):
                for j in range(self.niche_rst_size):
                    coordinates_values['center_x'].append(
                        self.nichespace_extent[0] + j * self.nichespace_cell_width + self.nichespace_cell_width / 2
                    )
                    coordinates_values['center_y'].append(
                        self.nichespace_extent[3] - i * self.nichespace_cell_height - self.nichespace_cell_height / 2
                    )
                    coordinates_values['value_deepsdm'].append(nichespace_deepsdm[i, j])
            df_cor = pd.DataFrame(coordinates_values)
            df_cor = df_cor.query('value_deepsdm > 0').reset_index(drop=True)
            df_cor_only = df_cor[['center_x', 'center_y']]
            cov_matrix = np.cov(df_cor_only, rowvar=False)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            center_x = np.average(df_cor['center_x'], weights=df_cor['value_deepsdm'])
            center_y = np.average(df_cor['center_y'], weights=df_cor['value_deepsdm'])
            center = np.array([center_x, center_y])
            df_cor['distance_mah'] = df_cor_only.apply(lambda row: mahalanobis(row, center, inv_cov_matrix), axis=1)
            condition_deepsdm = df_cor['value_deepsdm'] > 0
            rho_eco, p_eco = spearmanr(df_cor['distance_mah'][condition_deepsdm], df_cor['value_deepsdm'][condition_deepsdm])
            df_spearman_ecogeo.loc[len(df_spearman_ecogeo)] = ['Niche_center', species, rho_eco, p_eco]

            x_eco = df_cor['distance_mah'].values
            y_eco = df_cor['value_deepsdm'].values
            y_eco_normalize = (y_eco - np.min(y_eco)) / (np.max(y_eco) - np.min(y_eco)) * (1 - 2 * epsilon) + epsilon
            X_eco = sm.add_constant(x_eco)
            from statsmodels.genmod.families.links import Logit as smLogit
            model_eco = Beta(endog=y_eco_normalize,
                             exog=X_eco,
                             Z=None,
                             link=Logit(),
                             link_phi=sm.families.links.Log())
            result_eco = model_eco.fit(disp=False)
            params_eco = result_eco.params
            intercept_mu_eco = params_eco[0]
            slope_mu_eco = params_eco[1]
            intercept_phi_eco = params_eco[2]
            x_min_eco = float(np.min(x_eco))
            x_max_eco = float(np.max(x_eco))
            y_min_eco = float(np.min(y_eco))
            y_max_eco = float(np.max(y_eco))
            ecological_fit_info[species] = {
                "mu_link": "logit",
                "phi_link": "log",
                "params_mu": [float(intercept_mu_eco), float(slope_mu_eco)],
                "params_phi": [float(intercept_phi_eco)],
                "x_min": x_min_eco,
                "x_max": x_max_eco,
                "y_min": y_min_eco,
                "y_max": y_max_eco,
                "epsilon": epsilon
            }

            x_geo = distances_all
            y_geo = cell_values_all
            y_geo_normalize = (y_geo - np.min(y_geo)) / (np.max(y_geo) - np.min(y_geo)) * (1 - 2 * epsilon) + epsilon
            X_geo = sm.add_constant(x_geo)
            model_geo = Beta(endog=y_geo_normalize,
                             exog=X_geo,
                             Z=None,
                             link=Logit(),
                             link_phi=sm.families.links.Log())
            result_geo = model_geo.fit(disp=False)
            params_geo = result_geo.params
            intercept_mu_geo = params_geo[0]
            slope_mu_geo = params_geo[1]
            intercept_phi_geo = params_geo[2]
            x_min_geo = float(np.min(x_geo))
            x_max_geo = float(np.max(x_geo))
            y_min_geo = float(np.min(y_geo))
            y_max_geo = float(np.max(y_geo))
            geographical_fit_info[species] = {
                "mu_link": "logit",
                "phi_link": "log",
                "params_mu": [float(intercept_mu_geo), float(slope_mu_geo)],
                "params_phi": [float(intercept_phi_geo)],
                "x_min": x_min_geo,
                "x_max": x_max_geo,
                "y_min": y_min_geo,
                "y_max": y_max_geo,
                "epsilon": epsilon
            }

        # Export the DataFrame of Spearman's correlation to CSV.
        # Save the niche-based Beta regression parameters and the geographic-based Beta regression parameters in JSON.
        df_spearman_ecogeo.to_csv(self.df_spearman_ecogeo_path, index = None)
        self.df_spearman_ecogeo = df_spearman_ecogeo
        with open(self.niche_beta_params_path, 'w') as f:
            json.dump(ecological_fit_info, f, indent = 4)
            self.niche_beta_params = ecological_fit_info
        with open(self.geographical_beta_params_path, 'w') as f:
            json.dump(geographical_fit_info, f, indent = 4)
            self.geographical_beta_params = geographical_fit_info
            
        return df_spearman_ecogeo, ecological_fit_info, geographical_fit_info

    # For Fig6
    def calculate_cph_spearman_beta_cluster(self, species_exclude):
        """
        Compute Beta regression in geographical/niche space 
        for species grouped by cluster. Summarizes results 
        for each cluster.
        """
        ecological_fit_info = {}
        geographical_fit_info = {}
        epsilon = 1e-8
            
        unique_label = np.unique(list(self.label_to_color.keys()))
        
        if len(species_exclude) > 0:
            self.species_list_exclude = [sp for sp in self.species_list_predict if sp not in species_exclude]
        else:
            self.species_list_exclude = self.species_list_predict
            
        for i, lbl in enumerate(unique_label):
            print(lbl)
            print(type(lbl))
            species_list_cluster = np.array(self.species_list_exclude)[np.array(self.cluster_labels) == int(lbl)]
            print(species_list_cluster)
            df_cluster = []
            distances_all_cluster = []
            cell_values_all_cluster = []
            for species in species_list_cluster:
                img_sum = None
                with h5py.File(self.deepsdm_h5_path.replace('[SPECIES]', species), 'r') as hf:
                    for date in self.date_list_predict:
                        if img_sum is None:
                            img_sum = hf[date][:].copy()
                        else:
                            img_sum = np.maximum(img_sum, hf[date][:].copy())

                x_indices = np.where(self.extent_binary == 1)[1]
                y_indices = np.where(self.extent_binary == 1)[0]
                lon_lat_pairs = [xy(self.transform, y, x) for y, x in zip(y_indices, x_indices)]
                lons, lats = zip(*lon_lat_pairs)
                lons = np.array(lons)
                lats = np.array(lats)
                valid_mask = ~np.isnan(img_sum)
                total = img_sum[valid_mask].sum()
                y_weighted = (np.where(valid_mask)[0] * img_sum[valid_mask]).sum() / total
                x_weighted = (np.where(valid_mask)[1] * img_sum[valid_mask]).sum() / total
                lon_center, lat_center = rasterio.transform.xy(self.transform, y_weighted, x_weighted, offset='center')
                distances_all = haversine(lats, lons, lat_center, lon_center)
                cell_values_all = img_sum[valid_mask].flatten()
                distances_all_cluster.append(distances_all)
                cell_values_all_cluster.append(cell_values_all)

                with h5py.File(self.plot_path_nichespace_h5.replace('[SPECIES]', species), 'r') as hf:
                    nichespace_deepsdm = hf['deepsdm_all_month_max'][:]

                coordinates_values = {'center_x': [], 'center_y': [], 'value_deepsdm': []}
                for i in range(self.niche_rst_size):
                    for j in range(self.niche_rst_size):
                        coordinates_values['center_x'].append(
                            self.nichespace_extent[0] + j * self.nichespace_cell_width + self.nichespace_cell_width / 2
                        )
                        coordinates_values['center_y'].append(
                            self.nichespace_extent[3] - i * self.nichespace_cell_height - self.nichespace_cell_height / 2
                        )
                        coordinates_values['value_deepsdm'].append(nichespace_deepsdm[i, j])
                df_cor = pd.DataFrame(coordinates_values)
                df_cor = df_cor.query('value_deepsdm > 0').reset_index(drop=True)
                df_cor_only = df_cor[['center_x', 'center_y']]
                cov_matrix = np.cov(df_cor_only, rowvar=False)
                inv_cov_matrix = np.linalg.inv(cov_matrix)
                center_x = np.average(df_cor['center_x'], weights=df_cor['value_deepsdm'])
                center_y = np.average(df_cor['center_y'], weights=df_cor['value_deepsdm'])
                center = np.array([center_x, center_y])
                df_cor['distance_mah'] = df_cor_only.apply(lambda row: mahalanobis(row, center, inv_cov_matrix), axis=1)
                df_cluster.append(df_cor)

            df_cluster_cor = pd.concat(df_cluster)
            distances_all_cluster = np.concatenate(distances_all_cluster)
            cell_values_all_cluster = np.concatenate(cell_values_all_cluster)

            x_eco = df_cluster_cor['distance_mah'].values
            y_eco = df_cluster_cor['value_deepsdm'].values
            y_eco_normalize = (y_eco - np.min(y_eco)) / (np.max(y_eco) - np.min(y_eco)) * (1 - 2 * epsilon) + epsilon
            X_eco = sm.add_constant(x_eco)
            model_eco = Beta(endog=y_eco_normalize,
                             exog=X_eco,
                             Z=None,
                             link=Logit(),
                             link_phi=sm.families.links.Log())
            result_eco = model_eco.fit(disp=False)
            params_eco = result_eco.params
            intercept_mu_eco = params_eco[0]
            slope_mu_eco = params_eco[1]
            intercept_phi_eco = params_eco[2]
            x_min_eco = float(np.min(x_eco))
            x_max_eco = float(np.max(x_eco))
            y_min_eco = float(np.min(y_eco))
            y_max_eco = float(np.max(y_eco))

            ecological_fit_info[lbl] = {
                "mu_link": "logit",
                "phi_link": "log",
                "params_mu": [float(intercept_mu_eco), float(slope_mu_eco)],
                "params_phi": [float(intercept_phi_eco)],
                "x_min": x_min_eco,
                "x_max": x_max_eco,
                "y_min": y_min_eco,
                "y_max": y_max_eco,
                "epsilon": epsilon
            }

            with open(self.beta_cluster_result_path.replace('[CENTER_TYPE]', 'Niche_center').replace('[CLUSTER]', lbl), 'w') as file:
                file.write(result_eco.summary().as_text())

            x_geo = distances_all_cluster
            y_geo = cell_values_all_cluster
            y_geo_normalize = (y_geo - np.min(y_geo)) / (np.max(y_geo) - np.min(y_geo)) * (1 - 2 * epsilon) + epsilon
            X_geo = sm.add_constant(x_geo)
            model_geo = Beta(endog=y_geo_normalize,
                             exog=X_geo,
                             Z=None,
                             link=Logit(),
                             link_phi=sm.families.links.Log())
            result_geo = model_geo.fit(disp=False)
            params_geo = result_geo.params
            intercept_mu_geo = params_geo[0]
            slope_mu_geo = params_geo[1]
            intercept_phi_geo = params_geo[2]
            x_min_geo = float(np.min(x_geo))
            x_max_geo = float(np.max(x_geo))
            y_min_geo = float(np.min(y_geo))
            y_max_geo = float(np.max(y_geo))

            geographical_fit_info[lbl] = {
                "mu_link": "logit",
                "phi_link": "log",
                "params_mu": [float(intercept_mu_geo), float(slope_mu_geo)],
                "params_phi": [float(intercept_phi_geo)],
                "x_min": x_min_geo,
                "x_max": x_max_geo,
                "y_min": y_min_geo,
                "y_max": y_max_geo,
                "epsilon": epsilon
            }

            with open(self.beta_cluster_result_path.replace('[CENTER_TYPE]', 'Geographical_center').replace('[CLUSTER]', lbl), 'w') as file:
                file.write(result_geo.summary().as_text())

        # Write the Beta regression results (cluster-based) for niche and geographical centers.
        with open(self.niche_beta_cluster_params_path, 'w') as f:
            json.dump(ecological_fit_info, f, indent = 4)
            self.niche_beta_cluster_params = ecological_fit_info
        with open(self.geographical_beta_cluster_params_path, 'w') as f:
            json.dump(geographical_fit_info, f, indent = 4)
            self.geographical_beta_cluster_params = geographical_fit_info
        return ecological_fit_info, geographical_fit_info

    # For Fig6
    def plot_subplots(self, species_to_print, species_exclude = []):
        """
        Generate multiple figure subplots for each species to visualize:
         1) Niche space predictions
         2) Geographic predictions
         3) Beta regression results (niche space)
         4) Beta regression results (geographic space)
        """
        [create_folder(os.path.join(self.plot_path_cph_subplots, f'Fig6_subplots_{i+1}')) for i in range(4)]
        x_indices = np.where(self.extent_binary == 1)[1]
        y_indices = np.where(self.extent_binary == 1)[0]
        lon_lat_pairs = [xy(self.transform, y, x) for y, x in zip(y_indices, x_indices)]
        lons, lats = zip(*lon_lat_pairs)
        lons = np.array(lons)
        lats = np.array(lats)

        if len(species_exclude) > 0:
            self.species_list_exclude = [sp for sp in self.species_list_predict if sp not in species_exclude]
        else:
            self.species_list_exclude = self.species_list_predict
        
        for species in species_to_print:
            img_sum = None
            with h5py.File(self.deepsdm_h5_path.replace('[SPECIES]', species), 'r') as hf:
                for date in self.date_list_predict:
                    if img_sum is None:
                        img_sum = hf[date][:].copy()
                    else:
                        img_sum = np.maximum(img_sum, hf[date][:].copy())

            lon_center, lat_center = rasterio.transform.xy(
                self.transform,
                (np.where(~np.isnan(img_sum))[0] * img_sum[~np.isnan(img_sum)]).sum() / img_sum[~np.isnan(img_sum)].sum(),
                (np.where(~np.isnan(img_sum))[1] * img_sum[~np.isnan(img_sum)]).sum() / img_sum[~np.isnan(img_sum)].sum(),
                offset='center'
            )
            distances_all = haversine(lats, lons, lat_center, lon_center)
            cell_values_all = img_sum[~np.isnan(img_sum)].flatten()

            df_species = feather.read_dataframe(self.plot_path_df_species.replace('[SPECIES]', species))
            df_species = pd.concat([df_species, self.df_grid], axis=1)

            with h5py.File(self.plot_path_nichespace_h5.replace('[SPECIES]', species), 'r') as hf:
                nichespace_deepsdm = hf['deepsdm_all_month_max'][:]

            # Set up niche coordinates
            coordinates_values = {'center_x': [], 'center_y': [], 'value_deepsdm': []}
            for i in range(self.niche_rst_size):
                for j in range(self.niche_rst_size):
                    coordinates_values['center_x'].append(
                        self.nichespace_extent[0] + j * self.nichespace_cell_width + self.nichespace_cell_width / 2
                    )
                    coordinates_values['center_y'].append(
                        self.nichespace_extent[3] - i * self.nichespace_cell_height - self.nichespace_cell_height / 2
                    )
                    coordinates_values['value_deepsdm'].append(nichespace_deepsdm[i, j])

            df_cor = pd.DataFrame(coordinates_values).query('value_deepsdm > 0').reset_index(drop=True)
            df_cor_only = df_cor.loc[df_cor['value_deepsdm'] > 0, ['center_x', 'center_y']].reset_index(drop=True)
            cov_matrix = np.cov(df_cor_only, rowvar=False)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            center_x = np.multiply(np.array(coordinates_values['center_x']), np.array(coordinates_values['value_deepsdm'])).sum() / np.array(coordinates_values['value_deepsdm']).sum()
            center_y = np.multiply(np.array(coordinates_values['center_y']), np.array(coordinates_values['value_deepsdm'])).sum() / np.array(coordinates_values['value_deepsdm']).sum()
            center = np.array([center_x, center_y])
            df_cor['distance_mah'] = df_cor_only.apply(lambda row: mahalanobis(row, center, inv_cov_matrix), axis=1)
            condition_deepsdm = df_cor['value_deepsdm'] > 0

            # Plot niche space (subplot 1)
            fig, ax = plt.subplots(figsize=mm2inch(40*1.2, 50*1.2), gridspec_kw={'left': 0.25, 'right': 0.95, 'bottom': 0.12, 'top': 1-0.05/1.25})
            ax_imshow = ax.imshow(nichespace_deepsdm, cmap='coolwarm', extent=self.nichespace_extent)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_ylabel('PC2')
            ax.set_xlabel('PC1')
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax_pos = ax.get_position()
            cbar_ax = fig.add_axes([
                ax_pos.x0,
                ax_pos.y0 - ax_pos.height*0.38,
                ax_pos.width*1,
                ax_pos.height*0.04
            ])
            cbar = fig.colorbar(ax_imshow, ax=ax, orientation='horizontal', cax=cbar_ax)
            ax.scatter(center[0], center[1], c='black', s=15, marker='x', linewidths=0.8)
            plot_output = os.path.join(self.plot_path_cph_subplots, 'Fig6_subplots_1', f'Fig6_subplots_1_{species}.pdf')
            plt.savefig(plot_output, dpi=500, transparent=True)
            plt.close()
            
            # Plot geographic space (subplot 2)
            fig, ax = plt.subplots(figsize=mm2inch(40*1.1, 50*1.1), gridspec_kw={'left': 0.25, 'right': 0.95, 'bottom': 0.96-28/1.0921832795998423/50, 'top': 0.96})
            ax_imshow = ax.imshow(img_sum, cmap='coolwarm', extent=self.extent_binary_extent, vmin=0)
            ax.set_ylabel('Latitude (N)', size=6.5)
            ax.set_xlabel('Longitude (E)', size=6.5)
            ax.set_xticks([120, 121, 122])
            ax.scatter(lon_center, lat_center, c='black', s=15, marker='x', linewidths=0.8)
            ax_pos = ax.get_position()
            cbar_ax = fig.add_axes([
                ax_pos.x0,
                ax_pos.y0 - ax_pos.height*0.38,
                ax_pos.width*1,
                ax_pos.height*0.04
            ])
            cbar = fig.colorbar(ax_imshow, ax=ax, orientation='horizontal', cax=cbar_ax)
            vmin, vmax = cbar.vmin, cbar.vmax
            ticks = np.arange(np.floor(vmin/0.2)*0.2, np.floor(vmax/0.2)*0.2 + 0.2, 0.2)
            cbar.set_ticks(ticks)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plot_output = os.path.join(self.plot_path_cph_subplots, 'Fig6_subplots_2', f'Fig6_subplots_2_{species}.pdf')
            plt.savefig(plot_output, dpi=500, transparent=True)
            plt.close()

            # Plot Beta regression in niche space (subplot 3)
            fig, ax = plt.subplots(figsize=mm2inch(40*1.2, 40*1.2), gridspec_kw={'left': 0.25, 'right': 0.95, 'bottom': 0.15, 'top': 0.95})
            ax.scatter(df_cor['distance_mah'][condition_deepsdm], df_cor['value_deepsdm'][condition_deepsdm], alpha=0.2, color='grey', s=0.1)
            ax.set_ylabel('Suitablity')
            ax.set_xlabel('Distance')
            ax.set_ylim(bottom=0)
            info = self.niche_beta_params[species]
            params_mu = np.array(info["params_mu"])
            x_min = info["x_min"]
            x_max = info["x_max"]
            y_min = info["y_min"]
            y_max = info["y_max"]
            eps = info["epsilon"]
            x_line = np.linspace(x_min, x_max, 100)
            X_line = np.column_stack([np.ones_like(x_line), x_line])
            linear_pred = X_line @ params_mu
            mu_pred_norm = logistic(linear_pred)
            mu_pred = ((mu_pred_norm - eps) / (1 - 2*eps)) * (y_max - y_min) + y_min
            cluster_idx = self.cluster_labels[self.species_list_exclude.index(species)]
            ax.plot(x_line, mu_pred, color=self.label_to_color[str(cluster_idx)], linewidth=1)
            ax.set_box_aspect(self.nichespace_cell_height / self.nichespace_cell_width)
            plot_output = os.path.join(self.plot_path_cph_subplots, 'Fig6_subplots_3', f'Fig6_subplots_3_{species}.pdf')
            plt.savefig(plot_output, dpi=500, transparent=True)
            plt.close()
            
            plot_output = os.path.join(self.plot_path_cph_subplots, 'Fig6_subplots_3', f'Fig6_subplots_3_{species}.png')
            plt.savefig(plot_output, dpi=2000, transparent=True)
            plt.close()

            # Plot Beta regression in geographic space (subplot 4)
            fig, ax = plt.subplots(figsize=mm2inch(40*1.2, 40*1.2), gridspec_kw={'left': 0.25, 'right': 0.95, 'bottom': 0.15, 'top': 0.95})
            ax.scatter(distances_all, cell_values_all, alpha=0.1, color='grey', s=0.1)
            ax.set_ylabel('Suitablity')
            ax.set_xlabel('Distance (km)')
            ax.set_ylim(bottom=0)
            info = self.geographical_beta_params[species]
            params_mu = np.array(info["params_mu"])
            x_min = info["x_min"]
            x_max = info["x_max"]
            y_min = info["y_min"]
            y_max = info["y_max"]
            eps = info["epsilon"]
            x_line = np.linspace(x_min, x_max, 100)
            X_line = np.column_stack([np.ones_like(x_line), x_line])
            linear_pred = X_line @ params_mu
            mu_pred_norm = logistic(linear_pred)
            mu_pred = ((mu_pred_norm - eps) / (1 - 2*eps)) * (y_max - y_min) + y_min
            cluster_idx = self.cluster_labels[self.species_list_exclude.index(species)]
            ax.plot(x_line, mu_pred, color=self.label_to_color[str(cluster_idx)], linewidth=1)
            ax.set_box_aspect(self.nichespace_cell_height / self.nichespace_cell_width)
            plot_output = os.path.join(self.plot_path_cph_subplots, 'Fig6_subplots_4', f'Fig6_subplots_4_{species}.pdf')
            plt.savefig(plot_output, dpi=500, transparent=True)
            plt.close()

            plot_output = os.path.join(self.plot_path_cph_subplots, 'Fig6_subplots_4', f'Fig6_subplots_4_{species}.png')
            plt.savefig(plot_output, dpi=2000, transparent=True)
            plt.close()

    # For FigSupplementary
    def species_occ_env_violinplot(self, env_plot, species_exclude=[]):
        """
        Kruskal–Wallis tests of an environmental factor across three clusters.
        Each species contributes ONE median value (computed across its occurrence pixels).
        The violin plot shows per-cluster distributions (species-level medians) and
        includes pairwise significance annotations (lines + stars). Results are saved to JSON.
        """
    
        # ---------- Config ----------
        n_clusters = 3
        clusters = list(range(1, n_clusters + 1))
    
        # ---------- Species filtering ----------
        if len(species_exclude) > 0:
            self.species_list_exclude = [sp for sp in self.species_list_predict if sp not in species_exclude]
        else:
            self.species_list_exclude = self.species_list_predict
    
        # ---------- Collect species-level medians ----------
        cluster_env_values = {k: [] for k in clusters}     # species-level medians per cluster
        cluster_species_names = {k: [] for k in clusters}
    
        for cluster in clusters:
            # Species in the current cluster (assumes labels are 1..n_clusters)
            species_list_cluster = np.array(self.species_list_exclude)[np.array(self.cluster_labels) == cluster].tolist()
    
            for species in species_list_cluster:
                # Read species occurrence data
                df_species = feather.read_dataframe(self.plot_path_df_species.replace('[SPECIES]', species))
    
                # Occurrence vector and environmental values for the selected variable
                occ_value = df_species[sorted([k for k in df_species.keys() if k.startswith('occ')])].values.flatten()
                env_value = self.df_env_pca[sorted([k for k in self.df_env_pca.keys() if k.startswith(env_plot)])].values.flatten()
    
                # Remove NaNs
                mask = ~np.isnan(occ_value) & ~np.isnan(env_value)
                if not np.any(mask):
                    continue
                occ_value = occ_value[mask]
                env_value = env_value[mask]
    
                # Environmental values at occurrence pixels only
                i_threshold = np.where(occ_value == 1)[0]
                if i_threshold.size == 0:
                    continue
                env_occ = env_value[i_threshold]
    
                # Species-level median (the unit of analysis)
                sp_median = float(np.median(env_occ))
    
                cluster_env_values[cluster].append(sp_median)
                cluster_species_names[cluster].append(species)
    
        # Guard against empty clusters
        for k in clusters:
            if len(cluster_env_values[k]) == 0:
                print(f"[Warning] Cluster {k} has no species medians for '{env_plot}'. Skipping stats/plot.")
                return
    
        # ---------- Statistics ----------
        # Global Kruskal–Wallis across 3 clusters
        H, p_kw = kruskal(*(cluster_env_values[k] for k in clusters))
    
        # Pairwise Kruskal–Wallis
        pairs = list(itertools.combinations(clusters, 2))
        pairwise_results = {}
        for (a, b) in pairs:
            stat, p = kruskal(cluster_env_values[a], cluster_env_values[b])
            pairwise_results[f'Cluster {a} vs Cluster {b}'] = (stat, p)
    
        # ---------- Violin plot (with significance annotations) ----------
        fig, ax = plt.subplots(figsize=mm2inch(40, 50), constrained_layout=True)
    
        data_for_violin = [cluster_env_values[k] for k in clusters]
    
        # Quantile lines (25% and 75%) for each cluster, only if enough samples
        quantiles_list = []
        for vals in data_for_violin:
            quantiles_list.append([0.25, 0.75] if len(vals) >= 4 else [])
    
        violin = ax.violinplot(
            data_for_violin,
            showmeans=False, showmedians=True, showextrema=True, widths=0.6,
            quantiles=quantiles_list
        )
    
        # Appearance
        for i, body in enumerate(violin['bodies']):
            body.set_facecolor(self.color_list[i % len(self.color_list)])
            body.set_alpha(0.7)
            body.set_edgecolor('black')
            body.set_linewidth(0.5)
    
        for key in ['cmaxes', 'cmins', 'cbars', 'cmedians', 'cquantiles']:
            if key in violin:
                c_element = violin[key]
                c_element.set_linewidth(0.5)
                c_element.set_color('black')
    
        # X labels
        ax.set_xticks(clusters)
        ax.set_xticklabels([f'Cluster {k}' for k in clusters])
    
        # Y label
        ax.set_ylabel(convert_to_env_list_detail([env_plot])[0])
        
        # Format y-axis tick labels to 1 decimal places
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        
        # --------- Add significance lines + stars ----------
        # Use data range to place annotation levels above violins
        y_max = max(max(cluster_env_values[k]) for k in clusters)
        y_min = min(min(cluster_env_values[k]) for k in clusters)
        y_step = (y_max - y_min) * 0.07 if (y_max > y_min) else 0.1
    
        # Default order and vertical levels (like your old code)
        positions = [(1, 2), (1, 3), (2, 3)]
        y_levels = [y_max + 0.5 * y_step, y_max + 1.5 * y_step, y_max + 2.5 * y_step]
    
        # Draw lines and stars
        for i, ((pos1, pos2), y_pos) in enumerate(zip(positions, y_levels)):
            key = f'Cluster {pos1} vs Cluster {pos2}'
            if key in pairwise_results:
                p_value = pairwise_results[key][1]
                significance = get_significance_stars(p_value)
                ax.plot([pos1, pos2], [y_pos, y_pos], color='black', linewidth=0.5)
                if significance == 'n.s.':
                    ax.text((pos1 + pos2) / 2, y_pos, significance, ha='center', va='bottom', fontsize=5)
                else:
                    ax.text((pos1 + pos2) / 2, y_pos, significance, ha='center', va='center', fontsize=5)
    
        # Expand ylim to make room for annotations
        ax.set_ylim(bottom=min(y_min, ax.get_ylim()[0]), top=y_levels[-1] + 1.5 * y_step)
    
        # Save plot (same filename pattern as before, now includes 3 clusters)
        plot_output = os.path.join(self.plot_path_nichespace_clustering, f'{env_plot}_violin_species_median_clusters.pdf')
        plt.savefig(plot_output, dpi=500, transparent=True)
        plt.show()
    
        # Console summary
        print(f"Overall Kruskal–Wallis ({n_clusters} clusters): H={H:.3f}, p={p_kw:.3g}")
        print("Pairwise results (species-level medians):")
        for k, (stat, p) in pairwise_results.items():
            print(f"  {k}: H={stat:.3f}, p={p:.3g}")
    
        # ---------- Save stats to JSON ----------
        res_dict = {
            env_plot: {
                "data": {
                    str(k): {
                        "species": cluster_species_names[k],
                        "medians": [float(x) for x in cluster_env_values[k]]
                    } for k in clusters
                },
                "stats": {
                    "kruskal": {"H": float(H), "p": float(p_kw)},
                    "pairwise": {
                        f"{a}-{b}": {
                            "H": float(pairwise_results[f'Cluster {a} vs Cluster {b}'][0]),
                            "p": float(pairwise_results[f'Cluster {a} vs Cluster {b}'][1])
                        } for (a, b) in pairs
                    },
                    "n_per_cluster": {str(k): int(len(cluster_env_values[k])) for k in clusters}
                }
            }
        }
    
        # Append or create JSON file
        if os.path.exists(self.species_cluster_env_values_stats_path):
            with open(self.species_cluster_env_values_stats_path, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                    if not isinstance(existing, dict):
                        existing = {}
                except json.JSONDecodeError:
                    existing = {}
        else:
            existing = {}
    
        existing[env_plot] = res_dict[env_plot]
        self.species_cluster_env_values_stats = existing
    
        with open(self.species_cluster_env_values_stats_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    
        print(f"[Saved] Updated results for '{env_plot}' to: {self.species_cluster_env_values_stats_path}")


    
    def load_existing_files(self):
        """
        Check if various files created in previous runs exist.
        If found, load them into the class instance to speed up re-runs.
        """
        self.avg_elev = None
        self.df_attention = None
        self.df_env_corr = None
        self.df_env_pca = None
        self.pc_info = None
        self.bin_info = None
        self.extent_info = None
        self.df_grid = None
        self.df_spearman = None
        self.cluster_avg_nichespace = None
        self.cluster_labels = None
        self.df_spearman_ecogeo = None
        self.df_env_corr = None
        self.label_to_color = None
        self.df_nichespace_center_coordinate = None
        
        if os.path.exists(self.avg_elev_path):
            self.avg_elev = pd.read_csv(self.avg_elev_path)

        if os.path.exists(self.df_attention_path):
            self.df_attention = pd.read_csv(self.df_attention_path, index_col=0)

        if os.path.exists(self.df_env_pca_path):
            self.df_env_pca = feather.read_dataframe(self.df_env_pca_path)

        if os.path.exists(self.pc_info_path):
            with open(self.pc_info_path, 'r') as f:
                self.pc_info = yaml.safe_load(f)

        if os.path.exists(self.bin_info_path):
            with open(self.bin_info_path, 'r') as f:
                self.bin_info = yaml.safe_load(f)

        if os.path.exists(self.extent_info_path):
            with open(self.extent_info_path, 'r') as f:
                self.extent_info = yaml.safe_load(f)
            self.nichespace_extent = [
                self.extent_info[f'PC{self.x_pca:02d}_extent_min'],
                self.extent_info[f'PC{self.x_pca:02d}_extent_max'],
                self.extent_info[f'PC{self.y_pca:02d}_extent_min'],
                self.extent_info[f'PC{self.y_pca:02d}_extent_max']
            ]
            self.nichespace_cell_width = (self.nichespace_extent[1] - self.nichespace_extent[0]) / self.niche_rst_size
            self.nichespace_cell_height = (self.nichespace_extent[3] - self.nichespace_extent[2]) / self.niche_rst_size

        if os.path.exists(self.df_grid_path):
            self.df_grid = feather.read_dataframe(self.df_grid_path)

        if os.path.exists(self.df_spearman_path):
            self.df_spearman = pd.read_csv(self.df_spearman_path)

        if os.path.exists(self.cluster_avg_nichespace_path):
            with open(self.cluster_avg_nichespace_path, 'r') as f:
                self.cluster_avg_nichespace = yaml.safe_load(f)

        if os.path.exists(self.env_pca_loadings_path):
            self.env_pca_loadings = pd.read_csv(self.env_pca_loadings_path, index_col=0)

        if os.path.exists(self.df_nichespace_center_coordinate_path):
            self.df_nichespace_center_coordinate = pd.read_csv(self.df_nichespace_center_coordinate_path, index_col = 0)
            
        if os.path.exists(self.cluster_labels_path):
            with open(self.cluster_labels_path, 'r') as f:
                self.cluster_labels = yaml.load(f, Loader=yaml.FullLoader)

        if os.path.exists(self.df_spearman_ecogeo_path):
            self.df_spearman_ecogeo = pd.read_csv(self.df_spearman_ecogeo_path)

        if os.path.exists(self.niche_beta_params_path):
            with open(self.niche_beta_params_path, 'r') as f:
                self.niche_beta_params = json.load(f)

        if os.path.exists(self.geographical_beta_params_path):
            with open(self.geographical_beta_params_path, 'r') as f:
                self.geographical_beta_params = json.load(f)

        if os.path.exists(self.niche_beta_cluster_params_path):
            with open(self.niche_beta_cluster_params_path, 'r') as f:
                self.niche_beta_cluster_params = json.load(f)

        if os.path.exists(self.geographical_beta_cluster_params_path):
            with open(self.geographical_beta_cluster_params_path, 'r') as f:
                self.geographical_beta_cluster_params = json.load(f)

        if os.path.exists(self.df_env_corr_path):
            self.df_env_corr = pd.read_csv(self.df_env_corr_path, index_col=0)

        if os.path.exists(self.label_to_color_path):
            with open(self.label_to_color_path, 'r') as f:
                self.label_to_color = json.load(f)

        if os.path.exists(self.species_cluster_env_values_stats_path):
            with open(self.species_cluster_env_values_stats_path, 'r') as f:
                self.species_cluster_env_values_stats = json.load(f)

def create_folder(file_path):
    """
    Utility function to create directories if they do not already exist.
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def png_operation(rst):
    """
    Normalize an input 2D array to 0-255, convert to color map (Jet),
    and resize the output PNG to 300x300.
    """
    grid_normalized = cv2.normalize(rst, None, 0, 255, cv2.NORM_MINMAX)
    grid_uint8 = grid_normalized.astype(np.uint8)
    colored_image = cv2.applyColorMap(grid_uint8, cv2.COLORMAP_JET)
    new_size = (300, 300)
    resized_png = cv2.resize(colored_image, new_size, interpolation=cv2.INTER_CUBIC)
    return resized_png


def haversine(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine function to calculate great-circle distance.
    Returns distance in kilometers.
    """
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def mm2inch(*values):
    """
    Convert millimeters to inches (commonly used for figure sizing in matplotlib).
    """
    return [v / 25.4 for v in values]


def set_mpl_defaults():
    """
    Configure default plotting parameters for matplotlib.
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.size'] = 5
    mpl.rcParams['axes.labelsize'] = 7
    mpl.rcParams['xtick.labelsize'] = 5
    mpl.rcParams['ytick.labelsize'] = 5
    mpl.rcParams['legend.fontsize'] = 5
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['xtick.major.size'] = 2.5
    mpl.rcParams['ytick.major.size'] = 2.5
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['boxplot.boxprops.linewidth'] = 0.5
    mpl.rcParams['boxplot.medianprops.linewidth'] = 0.5
    mpl.rcParams['boxplot.whiskerprops.linewidth'] = 0.5
    mpl.rcParams['boxplot.capprops.linewidth'] = 0.5
    mpl.rcParams['boxplot.flierprops.markersize'] = 0.5
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['lines.markersize'] = 3
    mpl.rcParams['boxplot.medianprops.color'] = 'black'
    mpl.rcParams['hatch.linewidth'] = 0.5


def get_significance_stars(p_value):
    """
    Return significance stars based on p-value thresholds.
    """
    if p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    else:
        return 'n.s.'


def convert_to_env_list_detail(env_list_original):
    """
    Convert short environment variable names to more descriptive ones.
    """
    env_list_change = {
        'clt': 'Cloud area fraction',
        'hurs': 'Relative humidity',
        'pr': 'Precipitation',
        'rsds': 'Shortwave radiation',
        'sfcWind': 'Wind speed',
        'tas': 'Temperature',
        'EVI': 'EVI',
        'landcover_PC01': 'LandcoverPC1',
        'landcover_PC02': 'LandcoverPC2',
        'landcover_PC03': 'LandcoverPC3',
        'landcover_PC04': 'LandcoverPC4',
        'landcover_PC05': 'LandcoverPC5',
        'landcover_PC06': 'LandcoverPC6',
    }
    return [env_list_change[i] for i in env_list_original]


def reorder_df_attention(df_attention, env_order='LandcoverPC1'):
    """
    Reorder attention DataFrame by a specified environmental factor
    and then by the mean values. 
    """
    df_attention_order = df_attention.copy()
    df_attention_order.columns = convert_to_env_list_detail(df_attention_order.columns)
    df_attention_order = df_attention_order.sort_values(by=env_order, ascending=True)
    factor_order = df_attention_order.mean().sort_values(ascending=True).index
    df_attention_order = df_attention_order[factor_order]
    df_attention_order.index = [f"{sp.split('_')[0]} {sp.split('_')[1]}" for sp in list(df_attention_order.index)]
    return df_attention_order


def calculate_weighted_centroid(grid, extent):
    """
    Calculate the weighted centroid of a 2D raster grid using the pixel values.
    """
    rows, cols = grid.shape
    x_coords = np.linspace(extent[0], extent[1], cols)
    y_coords = np.linspace(extent[3], extent[2], rows)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    values = grid.flatten()
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    valid_mask = values > 0
    values = values[valid_mask]
    x_flat = x_flat[valid_mask]
    y_flat = y_flat[valid_mask]
    weighted_x = np.sum(values * x_flat) / np.sum(values)
    weighted_y = np.sum(values * y_flat) / np.sum(values)
    return weighted_x, weighted_y


def cosine_similarity_manual(vec1, vec2):
    """
    Manual cosine similarity calculation for two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product if norm_product != 0 else 0


def get_performance_stats(indicator_merged):
    """
    Extract performance metrics for both DeepSDM and MaxEnt,
    run paired t-tests, and gather results for plotting or summaries.
    """
    indicators = {
        'AUC': ('DeepSDM_AUC', 'MaxEnt_AUC'),
        'F1': ('DeepSDM_F1', 'MaxEnt_F1'),
        'TSS': ('DeepSDM_TSS', 'MaxEnt_TSS'),
        "Cohen's Kappa": ('DeepSDM_Kappa', 'MaxEnt_Kappa')
    }
    data_deepsdm = []
    data_maxent = []
    ticks = []
    significance_stars = []
    n = []
    t_stats = []
    p_values = []

    for indicator_name, (col_deepsdm, col_maxent) in indicators.items():
        indicator_filtered = indicator_merged[
            (indicator_merged[col_deepsdm] != -9999) & (indicator_merged[col_deepsdm].notna()) &
            (indicator_merged[col_maxent] != -9999) & (indicator_merged[col_maxent].notna())
        ]
        n.append(len(indicator_filtered))
        data_deepsdm.append(indicator_filtered[col_deepsdm].values)
        data_maxent.append(indicator_filtered[col_maxent].values)
        ticks.append(indicator_name)
        t_stat, p_value = ttest_rel(indicator_filtered[col_deepsdm], indicator_filtered[col_maxent])
        significance_stars.append(get_significance_stars(p_value))
        t_stats.append(t_stat)
        p_values.append(p_value)

    return data_deepsdm, data_maxent, ticks, significance_stars, n, t_stats, p_values


def logistic(z):
    """Logistic function commonly used in Beta regression link."""
    return 1 / (1 + np.exp(-z))


def lda_pcspace_importance(mu, y, load_df, use_cosine_only=False):
    
    lda = LinearDiscriminantAnalysis(solver='svd')
    lda.fit(mu, y)
    U = np.asarray(lda.scalings_)          # (2, q)
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    # Axis explanatory ratio (use average if not available)
    if getattr(lda, 'explained_variance_ratio_', None) is not None:
        w = np.asarray(lda.explained_variance_ratio_, float)
        w = w / w.sum()
    else:
        w = np.ones(U.shape[1]) / U.shape[1]

    # Normalize LDA axes to unit length
    Uhat = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)  # (2, q)

    # Extract variable loadings on the PC plane (p×2)
    L2 = load_df[['PC01', 'PC02']].to_numpy(dtype=float)         # (p, 2)

    # Calculate importance (entirely within the PC plane)
    if use_cosine_only:
        # Directional consistency: normalize variable vectors, then compute inner product with each LDA axis
        L2_unit = L2 / (np.linalg.norm(L2, axis=1, keepdims=True) + 1e-12)   # (p,2)
        P = L2_unit @ Uhat          # (p, q) = cos(theta)
        Imp_raw = (P**2) @ w        # sum_l w_l * cos^2
    else:
        # Projection energy including vector length: directly project onto LDA axes
        P = L2 @ Uhat               # (p, q) = projection length
        Imp_raw = (P**2) @ w        # sum_l w_l * projection^2

    # Normalize (so that the sum equals 1)
    Imp = Imp_raw / (Imp_raw.sum() + 1e-12)
    return Imp, Uhat, w, L2

    
# ==================================================================================
# Beta Regression Implementation
# ==================================================================================
u"""
Beta regression for modeling rates and proportions.
References
----------
Grün, Bettina, Ioannis Kosmidis, and Achim Zeileis. Extended beta regression
in R: Shaken, stirred, mixed, and partitioned. No. 2011-22. Working Papers in
Economics and Statistics, 2011.
Smithson, Michael, and Jay Verkuilen. "A better lemon squeezer?
Maximum-likelihood regression with beta-distributed dependent variables."
Psychological methods 11.1 (2006): 54.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gammaln as lgamma
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod.families import Binomial


class Logit(sm.families.links.Logit):
    """Logit transform that avoids overflow for large numbers."""
    def inverse(self, z):
        return 1 / (1. + np.exp(-z))


_init_example = """
    Beta regression with default of logit-link for exog and log-link
    for precision.
    >>> mod = Beta(endog, exog)
    >>> rslt = mod.fit()
    >>> print rslt.summary()
    We can also specify a formula and a specific structure and use the
    identity-link for phi.
    >>> from sm.families.links import identity
    >>> Z = patsy.dmatrix('~ temp', dat, return_type='dataframe')
    >>> mod = Beta.from_formula('iyield ~ C(batch, Treatment(10)) + temp',
    ...                         dat, Z=Z, link_phi=identity())
    In the case of proportion-data, we may think that the precision depends on
    the number of measurements. E.g for sequence data, on the number of
    sequence reads covering a site:
    >>> Z = patsy.dmatrix('~ coverage', df)
    >>> mod = Beta.from_formula('methylation ~ disease + age + gender + coverage', df, Z)
    >>> rslt = mod.fit()
"""


class Beta(GenericLikelihoodModel):
    """
    Beta Regression.
    This implementation uses `phi` as a precision parameter equal to
    `a + b` from the Beta parameters.
    """

    def __init__(self, endog, exog, Z=None, link=Logit(),
                 link_phi=sm.families.links.Log(), **kwds):
        assert np.all((0 < endog) & (endog < 1))
        if Z is None:
            extra_names = ['phi']
            Z = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['precision-%s' % zc for zc in (
                Z.columns if hasattr(Z, 'columns') else range(1, Z.shape[1] + 1))]
        kwds['extra_params_names'] = extra_names
        super(Beta, self).__init__(endog, exog, **kwds)
        self.link = link
        self.link_phi = link_phi
        self.Z = Z
        assert len(self.Z) == len(self.endog)

    def nloglikeobs(self, params):
        """Negative log-likelihood."""
        return -self._ll_br(self.endog, self.exog, self.Z, params)

    def fit(self, start_params=None, maxiter=100000, disp=False,
            method='bfgs', **kwds):
        """
        Fit the model via maximum likelihood.
        """
        if start_params is None:
            start_params = sm.GLM(self.endog, self.exog, family=Binomial()).fit(disp=False).params
            start_params = np.append(start_params, [0.5] * self.Z.shape[1])
        return super(Beta, self).fit(start_params=start_params,
                                     maxiter=maxiter,
                                     method=method, disp=disp, **kwds)

    def _ll_br(self, y, X, Z, params):
        """Log-likelihood function for Beta regression."""
        nz = self.Z.shape[1]
        Xparams = params[:-nz]
        Zparams = params[-nz:]
        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_phi.inverse(np.dot(Z, Zparams))

        if np.any(phi <= np.finfo(float).eps):
            return np.array(-np.inf)

        ll = lgamma(phi) - lgamma(mu * phi) - lgamma((1 - mu) * phi) \
            + (mu * phi - 1) * np.log(y) + (((1 - mu) * phi) - 1) \
            * np.log(1 - y)
        return ll