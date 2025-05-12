import numpy as np
import os
import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import rasterio
# from osgeo import gdal
# from osgeo import gdalconst
import cv2
from matplotlib import pyplot as plt
import re
from sklearn.decomposition import PCA
import yaml
import hashlib
import h5py
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, calculate_default_transform, Resampling
import warnings

class RasterHelper:
    
    def __init__(self):
        super().__init__()
        
        self.species_filter = None
        self.species_list = None
        
        self.no_data = -9999
        
        if not os.path.exists('./workspace'):
            os.makedirs('./workspace')
        
        # new extent_binary which based on all environmental layers extent
        self.extent_binary_intersection = None
        
    def res_rounder(self, a, second_round=None):
        if second_round is None:
            return round(a, self.num_digits_after_decimal)
        else:
            return round(round(a, self.num_digits_after_decimal), second_round)

    def time_span(self, sourcedate):
        delta = relativedelta(years = self.year_span, months = self.month_span, days = self.day_span)
        return sourcedate + delta

    def time_step(self, sourcedate):
        delta = relativedelta(years = self.year_step, months = self.month_step, days = self.day_step)
        return sourcedate + delta
        
    def set_temporal_conf(self, temporal_conf):
        self.date_start = datetime.strptime(temporal_conf.date_start, '%Y-%m-%d')
        self.date_end = datetime.strptime(temporal_conf.date_end, '%Y-%m-%d')
        self.year_span = 0
        self.month_span = temporal_conf.month_span
        self.day_span = 0
        self.year_step = 0
        self.month_step = temporal_conf.month_step
        self.day_step = 0
        self.day_first = 0
        self.day_last = (self.date_end - self.date_start).days

    def create_extent_binary_from_env_layer(self, input_env, spatial_conf):
        self.spatial_conf = spatial_conf
        # Open source environment layer
        with rasterio.open(input_env) as src:
            src_crs = src.crs
            src_transform = src.transform

        # Determine output resolution and grid
        out_res = abs(spatial_conf.out_res) if spatial_conf.out_res else abs(src_transform.a)
        spatial_conf.out_res = out_res
        parts = str(out_res).split('.')
        self.num_digits_after_decimal = min(len(parts[1]) if len(parts)>1 else 0, 6)
        spatial_conf.x_num_cells = int(spatial_conf.num_of_grid_x * spatial_conf.grid_size)
        spatial_conf.y_num_cells = int(spatial_conf.num_of_grid_y * spatial_conf.grid_size)
        spatial_conf.x_end = self.res_rounder(spatial_conf.x_start + spatial_conf.x_num_cells * out_res)
        spatial_conf.y_end = self.res_rounder(spatial_conf.y_start + spatial_conf.y_num_cells * out_res)

        dst_transform = Affine(out_res, 0, spatial_conf.x_start,
                               0, -out_res, spatial_conf.y_end)
        profile = {
            'driver': 'GTiff',
            'height': spatial_conf.y_num_cells,
            'width': spatial_conf.x_num_cells,
            'count': 1,
            'dtype': 'float32',
            'crs': src_crs,
            'transform': dst_transform,
            'nodata': self.no_data
        }
        med_tif = './workspace/extent_env_example.tif'
        # Reproject to uniform grid
        with rasterio.open(input_env) as src, rasterio.open(med_tif, 'w', **profile) as dst:
            reproject(
                source=src.read(1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear
            )
        # Binarize
        with rasterio.open(med_tif) as src:
            arr = src.read(1)
            bin_arr = arr.astype('int16')
            bin_profile = src.profile.copy()
            bin_profile.update({'dtype':'int16'})
            out_tif = './workspace/extent_binary.tif'
            with rasterio.open(out_tif, 'w', **bin_profile) as dst:
                dst.write(bin_arr, 1)
        return spatial_conf

    def raw_to_medium_(self, raw_env_tif, medium_env_tif):
        """
        Reproject a raw environmental TIFF to match the grid of
        './workspace/extent_binary.tif' and save the result as 'medium_env_tif'.
        After reprojection, update the extent intersection mask.
        """
        # 1. Open the reference binary extent TIFF to get target grid properties
        ref_path = './workspace/extent_binary.tif'
        with rasterio.open(ref_path) as ref:
            dst_transform = ref.transform          # target affine transform
            dst_crs       = ref.crs               # target coordinate reference system
            dst_width     = ref.width             # target width in pixels
            dst_height    = ref.height            # target height in pixels
            profile       = ref.profile.copy()    # copy base file profile for writing
    
        # 2. Read the source raw environmental TIFF
        with rasterio.open(raw_env_tif) as src:
            src_data      = src.read(1)           # read the first band
            src_transform = src.transform         # source affine transform
            src_nodata    = src.nodata
            # if the source CRS is missing, warn and assume reference CRS
            if src.crs is None:
                warnings.warn(
                    f"{raw_env_tif} has no CRS—assuming {dst_crs}",
                    UserWarning
                )
                src_crs = dst_crs
            else:
                src_crs = src.crs
                
        # 3. Update the profile for the output file
        profile.update({
            'driver': 'GTiff',                    # output format
            'height': dst_height,                 # match reference height
            'width': dst_width,                   # match reference width
            'count': 1,                           # single band
            'dtype': 'float32',                   # data type
            'crs': dst_crs,                       # output CRS same as reference
            'transform': dst_transform,           # output transform same as reference
            'nodata': self.no_data                # nodata value
        })
    
        # 4. Perform reprojection and write to destination TIFF
        with rasterio.open(medium_env_tif, 'w', **profile) as dst:
            reproject(
                source=src_data,
                destination=rasterio.band(dst, 1),
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,     # use bilinear interpolation
                src_nodata=src_nodata, 
                dst_nodata=self.no_data
            )
    
        # 5. Update the intersection mask with the newly aligned layer
        self.intersect_extents(medium_env_tif)
        
        
        

    def raw_to_medium_agg_(self, doy_to_month_tifs):
        """
        Aggregate multiple raw daily TIFFs into a single monthly TIFF by:
        1. Reprojecting each raw TIFF to the reference grid.
        2. Stacking the reprojected arrays.
        3. Computing the mean value cell-wise.
        4. Writing out the averaged TIFF and updating the extent intersection.
        """
        # 1. Collect raw input paths and expected medium output path
        raw_env_tifs = [item['raw'] for item in doy_to_month_tifs]
        medium_env_out = doy_to_month_tifs[0]['medium']
        assert len(set([item['medium'] for item in doy_to_month_tifs])) == 1, \
            "All medium outputs must be the same path"
    
        # 2. Open the reference extent binary to get grid properties
        ref_path = './workspace/extent_binary.tif'
        with rasterio.open(ref_path) as ref:
            dst_transform = ref.transform          # target affine transform
            dst_crs       = ref.crs               # target CRS
            dst_width     = ref.width             # target width
            dst_height    = ref.height            # target height
            profile       = ref.profile.copy()    # base profile for outputs
    
        # 3. Prepare a list to hold reprojected arrays
        reprojected_arrs = []
    
        # 4. Reproject each raw TIFF to the reference grid
        for raw_tif in raw_env_tifs:
            with rasterio.open(raw_tif) as src:
                src_data      = src.read(1)            # read source band
                src_transform = src.transform          # source affine
                src_crs       = src.crs               # source CRS
    
            # Allocate array filled with NaN for reprojection
            dest_array = np.full((dst_height, dst_width), np.nan, dtype='float32')
    
            # Perform reprojection into the NumPy array
            reproject(
                source=src_data,
                destination=dest_array,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan
            )
    
            reprojected_arrs.append(dest_array)
    
        # 5. Compute the cell-wise mean, ignoring NaNs
        stacked = np.stack(reprojected_arrs, axis=0)
        avg_array = np.nanmean(stacked, axis=0)
    
        # 6. Replace NaNs with the designated no-data value
        avg_array = np.where(np.isnan(avg_array), self.no_data, avg_array)
    
        # 7. Update output profile for writing
        profile.update({
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': 1,
            'nodata': self.no_data,
            'transform': dst_transform,
            'crs': dst_crs,
            'height': dst_height,
            'width': dst_width
        })
    
        # 8. Write the averaged array to the medium TIFF
        with rasterio.open(medium_env_out, 'w', **profile) as dst:
            dst.write(avg_array.astype('float32'), 1)
    
        # 9. Update the global extent intersection mask
        self.intersect_extents(medium_env_out)
        
        
    def build_env_(self, env, conf):
        pattern = conf['filename_template'].replace('[YEAR]', r'(?P<year>\d{4})').replace('[MONTH]', r'(?P<month>\d{1,2})').replace('[DOY]', r'(?P<doy>\d{3})') + '$'
        # Convert template to regex pattern
        regex = re.compile(pattern)

        medium_env_dir = 'medium'

        if env not in self.env_medium_list:
            self.env_medium_list[env] = {}

        try:
            assert(len(conf['year_coverages']) == 2)
            year_coverages_start, year_coverages_end = conf['year_coverages']
            if year_coverages_start is None:
                year_coverages_start = self.date_start.year
            if year_coverages_end is None:
                year_coverages_end = self.date_end.year
        except:
            year_coverages_start = self.date_start.year
            year_coverages_end = self.date_end.year

        self.doy_to_month_tifs = None
            
        for fname in os.listdir(conf['raw_env_dir']):

            year = None
            month = None
            doy = None

            every_year = False
            every_month = False
            with_doy = False
            
            # Filter files based on the regex pattern
            matched = regex.search(fname)
            if matched:
                try:
                    year = matched.group('year')
                    every_year = True
                except:
                    pass

                try:
                    month = matched.group('month')
                    every_month = True
                except:
                    pass

                try:
                    doy = matched.group('doy')
                    with_doy = True
                except:
                    pass
                
                # You can either set month or day of year, but not both
                assert(not(every_month and with_doy))
                
                if every_year and every_month:
                    # this part has not been tested yet
                    y = int(year)
                    m = int(month)
                    
                    env_out = conf['env_out_template'].replace('[YEAR]', f'{y:04d}').replace('[MONTH]', f'{m:02d}')
                    full_out_path = os.path.join(medium_env_dir, env)
                    if not os.path.isdir(full_out_path):
                        os.makedirs(full_out_path)
                    self.raw_to_medium_(f'{conf["raw_env_dir"]}/{fname}', f'{full_out_path}/{env_out}')
                    # fill no shit, missing data means missing data
                    # representation of yyyy-mm-dd
                    y_m_d = datetime.strftime(datetime.strptime(f'{y}-{m}', '%Y-%m'), '%Y-%m-%d')
                    self.env_medium_list[env][y_m_d] = f'{full_out_path}/{env_out}'
                    # print(env_out)

                elif every_month:
                    m = int(month)
                    env_out = conf['env_out_template'].replace('[MONTH]', f'{m:02d}')
                    full_out_path = os.path.join(medium_env_dir, env)
                    if not os.path.isdir(full_out_path):
                        os.makedirs(full_out_path)
                    self.raw_to_medium_(f'{conf["raw_env_dir"]}/{fname}', f'{full_out_path}/{env_out}')
                    # fill year
                    for y in range(year_coverages_start, year_coverages_end + 1):
                        # representation of yyyy-mm-dd
                        y_m_d = datetime.strftime(datetime.strptime(f'{y}-{m}', '%Y-%m'), '%Y-%m-%d')
                        self.env_medium_list[env][y_m_d] = f'{full_out_path}/{env_out}'

                elif every_year and not with_doy:
                    y = int(year)
                    env_out = conf['env_out_template'].replace('[YEAR]', f'{y:04d}')
                    full_out_path = os.path.join(medium_env_dir, env)
                    if not os.path.isdir(full_out_path):
                        os.makedirs(full_out_path)
                    self.raw_to_medium_(f'{conf["raw_env_dir"]}/{fname}', f'{full_out_path}/{env_out}')
                    # fill month
                    for m in range(1, 13):
                        # representation of yyyy-mm-dd
                        y_m_d = datetime.strftime(datetime.strptime(f'{y}-{m}', '%Y-%m'), '%Y-%m-%d')
                        self.env_medium_list[env][y_m_d] = f'{full_out_path}/{env_out}'

                elif every_year and with_doy:
                    # this part has not been tested yet
                    # need aggregation
                    y = int(year)
                    doy = int(doy)
                    date_ = datetime.strptime(f'{y} {doy}', '%Y %j')
                    m = date_.month
                    
                    if self.doy_to_month_tifs is None:
                        self.doy_to_month_tifs = dict()
                    
                    y_m_d = datetime.strftime(datetime.strptime(f'{y}-{m}', '%Y-%m'), '%Y-%m-%d')
                    if y_m_d not in self.doy_to_month_tifs:
                        self.doy_to_month_tifs[y_m_d] = []
                        
                    env_out = conf['env_out_template'].replace('[YEAR]', f'{y:04d}').replace('[MONTH]', f'{m:02d}')
                    full_out_path = os.path.join(medium_env_dir, env)
                    if not os.path.isdir(full_out_path):
                        os.makedirs(full_out_path)

                    self.doy_to_month_tifs[y_m_d].append(dict(
                        raw = f'{conf["raw_env_dir"]}/{fname}',
                        medium = f'{full_out_path}/{env_out}'
                    ))
                    # self.raw_to_medium_(f'{conf["raw_env_dir"]}/{fname}', f'{full_out_path}/{env_out}')
                    # # fill month
                    # for m in range(1, 13):
                    #     self.env_medium_list[env][f'{y:04d}-{m:02d}'] = f'{full_out_path}/{env_out}'
                    # print(env_out)

                else:
                    env_out = conf['env_out_template']
                    full_out_path = os.path.join(medium_env_dir, env)
                    if not os.path.isdir(full_out_path):
                        os.makedirs(full_out_path)
                    self.raw_to_medium_(f'{conf["raw_env_dir"]}/{fname}', f'{full_out_path}/{env_out}')
                    # fill both year and month
                    for y in range(year_coverages_start, year_coverages_end + 1):
                        for m in range(1, 13):
                            # representation of yyyy-mm-dd
                            y_m_d = datetime.strftime(datetime.strptime(f'{y}-{m}', '%Y-%m'), '%Y-%m-%d')
                            self.env_medium_list[env][y_m_d] = f'{full_out_path}/{env_out}'
    #                 print(env_out)
        if self.doy_to_month_tifs is not None:
            self.doy_to_month_tifs = dict(sorted(self.doy_to_month_tifs.items()))
            for y_m_d in self.doy_to_month_tifs:
                self.raw_to_medium_agg_(self.doy_to_month_tifs[y_m_d])
                self.env_medium_list[env][y_m_d] = self.doy_to_month_tifs[y_m_d][0]['medium'] #f'{full_out_path}/{env_out}'

        self.env_medium_list[env] = dict(sorted(self.env_medium_list[env].items()))
        
        
    def raw_to_medium(self, env_raw_conf):
        """
        Process raw environment configurations, build medium layers,
        then apply flood-fill to finalize extent intersection mask.
        """
        # 1. Build all medium environment layers
        self.env_medium_list = {}
        for env, confs in env_raw_conf.items():
            for conf in confs:
                print(conf)
                self.build_env_(env, conf)

        # 2. Flood-fill the extent intersection mask using seed at (1559, 0)
        # Convert binary mask (0/1) to 0/255 range
        mask_orig = (self.extent_binary_intersection.astype(np.uint8) * 255)
        # Copy mask for flood-fill operation
        filled = mask_orig.copy()
        height, width = filled.shape
        fill_mask = np.zeros((height + 2, width + 2), np.uint8)
        # Perform flood-fill from the specified seed point
        cv2.floodFill(filled, fill_mask, (1559, 0), 255)
        # Invert flood-filled result to get non-connected background
        inv_filled = cv2.bitwise_not(filled)
        # Combine original mask and inverted flood-fill to restore full mask
        combined = cv2.bitwise_or(mask_orig, inv_filled)
        # Convert to final binary mask (0/1)
        final_mask = np.sign(combined).astype(np.int32)

        # 3. Write the final mask back to extent_binary.tif using rasterio
        tif_path = './workspace/extent_binary.tif'
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()
        profile.update(dtype='int32', nodata=self.no_data)
        with rasterio.open(tif_path, 'w', **profile) as dst:
            dst.write(final_mask, 1)
        
    def random_split_train_val(self, train_ratio=0.7):
        spatial_conf = self.spatial_conf
        total_grids = spatial_conf.num_of_grid_y * spatial_conf.num_of_grid_x
        num_train_grids = int(np.round(total_grids * train_ratio))
        
        train_val_partitions = np.where(np.random.uniform(size=(spatial_conf.num_of_grid_y, spatial_conf.num_of_grid_x)) >= train_ratio, 0, 1)
        
        while train_val_partitions.sum() != num_train_grids:
            train_val_partitions = np.where(np.random.uniform(size=train_val_partitions.shape)>=train_ratio, 0, 1).astype(np.uint8)
            
        np.savetxt('./workspace/partition.txt', train_val_partitions, fmt='%i', delimiter=',')
        print('Partition saved at ./workspace/partition.txt.')
        
    def random_split_train_val_within_extent_bin(self, train_ratio=0.7):
        spatial_conf = self.spatial_conf
        extent_bin_partitions = self.convert_extent_binary_to_extent_partition()
        
        num_train_grids = int(np.round(extent_bin_partitions.sum() * train_ratio))
        train_val_partitions = np.where(np.random.uniform(size=(spatial_conf.num_of_grid_y, spatial_conf.num_of_grid_x)) >= train_ratio, 0, 1)
        while (train_val_partitions*extent_bin_partitions).sum() != num_train_grids:
            train_val_partitions = np.where(np.random.uniform(size=train_val_partitions.shape)>=train_ratio, 0, 1).astype(np.uint8)
        np.savetxt('./workspace/partition.txt', train_val_partitions, fmt='%i', delimiter=',')

    def convert_extent_binary_to_extent_partition(self):
        spatial_conf = self.spatial_conf
        with rasterio.open('./workspace/extent_binary.tif', 'r') as f:
            extent_bin_array = f.read(1)
        extent_bin_partitions = np.zeros((spatial_conf.num_of_grid_y, spatial_conf.num_of_grid_x))
        for i_row in range(spatial_conf.num_of_grid_y):
            for i_col in range(spatial_conf.num_of_grid_x):
                subplot = extent_bin_array[i_row*spatial_conf.grid_size:(i_row+1)*spatial_conf.grid_size, i_col*spatial_conf.grid_size:(i_col+1)*spatial_conf.grid_size]
                if subplot.sum() > 0:
                    extent_bin_partitions[i_row, i_col] = 1
        np.savetxt('./workspace/extent_partition.txt', extent_bin_partitions, fmt='%i', delimiter=',')
        return extent_bin_partitions
        
    def view_train_val_splits(self, partition_file='./workspace/partition.txt'):
        spatial_conf = self.spatial_conf
        with rasterio.open('./workspace/extent_binary.tif') as src:
            ext_bin_array = src.read(1)
        train_val_partitions = np.loadtxt(partition_file, delimiter=',')
        train_val_mask = cv2.resize(train_val_partitions, (spatial_conf.num_of_grid_x * spatial_conf.grid_size, spatial_conf.num_of_grid_y * spatial_conf.grid_size), interpolation=cv2.INTER_NEAREST)
        plt.imshow(ext_bin_array * (train_val_mask.astype(float) + .5) / 2)
        plt.show()


    def avg_and_mask_env_timespan(self):
        env_info = dict(
            info = dict(),
            dir_base = './'
        )        
        date_range = pd.date_range(start=self.date_start, end=self.date_end, freq='MS')
        y_m_combs = np.array([[date.year, date.month] for date in date_range])

        with rasterio.open('./workspace/extent_binary.tif') as extent_binary_raster:
            mask = extent_binary_raster.read(1)

        for env in self.env_medium_list:
            print(env)
            if env not in env_info['info']:
                env_info['info'][env] = dict()

            env_out_path = f'workspace/raster_data/env_aligned_timespan_avg/{env}'
            if not os.path.isdir(env_out_path):
                os.makedirs(env_out_path)

            # get env files inventory
            env_srcs = []
            for i in range(y_m_combs.shape[0]):
                y0, m0 = y_m_combs[i]
                y0_m0_d0 = datetime.strftime(datetime.strptime(f'{y0}-{m0}', '%Y-%m'), '%Y-%m-%d')
                try:
                    env_srcs.append(self.env_medium_list[env][y0_m0_d0])
                except:
                    print(f"******* Warning. Missing data of env:{env} on {y0_m0_d0}.")
                    self.env_medium_list[env][y0_m0_d0] = 'Not Available'
                    env_srcs.append(self.env_medium_list[env][y0_m0_d0])
                    
            env_unique_srcs = np.unique(env_srcs)

            env_collection_of_spans = np.empty(0)
#             for i in range(y_m_combs.shape[0]):
            i = 0
            while i < y_m_combs.shape[0]:
                y0, m0 = y_m_combs[i]
                y0_m0_d0 = datetime.strftime(datetime.strptime(f'{y0}-{m0}', '%Y-%m'), '%Y-%m-%d')
                env_arrs = []
                env_local_srcs = []
                env_local_src_ids = []
                y_m_d_list = []
                for y, m in y_m_combs[i:(i+self.month_span)]:
                    y_m_d = datetime.strftime(datetime.strptime(f'{y}-{m}', '%Y-%m'), '%Y-%m-%d')
                    y_m_d_list.append(y_m_d)
                    if self.env_medium_list[env][y_m_d] != 'Not Available':
                        with rasterio.open(self.env_medium_list[env][y_m_d]) as env_raster:
                            env_local_srcs.append(self.env_medium_list[env][y_m_d])
                            env_local_src_ids.append(np.where(env_unique_srcs==self.env_medium_list[env][y_m_d])[0][0])
                            env_arr_ = env_raster.read(1)
                            env_arr_ = np.where(env_arr_==self.no_data, np.nan, env_arr_)
                            env_arrs.append(env_arr_)
                            env_crs = env_raster.crs
                            env_transform = env_raster.transform
                    else:
                        print(f"******* Warning. Data of env: {env} on {y_m_d} is not available.")

                if len(env_arrs) == 0:
                    print(f"******* Error. Data missing on full span. ({'.'.join(y_m_d_list)})")
                    assert(len(env_arrs)>0)
                    assert(len(env_arrs) == len(env_local_srcs))
                    assert(len(env_local_src_ids) == len(env_local_srcs))

                fname_base = '.'.join(os.path.basename(self.env_medium_list[env][y0_m0_d0]).split('.')[:-1])
                if fname_base == '':
                    print(env, y0_m0_d0, self.env_medium_list[env][y0_m0_d0])
                    fname_base = f'{env}_{y0_m0_d0}_src_missing'

                srcs_, cnts_ = np.unique(env_local_src_ids, return_counts = True)
                postfixs = []

                if len(srcs_) > 1:
                    for src_i in range(len(srcs_)):
                        postfixs.append(f'{srcs_[src_i]}.{cnts_[src_i]}')
                    fname = f'{fname_base}_srcs{"and".join(postfixs)}_timespan_avg.tif'
                    
                    # operation if fname is too long 
                    if len(fname) > 200:
                        fname_hash = hashlib.md5("and".join(postfixs).encode()).hexdigest()
                        fname = f'{fname_base}_srcs_{fname_hash}_timespan_avg.tif'
                else:
                    fname = f'{fname_base}.tif'

                path_name = f'{env_out_path}/{fname}'

                if y0_m0_d0 not in env_info['info'][env]:
                    env_info['info'][env][y0_m0_d0] = dict()

                env_info['info'][env][y0_m0_d0]['tif_span_avg'] = path_name
                env_info['info'][env][y0_m0_d0]['tif_sources'] = list(env_unique_srcs[srcs_])
                env_arr_span_avg = np.nanmean(np.stack(env_arrs), axis=0)
                
                env_cell_avg = np.nanmean(np.stack(env_arrs))
                
                env_arr_span_avg = np.where(np.isnan(env_arr_span_avg), self.no_data, env_arr_span_avg)
                
                # fill in the missing cells with cell avg
                env_arr_span_avg = np.where((mask==1)&(env_arr_span_avg==self.no_data), env_cell_avg, env_arr_span_avg)
                env_arr_span_avg = np.where((mask==1), env_arr_span_avg, self.no_data)

                with rasterio.open(
                    os.path.join(path_name), 
                    'w',
                    height = env_arr_span_avg.shape[0],
                    width = env_arr_span_avg.shape[1], 
                    count = 1, 
                    nodata = self.no_data, 
                    crs = env_crs, 
                    dtype = rasterio.float32,
                    transform = env_transform
                ) as tif_out:
                    tif_out.write(env_arr_span_avg, 1)

                env_collection_of_spans = np.concatenate((env_collection_of_spans, env_arr_span_avg[mask == 1]))
                
                i += self.month_step

            env_info['info'][env]['mean'] = np.mean(env_collection_of_spans)
            env_info['info'][env]['sd'] = np.std(env_collection_of_spans)

        with open('./workspace/env_information.json', 'w') as f:
            json.dump(env_info, f)        

#         return env_info


    #########################################
        
    def create_k_info(self):
        
        
        # create files of the 'no_k' situation 
        self.nok_out = './workspace/raster_data/k_nok'
        if not os.path.exists(self.nok_out):
            os.makedirs(self.nok_out)
        nok_info = dict()
        nok_info['dir_base'] = self.nok_out
        nok_info['file_name'] = dict() 
        
        # create files of regular k situation        
        self.k_out = './workspace/raster_data/k'
        if not os.path.exists(self.k_out):
            os.makedirs(self.k_out)        
        k_info = dict()
        k_info['dir_base'] = self.k_out
        k_info['file_name'] = dict()

        with rasterio.open('./workspace/extent_binary.tif') as raster_:
            extent_transform = raster_.transform
            extent_binary = raster_.read(1)
            extent_crs = raster_.crs

        xres = abs(extent_transform[0])
        yres = abs(extent_transform[4])

        if self.species_filter is None:
            self.species_filter = pd.read_csv('./workspace/species_data/occurrence_data/species_occurrence_filter.csv')
            self.species_list = np.unique(self.species_filter.species)

        species_filter = self.species_filter.copy()
        species_filter['week'] = species_filter.daysincebegin // 7

        date_target_start = self.date_start
        date_target_end = self.time_span(date_target_start)
        day_target_start = (date_target_start - self.date_start).days
        day_target_end = (date_target_end - self.date_start).days


        while date_target_start <= self.date_end:
            species_filter_day = species_filter[(species_filter.daysincebegin < day_target_end) & (species_filter.daysincebegin >= day_target_start)]
            week_list = list(set(species_filter_day.week))

            rst_time_span = np.zeros([extent_binary.shape[0], extent_binary.shape[1]])
            
            for week in week_list:
                rst_week = np.zeros([extent_binary.shape[0], extent_binary.shape[1]])
                species_filter_week = species_filter_day[species_filter_day.week == week]
                for i, row in species_filter_week.iterrows():
                    nlong = int(self.res_rounder(abs(row['decimalLongitude'] - self.spatial_conf.x_start) / xres))
                    nlat = int(self.res_rounder(abs(self.spatial_conf.y_end - row['decimalLatitude']) / yres))
                    rst_week[nlat, nlong] = 1
                rst_time_span = rst_time_span + rst_week

            rst_result = np.where(extent_binary == 0, self.no_data, rst_time_span / len(week_list))
            date_target_log = f'{date_target_start.year}-{date_target_start.month:02d}-{date_target_start.day:02d}'
            with rasterio.open(
                os.path.join(self.k_out, f'k_{date_target_log}.tif'),
                'w', 
                height = extent_binary.shape[0], 
                width = extent_binary.shape[1], 
                count = 1, 
                nodata = self.no_data,
                crs = extent_crs, 
                dtype = rasterio.float32, 
                transform = extent_transform
            ) as img:
                img.write(rst_result, 1)

            k_info['file_name'][date_target_log] = f'k_{date_target_log}.tif'
            nok_info['file_name'][date_target_log] = 'nok.tif'

            date_target_start = self.time_step(date_target_start)
            date_target_end = self.time_span(date_target_start)
            day_target_start = (date_target_start - self.date_start).days
            day_target_end = (date_target_end - self.date_start).days

        # tifs without k situation
        with rasterio.open(
            'nok.tif',
            'w', 
            height = extent_binary.shape[0], 
            width = extent_binary.shape[1], 
            count = 1, 
            nodata = self.no_data,
            crs = extent_crs, 
            dtype = rasterio.float32, 
            transform = extent_transform
        ) as img:
            img.write(np.zeros([extent_binary.shape[0], extent_binary.shape[1]]), 1)              
            
            
        with open('./workspace/k_information.json', 'w') as f:
            json.dump(k_info, f)
        with open('./workspace/k_information_nok.json', 'w') as f:
            json.dump(nok_info, f)
            
    def create_species_raster(self, species_to_createraster = 'all'):

        if self.species_filter is None:
            self.species_filter = pd.read_csv('./workspace/species_data/occurrence_data/species_occurrence_filter.csv')
            self.species_list = np.unique(self.species_filter.species)
            
        species_filter = self.species_filter
        
        
        # 20250505
        # add a parameter to determine which species will be used to create occurrence rasters
        species_list = self.species_list
        if species_to_createraster != 'all':
            species_list = species_to_createraster
        
        self.sp_raster_out = './workspace/raster_data/species_occurrence'
        if not os.path.exists(self.sp_raster_out):
            os.makedirs(self.sp_raster_out)
        
        with rasterio.open('./workspace/extent_binary.tif') as raster_:
            extent_crs = raster_.crs
            extent_binary = raster_.read(1)
            extent_transform = raster_.transform
            
        xres = abs(extent_transform[0])
        yres = abs(extent_transform[4])

        sp_inf = dict()
        sp_inf['dir_base'] = self.sp_raster_out
        file_name = dict()
        
        for sp in species_list:

            # species information json
            file_name[sp] = dict()

            # filter data by species
            data_s = species_filter[species_filter['species'].values == sp]

            # date operation
            date_s = self.date_start
            t_s = self.day_first
            date_e = self.time_span(date_s)
            t_e = (date_e - date_s).days
            
            h5file_name = os.path.join(self.sp_raster_out, f'{sp}.h5')
            file_name[sp]['h5file_name'] = f'{sp}.h5'
            file_name[sp]['h5file_dataset_name'] = dict()
            with h5py.File(h5file_name, 'w') as h5f:
                
                while t_s <= self.day_last:

                    data_t = data_s[(data_s['daysincebegin'].values < t_e) & (data_s['daysincebegin'].values >= t_s)]
                    rst = np.zeros([extent_binary.shape[0], extent_binary.shape[1]])
                    for i, row in data_t.iterrows():
                        nlong = int(self.res_rounder(abs(row['decimalLongitude'] - self.spatial_conf.x_start) / xres))
                        nlat = int(self.res_rounder(abs(self.spatial_conf.y_end - row['decimalLatitude']) / yres))
                        try:
                            rst[nlat, nlong] = 1
                        except:
                            print("Error: Occurrence Point of species: {sp} out of bounds.")
                            print(f"Boundary: {extent_binary.shape}")
                            print(f'{row["decimalLatitude"]}, {row["decimalLongitude"]} to {nlat}, {nlong}')

                    date_span = f"{date_s.strftime('%Y')}-{date_s.strftime('%m')}-{date_s.strftime('%d')}"

#                     with rasterio.open(
#                         f"{self.sp_raster_out}/{sp_data_span_tif}", 
#                         'w', 
#                         height = extent_binary.shape[0], 
#                         width = extent_binary.shape[1],
#                         count = 1, 
#                         nodata = self.no_data, 
#                         crs = extent_crs, 
#                         dtype = rasterio.int16, 
#                         transform = extent_transform
#                     ) as dst:
#                         dst.write(rst * extent_binary, 1)
                    h5f.create_dataset(date_span, data = rst * extent_binary, compression = 'gzip', dtype = np.int16)
                    
                    file_name[sp]['h5file_dataset_name'][date_span] = date_span

                    date_s = self.time_step(date_s)
                    t_s = (date_s - self.date_start).days
                    date_e = self.time_span(date_s)
                    t_e = (date_e - self.date_start).days

        sp_inf['file_name'] = file_name
        with open('./workspace/species_information.json', 'w') as f:
            json.dump(sp_inf, f)
            
        self.sp_inf = sp_inf

    def raw_to_medium_CCI(self, CCI_conf):
        self.CCI_PCA_year = []
        for env in CCI_conf:
            self.CCI_value = np.empty((len(CCI_conf[env][0]['unique_class']), ), dtype = object)
            for conf in CCI_conf[env]:
                print(conf)
                self.build_env_CCI_(env, conf)
        if conf['PCA'] != None:
            self.CCI_PCA(CCI_conf)
            
    def build_env_CCI_(self, env, conf):
        pattern = conf['filename_template'].replace('[YEAR]', r'(?P<year>\d{4})') + '$'
        # Convert template to regex pattern
        regex = re.compile(pattern)

        medium_env_dir = 'medium'
        if conf['PCA'] == None:
            for value in conf['unique_class']:
                path = os.path.join(medium_env_dir, f'{env}_type{value:03d}')
                if not os.path.isdir(path):
                    os.makedirs(path)  
            
        for fname in os.listdir(conf['raw_env_dir']):
            every_year = False
            # Filter files based on the regex pattern
            matched = regex.search(fname)
            if matched:
                try:
                    year = matched.group('year')
                    every_year = True
                except:
                    pass
                if every_year:
                    y = int(year)
                    env_out = conf['env_out_template'].replace('[YEAR]', f'{y:04d}')
                    self.CCI_PCA_year.append(y)
                    self.raw_to_medium_CCI_(f'{conf["raw_env_dir"]}/{fname}', f'{medium_env_dir}/{env_out}', conf)


    def raw_to_medium_CCI_(self, raw_env_nc, medium_env_tif, conf):
        """
        Read a NetCDF environmental layer via Rasterio, reproject it to the reference grid,
        then either export binary class rasters (if no PCA) or accumulate values for PCA.
        """
        # 1. Load reference grid to get transform, CRS, and profile
        ref_path = './workspace/extent_binary.tif'
        with rasterio.open(ref_path) as ref:
            dst_transform = ref.transform       # target affine transform
            dst_crs       = ref.crs            # target coordinate reference system
            dst_width     = ref.width          # target width in pixels
            dst_height    = ref.height         # target height in pixels
            base_profile  = ref.profile.copy() # template profile for outputs
    
        # 2. Open the NetCDF subdataset
        src_path = f'NETCDF:{raw_env_nc}:{conf["layer_name"]}'
        with rasterio.open(src_path) as src:
            src_data      = src.read(1)         # read the data array
            src_transform = src.transform       # source affine transform
            src_nodata = src.nodata
            # if the source CRS is missing, warn and assume reference CRS
            if src.crs is None:
                warnings.warn(
                    f"{src_path} has no CRS—assuming {dst_crs}",
                    UserWarning
                )
                src_crs = dst_crs
            else:
                src_crs = src.crs
                
        # 3. Allocate an array for reprojection output
        dest_array = np.full((dst_height, dst_width), self.no_data, dtype='int32')
    
        # 4. Reproject using nearest-neighbor for categorical data
        reproject(
            source=src_data,
            destination=dest_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=src_nodata,
            dst_nodata=self.no_data
        )
    
        # 5. If no PCA is requested, export one binary TIFF per class
        if conf['PCA'] is None:
            profile = base_profile
            profile.update({
                'driver': 'GTiff',
                'dtype': 'int32',
                'count': 1,
                'nodata': self.no_data,
                'transform': dst_transform,
                'crs': dst_crs
            })
            for cls in conf['unique_class']:
                # Create binary mask for this class
                mask = (dest_array == cls).astype('int32')
                out_path = medium_env_tif.replace('[CLASS]', f'type{cls:03d}')
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(mask, 1)
        else:
            # 6. With PCA: collect presence values under valid extent for each class
            # Read reference extent mask
            with rasterio.open(ref_path) as ref:
                extent_mask = ref.read(1) == 1  # boolean mask
            self.extent_binary_reshape_idx = extent_mask.reshape(-1) == 1

            # Flatten and filter by extent mask
            flat_values = dest_array.flatten()[extent_mask.flatten()]
            # print(flat_values.shape)
            # Accumulate values in self.CCI_value list
            for idx, cls in enumerate(conf['unique_class']):
                class_presence = (flat_values == cls).astype('int32')
                if self.CCI_value[idx] is None:
                    self.CCI_value[idx] = class_presence
                else:
                    self.CCI_value[idx] = np.concatenate((self.CCI_value[idx], class_presence))
        
    def CCI_PCA(self, CCI_conf):
        """
        Perform PCA on collected categorical class presence data (self.CCI_value),
        then export the top components as GeoTIFF layers aligned to the reference grid.
        """
        # 1. Build a DataFrame for PCA
        # Concatenate each class presence vector as a column
        X = np.column_stack(self.CCI_value)   # 生成 shape=(n_samples, n_features) 的纯数值 2D 数组
        df_landcover = pd.DataFrame(X)
        # Use the 'unique_class' list from the last config to name columns
        # (Assumes all configs share the same 'unique_class')
        last_conf = next(iter(CCI_conf.values()))[-1]
        df_landcover.columns = last_conf['unique_class']
    
        # 2. Fit PCA
        pca = PCA()
        pca.fit(df_landcover)
        self.CCI_PCA_value = pca.transform(df_landcover)
        self.CCI_PCA_components = pca.components_
        self.CCI_PCA_variance_ratio = pca.explained_variance_ratio_
    
        # 3. Decide number of components to explain desired variance
        target = last_conf['PCA']
        cum_var = np.cumsum(self.CCI_PCA_variance_ratio)
        num_components = np.searchsorted(cum_var, target) + 1
        print(f'{num_components} components have been chosen.')
        print(f'Explain {cum_var[num_components-1]*100:.2f}% of variance.')
    
        # 4. Read reference grid properties for writing
        ref_path = './workspace/extent_binary.tif'
        with rasterio.open(ref_path) as ref:
            dst_transform = ref.transform
            dst_crs       = ref.crs
            dst_profile   = ref.profile.copy()
        # Update profile for PCA output
        dst_profile.update({
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': 1,
            'nodata': self.no_data,
            'transform': dst_transform,
            'crs': dst_crs
        })
    
        # 5. Export each selected principal component as a GeoTIFF
        num_cells = self.CCI_PCA_value.shape[0] // len(self.CCI_PCA_year)
        medium_env_dir = 'medium'
        for idx, year in enumerate(self.CCI_PCA_year):
            for comp in range(num_components):
                # Construct output path using template
                out_tif = os.path.join(
                    medium_env_dir,
                    last_conf['env_out_template']
                        .replace('[CLASS]', f'PC{(comp+1):02d}')
                        .replace('[YEAR]', f'{year:04d}')
                )
                os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    
                # Extract data for this year and component
                start = idx * num_cells
                end   = start + num_cells
                flat_values = self.CCI_PCA_value[start:end, comp]
    
                # Reconstruct full grid array
                grid_size = self.spatial_conf.y_num_cells * self.spatial_conf.x_num_cells
                full_array = np.full(grid_size, self.no_data, dtype='float32')
                full_array[self.extent_binary_reshape_idx] = flat_values
                grid_array = full_array.reshape(
                    self.spatial_conf.y_num_cells,
                    self.spatial_conf.x_num_cells
                )
    
                # Write to GeoTIFF
                with rasterio.open(out_tif, 'w', **dst_profile) as dst:
                    dst.write(grid_array, 1)
    
        # 6. Update env_medium_list for each year/month/component
        for year in self.CCI_PCA_year:
            for month in range(1, 13):
                ymd = datetime(year, month, 1).strftime('%Y-%m-%d')
                for comp in range(num_components):
                    key = f'landcover_PC{(comp+1):02d}'
                    if key not in self.env_medium_list:
                        self.env_medium_list[key] = {}
                    path = os.path.join(
                        medium_env_dir, key,
                        f'{key}_{year:04d}.tif'
                    )
                    self.env_medium_list[key][ymd] = path
                    
    def intersect_extents(self, tif):
        """
        Update the internal extent intersection mask by logically AND-ing
        the mask from the given TIFF with the existing intersection.
        """
        # 1. Read the new layer and create a boolean mask where data is valid
        with rasterio.open(tif) as src:
            arr = src.read(1)
            nodata = src.nodata
            valid_mask = arr != nodata  # True where there is data
    
        # 2. Initialize the intersection mask if this is the first layer
        if self.extent_binary_intersection is None:
            with rasterio.open('./workspace/extent_binary.tif') as ref:
                ref_arr = ref.read(1)
                # True where the original extent binary is 1
                self.extent_binary_intersection = (ref_arr == 1)
    
        # 3. Update the intersection mask by logical AND
        self.extent_binary_intersection = np.logical_and(
            self.extent_binary_intersection,
            valid_mask
        )
        
    def log_env_medium_list(self):
        medium_env_dir = 'medium'
        with open(f'./{medium_env_dir}/env_medium_list.json', 'w') as f:
            json.dump(self.env_medium_list, f)