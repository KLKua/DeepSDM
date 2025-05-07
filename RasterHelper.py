import numpy as np
import os
import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import rasterio
from osgeo import gdal
from osgeo import gdalconst
import cv2
from matplotlib import pyplot as plt
import re
from sklearn.decomposition import PCA
import yaml
import hashlib
import h5py

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
        
        in_ds = gdal.Open(input_env, gdalconst.GA_ReadOnly)

#         in_nodata = in_ds.GetRasterBand(1).GetNoDataValue()
#         assert(in_nodata is not None)

        _, xres, _, _, _, yres = in_ds.GetGeoTransform()
        in_proj = in_ds.GetProjection()

        # TODO
        # Need to clarify what value it would be with tif of southern hemisphere
        
        x_start = self.spatial_conf.x_start
        y_start = self.spatial_conf.y_start
#         x_end = self.spatial_conf.x_end
#         y_end = self.spatial_conf.y_end
        
        if self.spatial_conf.out_res is not None:
            self.spatial_conf.out_res = abs(self.spatial_conf.out_res)
        else:
            assert(abs(xres) == abs(yres))            
            self.spatial_conf.out_res = abs(xres)
            
        digit_parts = str(self.spatial_conf.out_res).split('.')
        assert(len(digit_parts) == 2)
        self.num_digits_after_decimal = min(len(digit_parts[1]), 6)

#         self.spatial_conf.x_start = self.res_rounder(self.res_rounder(x_start / self.spatial_conf.out_res, 0) * self.spatial_conf.out_res)
#         self.spatial_conf.y_start = self.res_rounder(self.res_rounder(y_start / self.spatial_conf.out_res, 0) * self.spatial_conf.out_res)

#         self.spatial_conf.x_num_cells = int(np.ceil(self.res_rounder(abs(x_end - self.spatial_conf.x_start) / self.spatial_conf.out_res)))
#         self.spatial_conf.y_num_cells = int(np.ceil(self.res_rounder(abs(y_end - self.spatial_conf.y_start) / self.spatial_conf.out_res)))

        self.spatial_conf.x_num_cells = int(self.spatial_conf.num_of_grid_x * self.spatial_conf.grid_size)
        self.spatial_conf.y_num_cells = int(self.spatial_conf.num_of_grid_y * self.spatial_conf.grid_size)

        self.spatial_conf.x_end = self.res_rounder(self.spatial_conf.x_start + self.spatial_conf.x_num_cells * self.spatial_conf.out_res)
        self.spatial_conf.y_end = self.res_rounder(self.spatial_conf.y_start + self.spatial_conf.y_num_cells * self.spatial_conf.out_res)
        out_trans = (self.spatial_conf.x_start, self.spatial_conf.out_res, 0, self.spatial_conf.y_end, 0, -self.spatial_conf.out_res)
        
        driver = gdal.GetDriverByName('GTiff')
        output = driver.Create('workspace/extent_env_example.tif', self.spatial_conf.x_num_cells, self.spatial_conf.y_num_cells, 1, gdalconst.GDT_Float32)

        output.GetRasterBand(1).SetNoDataValue(self.no_data)
        # 
        output.SetGeoTransform(out_trans)
        output.SetProjection(in_proj)

        #
        out_proj = in_proj
        gdal.ReprojectImage(in_ds, output, in_proj, out_proj, gdalconst.GRA_Bilinear)

        in_ds = None
        driver  = None
        output = None        
        
        #############################################################
        
        in_ds  = gdal.Open('workspace/extent_env_example.tif', gdalconst.GA_ReadOnly)
        in_trans = in_ds.GetGeoTransform()
        in_proj = in_ds.GetProjection()

        extent_array = in_ds.ReadAsArray()
        extent_array = np.where(extent_array == self.no_data, 0, 1)

        driver= gdal.GetDriverByName('GTiff')
        output = driver.Create('workspace/extent_binary.tif', self.spatial_conf.x_num_cells, self.spatial_conf.y_num_cells, 1, gdalconst.GDT_Int16)
        output.SetGeoTransform(in_trans)
        output.SetProjection(in_proj)
        export = output.GetRasterBand(1).WriteArray(extent_array)
        output.GetRasterBand(1).SetNoDataValue(in_ds.GetRasterBand(1).GetNoDataValue())

        in_ds = None
        driver = None
        output = None
        export = None
        
        return self.spatial_conf

    def raw_to_medium_(self, raw_env_tif, medium_env_tif):
        destination = gdal.Open('./workspace/extent_binary.tif')
        dst_transform = destination.GetGeoTransform()
        dst_projection = destination.GetProjection()

        src_tif  = gdal.Open(raw_env_tif, gdalconst.GA_ReadOnly)
        src_trans = src_tif.GetGeoTransform()
        src_proj = src_tif.GetProjection()

        dst_driver= gdal.GetDriverByName('GTiff')
        dst_tif = dst_driver.Create(medium_env_tif, 
                               destination.RasterXSize, 
                               destination.RasterYSize, 
                               1, 
                               gdalconst.GDT_Float32)

        # 设置输出文件地理仿射变换参数与投影
        dst_tif.GetRasterBand(1).SetNoDataValue(self.no_data)
        dst_tif.SetGeoTransform(dst_transform)
        dst_tif.SetProjection(dst_projection)

        # 重投影，插值方法为双线性内插法
        gdal.ReprojectImage(src_tif, dst_tif, src_proj, dst_projection, gdalconst.GRA_Bilinear)

        destination = None
        src_tif = None
        dst_driver  = None
        dst_tif = None

        
        # intersect envrionmental layer's extent 
        self.intersect_extents(medium_env_tif)
        
        
        
    def raw_to_medium_agg_(self, doy_to_month_tifs):
        
        raw_env_tifs = []
        medium_env_tifs = []
        
        for doy_to_month_tif in doy_to_month_tifs:
            raw_env_tifs.append(doy_to_month_tif['raw'])
            medium_env_tifs.append(doy_to_month_tif['medium'])
        
        assert(len(np.unique(medium_env_tifs))==1)
        
        destination = gdal.Open('./workspace/extent_binary.tif')
        dst_transform = destination.GetGeoTransform()
        dst_projection = destination.GetProjection()

        mem_raster_arrs = []
        for raw_env_tif in raw_env_tifs:
            src_tif  = gdal.Open(raw_env_tif, gdalconst.GA_ReadOnly)
            src_trans = src_tif.GetGeoTransform()
            src_proj = src_tif.GetProjection()
            
            #mem_driver= gdal.GetDriverByName('MEM')
            # not really mem driver
            mem_driver= gdal.GetDriverByName('GTiff')
            mem_tif = mem_driver.Create("/tmp/not_really_mem_driver.tif", 
                                   destination.RasterXSize, 
                                   destination.RasterYSize, 
                                   1, 
                                   gdalconst.GDT_Float32)
        
            # 设置输出文件地理仿射变换参数与投影
            mem_tif.GetRasterBand(1).SetNoDataValue(np.nan)
            mem_tif.SetGeoTransform(dst_transform)
            mem_tif.SetProjection(dst_projection)

            # 重投影，插值方法为双线性内插法
            gdal.ReprojectImage(src_tif, mem_tif, src_proj, dst_projection, gdalconst.GRA_Bilinear)
            
            mem_raster_arrs.append(mem_tif.GetRasterBand(1).ReadAsArray())
            mem_tif = None

        
        mem_raster_arr_avg = np.nanmean(np.stack(mem_raster_arrs), axis=0)
        
        mem_raster_arr_avg = np.where(np.isnan(mem_raster_arr_avg), self.no_data, mem_raster_arr_avg)
            
        dst_driver = gdal.GetDriverByName('GTiff')
        #dst_tif = dst_driver.CreateCopy(medium_env_tifs[0], mem_tif)

        dst_tif = mem_driver.Create(medium_env_tifs[0], 
                               destination.RasterXSize, 
                               destination.RasterYSize, 
                               1, 
                               gdalconst.GDT_Float32)

        dst_tif.SetGeoTransform(dst_transform)
        dst_tif.SetProjection(dst_projection)
        dst_tif.GetRasterBand(1).WriteArray(mem_raster_arr_avg)
        dst_tif.GetRasterBand(1).SetNoDataValue(self.no_data)
            
        destination = None
        src_tif = None
        dst_driver  = None
        dst_tif = None
        
        # intersect envrionmental layer's extent 
        self.intersect_extents(medium_env_tifs[0])
        
        
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
        self.env_medium_list = {}
        for env in env_raw_conf:
            for conf in env_raw_conf[env]:
                print(conf)
                self.build_env_(env, conf)
                
        # after all the env layers been read
        # start processing about the extent_binary_intersect 
        def floodfill(extent_binary_intersect):
            extent_binary_intersect_ = extent_binary_intersect.astype(np.uint8)*255
            mask = np.zeros((extent_binary_intersect_.shape[0]+2, extent_binary_intersect_.shape[1]+2), np.uint8)
            cv2.floodFill(extent_binary_intersect_, mask, (1559, 0), 255)
            extent_binary_intersect_inv = cv2.bitwise_not(extent_binary_intersect_)
            extent_binary_intersect_out = extent_binary_intersect.astype(np.uint8) * 255 | extent_binary_intersect_inv
            
            dst = gdal.Open('./workspace/extent_binary.tif', gdalconst.GA_ReadOnly)
            dst_transform = dst.GetGeoTransform()
            dst_projection = dst.GetProjection()
            dst_X = dst.RasterXSize
            dst_Y = dst.RasterYSize
            dst = None
            
            dst_driver= gdal.GetDriverByName('GTiff')
            dst_tif = dst_driver.Create('./workspace/extent_binary.tif', 
                      dst_X, 
                      dst_Y, 
                      1, 
                      gdalconst.GDT_Int32)

            dst_tif.SetGeoTransform(dst_transform)
            dst_tif.SetProjection(dst_projection)
            dst_tif.GetRasterBand(1).WriteArray(np.sign(extent_binary_intersect_out).astype('int'))
            dst_tif.GetRasterBand(1).SetNoDataValue(self.no_data)
            
            dst_driver = None
            dst_tif = None
            
        floodfill(self.extent_binary_intersection)
        
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
        with rasterio.open('./workspace/extent_binary.tif', 'r') as f:
            extent_bin_array = f.read(1)
        extent_bin_partitions = np.zeros((spatial_conf.num_of_grid_y, spatial_conf.num_of_grid_x))
        for i_row in range(spatial_conf.num_of_grid_y):
            for i_col in range(spatial_conf.num_of_grid_x):
                subplot = extent_bin_array[i_row*spatial_conf.grid_size:(i_row+1)*spatial_conf.grid_size, i_col*spatial_conf.grid_size:(i_col+1)*spatial_conf.grid_size]
                if subplot.sum() > 0:
                    extent_bin_partitions[i_row, i_col] = 1
        np.savetxt('./workspace/extent_partition.txt', extent_bin_partitions, fmt='%i', delimiter=',')
        
        num_train_grids = int(np.round(extent_bin_partitions.sum() * train_ratio))
        train_val_partitions = np.where(np.random.uniform(size=(spatial_conf.num_of_grid_y, spatial_conf.num_of_grid_x)) >= train_ratio, 0, 1)
        while (train_val_partitions*extent_bin_partitions).sum() != num_train_grids:
            train_val_partitions = np.where(np.random.uniform(size=train_val_partitions.shape)>=train_ratio, 0, 1).astype(np.uint8)
        np.savetxt('./workspace/partition.txt', train_val_partitions, fmt='%i', delimiter=',')
        
        
    def view_train_val_splits(self, partition_file='./workspace/partition.txt'):
        spatial_conf = self.spatial_conf
        ext_bin  = gdal.Open('./workspace/extent_binary.tif', gdalconst.GA_ReadOnly)
        ext_bin_array = ext_bin.ReadAsArray()
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
                    h5f.create_dataset(date_span, data = rst * extent_binary, compression = 'gzip')
                    
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
        destination = gdal.Open('./workspace/extent_binary.tif')
        dst_transform = destination.GetGeoTransform()
        dst_projection = destination.GetProjection()

        src_nc  = gdal.Open(f'NETCDF:{raw_env_nc}:{conf["layer_name"]}', gdalconst.GA_ReadOnly)
        src_proj = src_nc.GetProjection()

        dst_driver= gdal.GetDriverByName('GTiff')
        dst_tif = dst_driver.Create('/tmp/not_really_mem_driver.tif', 
                                    destination.RasterXSize, 
                                    destination.RasterYSize, 
                                    1, 
                                    gdalconst.GDT_Int32)
        dst_tif.SetGeoTransform(dst_transform)
        dst_tif.SetProjection(dst_projection)
        gdal.ReprojectImage(src_nc, dst_tif, src_proj, dst_projection, gdalconst.GRA_Mode)
        dst_value = dst_tif.GetRasterBand(1).ReadAsArray()
        dst_tif = None
        dst_driver = None
        src_nc = None
        
        # without PCA
        # directly export all the tiffs
        if conf['PCA'] == None:
            for value in conf['unique_class']:
                dst_unique_value = np.where(dst_value == value, 1, 0)
                dst_driver= gdal.GetDriverByName('GTiff')
                dst_tif = dst_driver.Create(medium_env_tif.replace('[CLASS]', f'type{value:03d}'), 
                                            destination.RasterXSize, 
                                            destination.RasterYSize, 
                                            1, 
                                            gdalconst.GDT_Int32)

                dst_tif.GetRasterBand(1).WriteArray(dst_unique_value)
                dst_tif.GetRasterBand(1).SetNoDataValue(self.no_data)
                dst_tif.SetGeoTransform(dst_transform)
                dst_tif.SetProjection(dst_projection)

                dst_driver  = None
                dst_tif = None
        
        # with PCA
        else:
            # extent_binary
            extent_binary = destination.GetRasterBand(1).ReadAsArray()
            self.extent_binary_reshape_idx = extent_binary.reshape(-1) == 1
            for i, value in enumerate(conf['unique_class']):
                
                # only compute PCA with the value in extent_binary
                dst_unique_value = np.where(dst_value[extent_binary == 1] == value, 1, 0)
                if self.CCI_value[i] is None:
                    self.CCI_value[i] = dst_unique_value.reshape(-1)
                else:
                    self.CCI_value[i] = np.concatenate((self.CCI_value[i], dst_unique_value.reshape(-1)))      
        
        destination = None
        
    def CCI_PCA(self, CCI_conf):
        
        for env in CCI_conf:
            for conf in CCI_conf[env]:
                pass
        
        df_landcover = pd.DataFrame()
        for value in self.CCI_value:
            df_landcover = pd.concat((df_landcover, pd.DataFrame(value)), axis = 1)
        df_landcover = df_landcover.set_axis(conf['unique_class'], axis = 1)
        
        pca = PCA()
        pca.fit(df_landcover)
        self.CCI_PCA_value = pca.transform(df_landcover)
        self.CCI_PCA_components = pca.components_
        self.CCI_PCA_variance_ratio = pca.explained_variance_ratio_
        
        # decide how many components should be logged
        num_components = 0
        while True:
            if self.CCI_PCA_variance_ratio[:num_components].sum() >= conf['PCA']:
                break
            num_components += 1
        print(f'{num_components} components have been chosen.')
        print(f'Explain {self.CCI_PCA_variance_ratio[:num_components].sum()*100}% of variance. ')
        
        # export the pca-value landcover layers
        destination = gdal.Open('./workspace/extent_binary.tif')
        dst_transform = destination.GetGeoTransform()
        dst_projection = destination.GetProjection()
        
        num_cell = self.CCI_PCA_value.shape[0] // len(self.CCI_PCA_year)
        medium_env_dir = 'medium'
        for i, year in enumerate(self.CCI_PCA_year):
            for n_com in range(num_components):
                medium_env_tif = os.path.join(medium_env_dir, conf['env_out_template']).replace('[CLASS]', f'PC{n_com:02d}').replace('[YEAR]', f'{year:04d}')
                
                # create folder 'land_cover_PCXX'
                if not os.path.exists('/'.join(medium_env_tif.split('/')[:-1])):
                    os.makedirs('/'.join(medium_env_tif.split('/')[:-1]))
                
                rst_value_extent = self.CCI_PCA_value[i*num_cell:(i+1)*num_cell, n_com]
                rst_value_fullsize = np.zeros([self.spatial_conf.y_num_cells * self.spatial_conf.x_num_cells, ])
                rst_value_fullsize[self.extent_binary_reshape_idx] = rst_value_extent
                rst_value = rst_value_fullsize.reshape(self.spatial_conf.y_num_cells, self.spatial_conf.x_num_cells)
                dst_driver = gdal.GetDriverByName('GTiff')
                dst_tif = dst_driver.Create(medium_env_tif, 
                                            destination.RasterXSize, 
                                            destination.RasterYSize, 
                                            1, 
                                            gdalconst.GDT_Float32)

                dst_tif.GetRasterBand(1).WriteArray(rst_value)
                dst_tif.GetRasterBand(1).SetNoDataValue(self.no_data)
                dst_tif.SetGeoTransform(dst_transform)
                dst_tif.SetProjection(dst_projection)

                dst_driver  = None
                dst_tif = None 
        destination = None
        
        
        for year in self.CCI_PCA_year:
            for month in range(1, 13):
                for pc in range(num_components):
                    if f'landcover_PC{pc:02d}' not in self.env_medium_list:
                        self.env_medium_list[f'landcover_PC{pc:02d}'] = {}
                        
                    # representation of yyyy-mm-dd
                    y_m_d = datetime.strftime(datetime.strptime(f'{year}-{month}', '%Y-%m'), '%Y-%m-%d')
                    self.env_medium_list[f'landcover_PC{pc:02d}'][y_m_d] = os.path.join(medium_env_dir, f'landcover_PC{pc:02d}', f'landcover_PC{pc:02d}_{year:04d}.tif')
                    
    def intersect_extents(self, tif):
        dst = gdal.Open(tif, gdalconst.GA_ReadOnly)
        if self.extent_binary_intersection is None:
            with rasterio.open('./workspace/extent_binary.tif', 'r') as f:
                extent_bin_array = f.read(1)
            assert(extent_bin_array.shape == (dst.RasterYSize, dst.RasterXSize))
            self.extent_binary_intersection = np.where(extent_bin_array == 1, True, False)
#             self.extent_binary_intersection = np.full((dst.RasterYSize, dst.RasterXSize), True)
        self.extent_binary_intersection = np.logical_and(~(dst.GetRasterBand(1).ReadAsArray() == dst.GetRasterBand(1).GetNoDataValue()), self.extent_binary_intersection)
        dst = None
        
    def log_env_medium_list(self):
        medium_env_dir = 'medium'
        with open(f'./{medium_env_dir}/env_medium_list.json', 'w') as f:
            json.dump(self.env_medium_list, f)