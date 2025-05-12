#%%
# import packages

from program_utils import *


#%%
# define paths
basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins/bedmap3_basins.nc'

path_to_avhrr_precp = r'/ra1/pubdat/AVHRR_CloudSat_proj/preci_maps/2010_IMERG_based_0.5_res_mean_daily_ERA5_added_0_train_thresh_variable_mask_7_upto_70_smoothed_win_2_Ebtehaj_masking_params_1/'
path_to_era5 = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_0.25deg/ERA5_to_netcdf_files'
path_to_airs_ir = r'/ra1/pubdat/AVHRR_CloudSat_proj/TOVSAIRS/AIRS_from_Eric_to_netcdf_files'
path_to_imerg = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV7/DataV7_2007-2020'
path_to_ssmis_17 = r'/ra1/pubdat/AVHRR_CloudSat_proj/SSMI/data/daily/SSMIS-F17-ncfiles_pnt5'

# paths to put satellite precip over basins data
imerg_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/imerg_precip'
avhrr_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/avhrr_precip'
ssmis_17_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/ssmis_17_precip'
airs_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/airs_precip'
era5_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/era5_precip'

#%%
# floating variables

# load the different precipitation files
imerg_files = sorted([os.path.join(path_to_imerg,x) for x in os.listdir(path_to_imerg) if x.endswith('.nc4')])
# Filter file paths to include only files from the year 2010
imerg_files_2010 = [file for file in imerg_files if os.path.basename(file).split('.')[4].split('-')[0].startswith('2010')]

avhrr_precp_files = sorted([os.path.join(path_to_avhrr_precp,x) for x in os.listdir(path_to_avhrr_precp) if x.endswith('.tif')])

era5_file_2010 = os.path.join(path_to_era5,'ERA5_daily_precipitation_2010.nc') #Total_precip_2010_0.5.nc 

airs_file_2010 = os.path.join(path_to_airs_ir,'3A_AIRSV6_IR_HDD_daily_precipitation_2010.nc')

ssmis_17_files_2010 = sorted([os.path.join(path_to_ssmis_17,x) for x in os.listdir(path_to_ssmis_17) if x.endswith('.nc')])
#-----------------------------------------------------------------------------------------

# read ancilary data
basins  = xr.open_dataset(basins_path)

#%%
# read and process satellite precipitation data

# process imerg files
for idx, im in enumerate(imerg_files_2010, start=1):
    fle_svnme = os.path.join(imerg_basin_path,os.path.basename(im).replace('.nc4', '_imbie_basin_precip.nc'))
    img_bsn = process_imerg_file_to_basin(im, misc_out, basins['imbie'])

    encoding = {img_bsn.name:{"zlib": True, "complevel": 9}}
    img_bsn.to_netcdf(os.path.join(imerg_basin_path, fle_svnme), 
                      mode='w', 
                      format='NETCDF4', 
                      encoding=encoding)

    # Print progress every 25 files
    if idx % 25 == 0:
        print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")

#-----------------------------------------------------------------------------------------

# process avhrr files
for idx, avhrr in enumerate(avhrr_precp_files, start=1):
    fle_svnme = os.path.join(avhrr_basin_path,os.path.basename(avhrr).replace('.tif', '_imbie_basin_precip.nc'))
    avh_bsn = process_avhrr_file_to_basin(avhrr, '2010', basins['imbie'])

    encoding = {avh_bsn.name:{"zlib": True, "complevel": 9}}
    avh_bsn.to_netcdf(os.path.join(avhrr_basin_path, fle_svnme), 
                      mode='w', 
                      format='NETCDF4', 
                      encoding=encoding)

    # Print progress every 25 files
    if idx % 25 == 0:
        print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")

#-----------------------------------------------------------------------------------------

# process ssmis_17 files
for idx, ss in enumerate(ssmis_17_files_2010, start=1):
    fle_svnme = os.path.join(ssmis_17_basin_path,os.path.basename(ss).replace('.nc', '_imbie_basin_precip.nc'))
    ssmis_bsn = process_ssmis_file_to_basin(ss, basins['imbie'])

    encoding = {ssmis_bsn.name:{"zlib": True, "complevel": 9}}
    ssmis_bsn.to_netcdf(os.path.join(ssmis_17_basin_path, fle_svnme), 
                        mode='w', 
                        format='NETCDF4', 
                        encoding=encoding)

    # Print progress every 25 files
    if idx % 25 == 0:
        print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")

#-----------------------------------------------------------------------------------------

# process airs files
airs_data = xr.open_dataarray(airs_file_2010)
if 'x' in airs_data.coords and 'y' in airs_data.coords:
    airs_data = airs_data.rename({'x': 'lon', 'y': 'lat'})

for idx, ai_tme in enumerate(airs_data.time.values, start=1):
    airs_time = pd.to_datetime(ai_tme).strftime('%Y%m%d')
    fle_svnme = os.path.join(airs_basin_path, f'AIRS_3A_AIRSV6_IR_HDD_daily_precipitation_imbie_basin_{airs_time}.nc')

    airs_bsn = process_airs_file_to_basin(airs_data, ai_tme, basins['imbie'], fle_svnme)

    encoding = {airs_bsn.name: {"zlib": True, "complevel": 9}}
    airs_bsn.to_netcdf(os.path.join(airs_basin_path, fle_svnme), 
                        mode='w', 
                        format='NETCDF4', 
                        encoding=encoding)

    # Print progress every 25 files
    if idx % 25 == 0:
        print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")


#-----------------------------------------------------------------------------------------
# process era5 files
era5_data = xr.open_dataarray(era5_file_2010)
if 'x' in era5_data.coords and 'y' in era5_data.coords:
    era5_data = era5_data.rename({'x': 'lon', 'y': 'lat'})

for idx, er_tme in enumerate(era5_data.time.values, start=1):
    era5_time = pd.to_datetime(er_tme).strftime('%Y%m%d')
    fle_svnme = os.path.join(era5_basin_path, f'ERA5_daily_precipitation_imbie_basin_{era5_time}.nc')

    era5_bsn = process_era5_file_to_basin(era5_data, er_tme, basins['imbie'], fle_svnme)

    encoding = {era5_bsn.name: {"zlib": True, "complevel": 9}}
    era5_bsn.to_netcdf(os.path.join(era5_basin_path, fle_svnme), 
                        mode='w', 
                        format='NETCDF4', 
                        encoding=encoding)

    # Print progress every 25 files
    if idx % 25 == 0:
        print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")