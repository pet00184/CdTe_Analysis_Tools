import os
import shutil
import numpy as np
from astropy.table import vstack, Table
from astropy.io import fits
import matplotlib.pyplot as plt
from utils import lab_plotting as lp
import re
import skinnyfiles as skinny



def split_folder_by_size(src_folder, max_size_mb=20):
    max_size_bytes = max_size_mb * 1024 * 1024
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    files = sorted(files)  

    current_batch_size = 0
    current_folder_index = 1
    current_folder_path = os.path.join(src_folder, f"batch_{current_folder_index}")
    os.makedirs(current_folder_path, exist_ok=True)

    for f in files:
        fpath = os.path.join(src_folder, f)
        fsize = os.path.getsize(fpath)

        if current_batch_size + fsize > max_size_bytes:
            current_folder_index += 1
            current_folder_path = os.path.join(src_folder, f"batch_{current_folder_index}")
            os.makedirs(current_folder_path, exist_ok=True)
            current_batch_size = 0

        shutil.move(fpath, os.path.join(current_folder_path, f))
        current_batch_size += fsize

    print(f"Done. Created {current_folder_index} batches.")

# def combine_paths(cdte_folder, cdte_dict):
#     #COMBINING THE HDUS
#     print('starting combining')
#     if cdte_dict['source'] == 'nosource':
#         path_list = ['PATH1']
#     else:
#         path_list = ['PATH2'] #['PATH1', 'PATH2', 'PATH3']
#     for pathnum in path_list:
#         fits_files = []
#         pathname = f'foxsi_{cdte_dict["cdte"]}_{cdte_dict["cdteid"]}_{pathnum}.fits'
#         print(pathname)

#         for dirpath, dirnames, filenames in sorted(os.walk(cdte_folder)):
#             if pathname in filenames:
#                 fits_files.append(os.path.join(dirpath, pathname))
#                 print(f'appended {dirpath} fits')

#         table_list = []
#         for fn in fits_files:
#             print(f"now doing {fn}")
#             with fits.open(fn) as hdul:
#                 hdu = hdul[1]  # assuming extension 1 is the binary table
#                 table = Table(hdu.data)
#                 table_list.append(table)

#         if not table_list:
#             print(f"No valid table data found for {pathname}")
#             continue

#         combined_table = vstack(table_list)

#         with fits.open(fits_files[0]) as hdul:
#             header = hdul[1].header.copy()

#         table_hdu = fits.BinTableHDU(data=combined_table, header=header)
#         hdul_out = fits.HDUList([fits.PrimaryHDU(), table_hdu])
#         output_path = os.path.join(cdte_folder, f'combined_{pathname}')
#         hdul_out.writeto(output_path, overwrite=True)
#         print(f"Wrote combined FITS to {output_path}")

# def combine_paths(cdte_folder, cdte_dict):

#     print("starting combining")

#     # Decide which paths to combine
#     if cdte_dict["source"] == "nosource":
#         path_list = ["PATH1"]
#     else:
#         path_list = ["PATH2"]  # only doing PATH2

#     # Helper: extract batch number from folder name
#     def batch_sort_key(filepath):
#         """
#         Sort key that extracts the integer from paths like:
#         .../batch_1/... or .../batch_12/...
#         """
#         match = re.search(r"batch_(\d+)", filepath)
#         if match:
#             return int(match.group(1))
#         return float("inf")

#     # Loop through paths
#     for pathnum in path_list:

#         fits_files = []

#         # FITS filename is the same inside every batch folder
#         if cdte_dict["source"] == "nosource":
#             pathname = f"foxsi_{cdte_dict['cdte']}_{cdte_dict['cdteid']}_{pathnum}.fits"
#         else:
#             pathname = f"skinny_foxsi_{cdte_dict['cdte']}_{cdte_dict['cdteid']}_{pathnum}.fits.gz"
#         print(f"\nLooking for: {pathname}")

#         # Walk through all batch directories and collect matching FITS files
#         for dirpath, dirnames, filenames in os.walk(cdte_folder):
#             if pathname in filenames:
#                 full_path = os.path.join(dirpath, pathname)
#                 fits_files.append(full_path)
#                 print(f"  found: {full_path}")

#         # If nothing found, skip
#         if not fits_files:
#             print(f"No FITS files found for {pathname}")
#             continue

#         # ✅ Sort files properly by batch number
#         fits_files.sort(key=batch_sort_key)

#         print("\nCorrect chronological batch order:")
#         for f in fits_files:
#             print(" ", f)

#         # Read tables from all FITS files
#         table_list = []
#         for fn in fits_files:
#             print(f"\nNow reading: {fn}")
#             with fits.open(fn) as hdul:
#                 table = Table(hdul[1].data)
#                 table_list.append(table)

#         # Combine into one big table
#         combined_table = vstack(table_list)

#         # Copy header from first file
#         with fits.open(fits_files[0]) as hdul:
#             header = hdul[1].header.copy()

#         # Write combined output FITS
#         output_path = os.path.join(cdte_folder, f"combined_{pathname}")

#         table_hdu = fits.BinTableHDU(data=combined_table, header=header)
#         hdul_out = fits.HDUList([fits.PrimaryHDU(), table_hdu])

#         hdul_out.writeto(output_path, overwrite=True)

#         print(f"\n✅ Wrote combined FITS to: {output_path}")


        

        # combined_hdul = fits.HDUList()
        # for fn in fits_files:
        #     print(f"now doing {fn}")
        #     with fits.open(fn) as hdul:
        #         # copy each HDU into the combined list
        #         for h in hdul:
        #             combined_hdul.append(h.copy())

        # # 3) Write out
        # combined_hdul.writeto(f'{cdte_folder}/combined_{pathname}', overwrite=True)

def combine_paths(cdte_folder, cdte_dict):

    print("starting combining")

    # ============================================================
    # Helper: extract batch number from folder name
    # ============================================================

    def batch_sort_key(filepath):
        """
        Sort key that extracts the integer from paths like:
        .../batch_1/... or .../batch_12/...
        """
        match = re.search(r"batch_(\d+)", filepath)
        if match:
            return int(match.group(1))
        return float("inf")

    # ============================================================
    # Helper: slim all batch PATH2 files first
    # ============================================================

    def slim_all_batch_path2_files():
        """
        Walk through batch_* folders and slim every PATH2 FITS file.
        Produces skinny_*.fits.gz outputs inside each batch folder.
        """

        print("\n===================================================")
        print(" Slimming all batch PATH2 FITS files BEFORE combining")
        print("===================================================")

        for dirpath, dirnames, filenames in os.walk(cdte_folder):

            # Only process batch folders
            if "batch_" not in dirpath:
                continue

            for fn in filenames:

                # Only slim the original PATH2 files
                if fn.endswith("_PATH2.fits"):

                    input_fits = os.path.join(dirpath, fn)

                    output_fits_gz = os.path.join(
                        dirpath,
                        "skinny_" + fn.replace(".fits", ".fits.gz")
                    )

                    # Skip if skinny version already exists
                    if os.path.exists(output_fits_gz):
                        print(f"✓ Already slimmed: {output_fits_gz}")
                        continue

                    print("\n------------------------------------------")
                    print(f"Slimming batch file:\n{input_fits}")
                    print(f"→ Output:\n{output_fits_gz}")
                    print("------------------------------------------")

                    skinny.slim_and_compress_fits(input_fits, output_fits_gz)

        print("\n===================================================")
        print(" Finished slimming batch PATH2 files!")
        print("===================================================\n")

    # ============================================================
    # Decide which paths to combine
    # ============================================================

    if cdte_dict["save_source"] == "nosource":
        path_list = ["PATH1"]
    else:
        path_list = ["PATH2"]

        # ✅ Automatically slim all batch PATH2 files first
        slim_all_batch_path2_files()

    # ============================================================
    # Combine loop
    # ============================================================

    for pathnum in path_list:

        fits_files = []

        # Choose correct filename depending on source
        if cdte_dict["save_source"] == "nosource":
            pathname = f"foxsi_{cdte_dict['cdte']}_{cdte_dict['cdteid']}_{pathnum}.fits"
        else:
            pathname = f"skinny_foxsi_{cdte_dict['cdte']}_{cdte_dict['cdteid']}_{pathnum}.fits.gz"

        print(f"\nLooking for: {pathname}")

        # Walk through all batch directories and collect matching FITS files
        for dirpath, dirnames, filenames in os.walk(cdte_folder):
            if pathname in filenames:
                full_path = os.path.join(dirpath, pathname)
                fits_files.append(full_path)
                print(f"  found: {full_path}")

        # If nothing found, skip
        if not fits_files:
            print(f"No FITS files found for {pathname}")
            continue

        # ✅ Sort properly by batch number
        fits_files.sort(key=batch_sort_key)

        print("\nCorrect chronological batch order:")
        for f in fits_files:
            print(" ", f)

        # Read tables from all FITS files
        table_list = []
        for fn in fits_files:
            print(f"\nNow reading: {fn}")
            with fits.open(fn) as hdul:
                table = Table(hdul[1].data)
                table_list.append(table)

        # Combine into one big table
        combined_table = vstack(table_list)

        # Copy header from first file
        with fits.open(fits_files[0]) as hdul:
            header = hdul[1].header.copy()

        # Write combined output FITS
        output_path = os.path.join(cdte_folder, f"combined_{pathname}")

        table_hdu = fits.BinTableHDU(data=combined_table, header=header)
        hdul_out = fits.HDUList([fits.PrimaryHDU(), table_hdu])

        hdul_out.writeto(output_path, overwrite=True)

        print(f"\n✅ Wrote combined FITS to: {output_path}")

# def do_plots(cdte_folder, cdte_dict, e_range):
#     #NOW DOING PLOTS!!
#     main_save_name = f'combined_foxsi_{cdte_dict["cdte"]}_{cdte_dict["cdteid"]}'
#     main_save_folder = f'{cdte_folder}/Images'
#     os.makedirs(main_save_folder, exist_ok=True)
#     test = lp.Path1Plots(os.path.join(cdte_folder, main_save_name) + '_PATH1.fits')

#     #ADC Image Plot:
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#     test.make_image(ax)
#     ax.set_title(cdte_dict['plot_title_name'])
#     plt.savefig(f'{main_save_folder}/{main_save_name}' + '_adcimage.png')

#     #PEDESTAL/SPECTROGRAM Plots
#     fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#     test.plot_pedestal(axs[0], axs[1], top_range=1000)
#     plt.suptitle(cdte_dict['plot_title_name'])
#     fig = plt.gcf()  # Get current figure
#     fig.colorbar(test.pt[3], ax=axs[0], orientation='vertical')
#     fig.colorbar(test.al[3], ax=axs[1], orientation='vertical')
#     plt.savefig(f'{main_save_folder}/{main_save_name}' + f'_manualpedestal.png')

#     fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#     test.plot_pedestal(axs[0], axs[1], top_range=1000, manual=False)
#     plt.suptitle(cdte_dict['plot_title_name'])
#     fig = plt.gcf()  # Get current figure
#     fig.colorbar(test.pt[3], ax=axs[0], orientation='vertical')
#     fig.colorbar(test.al[3], ax=axs[1], orientation='vertical')
#     plt.savefig(f'{main_save_folder}/{main_save_name}' + f'_pedestal.png')

#     fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#     test.plot_nocmn_pedestal(axs[0], axs[1], range=1000)
#     plt.suptitle(cdte_dict['plot_title_name'])
#     fig = plt.gcf()  # Get current figure
#     fig.colorbar(test.pt[3], ax=axs[0], orientation='vertical')
#     fig.colorbar(test.al[3], ax=axs[1], orientation='vertical')
#     plt.savefig(f'{main_save_folder}/{main_save_name}' + f'_nocmn_pedestal.png')

#     #TIMING PLOT
#     fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#     test.time_plots(axs[0], axs[1])
#     plt.suptitle(cdte_dict['plot_title_name'])
#     plt.savefig(f'{main_save_folder}/{main_save_name}' + f'_time.png')

#     #DOING PATH 2 PLOTS IF THERE IS A SOURCE
#     if not cdte_dict['save_source'] == 'nosource':
#         print('doing path2 plots!')
#         '''Makes all count and single count spectra from merged energy hit lists, using PATH2 FITS.'''
#         test = lp.Path2Plots(f'{cdte_folder}/combined_foxsi_{cdte_dict["cdte"]}_{cdte_dict["cdteid"]}' + '_PATH2.fits', cdte_dict['source'])

#         fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#         test.make_spectra(ax, e_range)
#         ax.set_title(cdte_dict['plot_title_name'] + ', All Events')
#         plt.savefig(f'{main_save_folder}/{main_save_name}' + f'_alleventspectra.png')

#         fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#         test.make_spectra(ax, e_range, single_hit=True)
#         ax.set_title(cdte_dict['plot_title_name'] + ', Single Strip Events')
#         plt.savefig(f'{main_save_folder}/{main_save_name}' + f'_singleeventspectra.png')

#         ## also doing the gain shifting plots!!
#         test2 = lp.GainShiftingPlotting(f'{cdte_folder}/combined_foxsi_{cdte_dict["cdte"]}_{cdte_dict["cdteid"]}' + '_PATH2.fits', cdte_dict['source'], cdte_dict['cdte'], f'{main_save_folder}/{main_save_name}' + f'_peakdifference.png', 
#                 f'{main_save_folder}/{main_save_name}' + f'_shiftcomparison.png')
#         test2.find_peak_shifts(f'{cdte_folder}/combined_foxsi_{cdte_dict["cdte"]}_{cdte_dict["cdteid"]}' + '_PATH2.fits', cdte_dict['source'])


from astropy.io import fits
from astropy.table import Table, vstack
import os

def fix_multi_hdu_file(bad_fits_path, output_path=None):
    print(f"Opening: {bad_fits_path}")

    # Collect all tables from HDUs beyond the primary
    table_list = []
    with fits.open(bad_fits_path) as hdul:
        for i, hdu in enumerate(hdul[1:], start=1): 
            if isinstance(hdu, fits.BinTableHDU):
                table = Table(hdu.data)
                table_list.append(table)
                print(f"Appended HDU {i} with {len(table)} rows")
            else:
                print(f"Skipped HDU {i}: not a binary table")

        if not table_list:
            print("No BinTableHDUs found.")
            return

        # Stack all tables together
        combined_table = vstack(table_list)

        # Use header from first binary table HDU
        header = hdul[1].header.copy()

    # Create new HDUList: PrimaryHDU + one combined BinTableHDU
    primary_hdu = fits.PrimaryHDU()
    table_hdu = fits.BinTableHDU(data=combined_table, header=header)
    new_hdul = fits.HDUList([primary_hdu, table_hdu])

    if output_path is None:
        base, ext = os.path.splitext(bad_fits_path)
        output_path = f"{base}_fixed{ext}"

    new_hdul.writeto(output_path, overwrite=True)
    print(f"Wrote fixed file to: {output_path}")