import os
from astropy.io import fits


# ============================================================
# Columns needed for ALL of your PATH2 plotting + analysis code
# ============================================================

PATH2_NEEDED_COLUMNS = [

    # --- Timing / runtime ---
    "ti",
    "unixtime",
    "livetime",

    # --- Trigger filtering ---
    "flag_pseudo",
    "hitnum_pt",

    # --- Clump filtering ---
    "al_merged_nhit",
    "pt_merged_nhit",

    # --- Raw hit multiplicity ---
    "al_nhit",
    "pt_nhit",

    # --- Remapped channels (pedestals + imaging) ---
    "remapch_al",
    "remapch_pt",

    # --- ADC values ---
    "adc_al",
    "adc_pt",
    "adc_cmn_al",
    "adc_cmn_pt",

    # --- Common mode arrays ---
    "cmn_al",
    "cmn_pt",

    # --- Merged energies (spectra + gain shifting + charge sharing) ---
    "al_merged_energy_list",
    "pt_merged_energy_list",

    # --- Strip positions (pitch masking) ---
    "al_merged_position_list",
    "pt_merged_position_list",
]


# ============================================================
# Main slimming function
# ============================================================

def slim_and_compress_fits(input_fits, output_fits_gz,
                           keep_columns=PATH2_NEEDED_COLUMNS):
    """
    Slim a large PATH2 CdTe FITS file down to only required columns,
    and write out a compressed .fits.gz copy.

    Parameters
    ----------
    input_fits : str
        Path to original large PATH2 FITS file

    output_fits_gz : str
        Output path (should end in .fits.gz)

    keep_columns : list
        Columns to preserve
    """

    print("\n===================================================")
    print(" Slimming + Compressing PATH2 CdTe FITS File")
    print("===================================================")

    # --- File sizes before ---
    original_size = os.path.getsize(input_fits) / 1e9
    print(f"\nInput file:  {input_fits}")
    print(f"Size:        {original_size:.2f} GB")

    # --- Open FITS ---
    print("\nOpening FITS file (memmap=True)...")
    with fits.open(input_fits, memmap=True) as hdul:

        data = hdul[1].data
        original_cols = data.columns.names

        print(f"\nOriginal number of columns: {len(original_cols)}")

        # # --- Keep only columns that actually exist ---
        # existing_cols = [c for c in keep_columns if c in original_cols]
        # missing_cols = [c for c in keep_columns if c not in original_cols]

        # print("\nKeeping columns:")
        # for c in existing_cols:
        #     print("   ✓", c)

        # if missing_cols:
        #     print("\nMissing columns (skipped safely):")
        #     for c in missing_cols:
        #         print("   ⚠", c)

        # # --- Extract reduced table ---
        # reduced_data = data[existing_cols]

        # # --- Build new FITS file ---
        # primary_hdu = hdul[0]
        # new_table_hdu = fits.BinTableHDU(reduced_data)

        # --- Keep only columns that actually exist ---
        existing_cols = [c for c in keep_columns if c in original_cols]

        print("\nKeeping columns:")
        for c in existing_cols:
            print("   ✓", c)

        # --- Build reduced FITS table safely ---
        print("\nRebuilding slim FITS table...")

        primary_hdu = hdul[0]
        new_cols = fits.ColDefs([data.columns[name] for name in existing_cols])
        new_table_hdu = fits.BinTableHDU.from_columns(new_cols)

        # Ensure output folder exists
        os.makedirs(os.path.dirname(output_fits_gz), exist_ok=True)

        # --- Write compressed output ---
        print(f"\nWriting compressed FITS file:\n{output_fits_gz}")

        fits.HDUList([primary_hdu, new_table_hdu]).writeto(
            output_fits_gz,
            overwrite=True,
            output_verify="silentfix"   # fixes harmless FITS header quirks
        )

    # --- File sizes after ---
    new_size = os.path.getsize(output_fits_gz) / 1e9
    savings = 100 * (1 - new_size / original_size)

    print("\n===================================================")
    print(" Done!")
    print("===================================================")
    print(f"Output size: {new_size:.2f} GB")
    print(f"Saved:       {savings:.1f}% disk space")
    print("===================================================\n")


# ============================================================
# Example usage
# ============================================================

# ============================================================
# Batch slimming utility
# ============================================================

def slim_all_batch_path2_files(base_folder, keep_columns=PATH2_NEEDED_COLUMNS):
    """
    Walk through batch_* folders and slim every PATH2 FITS file found.
    Produces skinny_*.fits.gz outputs inside each batch folder.
    """

    print("\n===================================================")
    print(" Slimming ALL batch PATH2 FITS files")
    print("===================================================")

    for dirpath, dirnames, filenames in os.walk(base_folder):

        # Only look inside batch_* folders
        if "batch_" not in dirpath:
            continue

        # Find PATH2 FITS files
        for fn in filenames:
            if fn.endswith("_PATH2.fits"):

                input_fits = os.path.join(dirpath, fn)

                # Output file name
                output_fits_gz = os.path.join(
                    dirpath,
                    "skinny_" + fn.replace(".fits", ".fits.gz")
                )

                print("\n------------------------------------------")
                print(f"Batch file found:\n{input_fits}")
                print(f"→ Writing skinny version:\n{output_fits_gz}")
                print("------------------------------------------")

                slim_and_compress_fits(
                    input_fits,
                    output_fits_gz,
                    keep_columns=keep_columns
                )

    print("\n===================================================")
    print(" Finished slimming all batch PATH2 files!")
    print("===================================================\n")


if __name__ == "__main__":

    # input_file = "/Users/pet00184/FOXSI_Analysis/updated2026plotting/BerkeleyCoolingTests/Aug282025/CdTe4Fe55/combined_foxsi_cdte4_run800_PATH2.fits"
    # output_file = "/Users/pet00184/FOXSI_Analysis/updated2026plotting/BerkeleyCoolingTests/Aug282025/CdTe4Fe55/skinny_foxsi_cdte4_run800_PATH2.fits.gz"

    # slim_and_compress_fits(input_file, output_file)

    #skinny the batches: 
    base_folder = "/Users/pet00184/FOXSI_Analysis/updated2026plotting/BerkeleyCoolingTests/Aug282025/CdTe4Fe55_30minRun/"
    slim_all_batch_path2_files(base_folder)
