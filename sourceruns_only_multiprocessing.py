import os
import sys
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import cdte_tools_py.io.load_raw as lr
from utils import split_files as sf
from utils import run_pipeline_plots as rpp

#############################################################################
''' This is the code to use if we did a source test in front of each detector, and we just want to look at the runs where the source was in front of 
that detector (not also processing the no source data for the other detectors.) The other code is much more adaptable if you are looking to do just things
from a specific run! Just use this one for the above case.
'''
cdte_list = ['CdTe1', 'CdTe3', 'CdTe4', 'CdTe5']
cdte_folder_list = ['Am241_CdTe1', 'Am241_CdTe3', 'Am241_CdTe4', 'Am241_CdTe5']
run_number = ['run3', 'run379', 'run447', 'run383']
directory = f'/Users/pet00184/FOXSI_Analysis/test_new_gse_stuff/Am241_CdTe3' #this should be the parent directory of the cdte_folder_list

date = 'July30'
voltage = '200V'
temp = 'neg20'
source = 'Am241'
energy_plotting_range = (10, 80)
lab_testing = True  # skip path3?

# For better plotting and saving files
save_source = 'Am241'
fancy_temp = '-20ºC'
############################################################################

cdte_id_dict = {'CdTe1': 'cdte1',
                'CdTe3': 'cdte3',
                'CdTe4': 'cdte4',
                'CdTe5': 'cdte5'}

## function to do each individual batch
def process_batch(batch, path, run_dict, energy_plotting_range, fancy_temp, voltage, lab_testing):
    """Process a single batch folder for one CdTe"""
    cdte_fits_folder = os.path.join(path, batch)
    ## checking to see if I already did these- sometimes things crash!
    if os.path.exists(f'{cdte_fits_folder}/foxsi_{run_dict["cdte"]}_{run_dict["cdteid"]}_PATH1.fits') and save_source == 'nosource':
        print(f'already did batch {batch}')
        return
    elif os.path.exists(f'{cdte_fits_folder}/foxsi_{run_dict["cdte"]}_{run_dict["cdteid"]}_PATH3.fits') and not save_source == 'nosource':
        print(f'already did batch {batch}')
        return

    save_path = f'{cdte_fits_folder}/Images'
    # os.makedirs(save_path, exist_ok=True)

    print(f'DOING {batch} NOW with temp {fancy_temp} at {voltage}')
    cdte_fits = [os.path.join(cdte_fits_folder, f) for f in sorted(os.listdir(cdte_fits_folder))]
    raw_data = lr.raw_from_files(file_list=cdte_fits)
    x = rpp.SavingAndPlotting(raw_data, save_path, run_dict, cdte_fits_folder)
    x.do_runs(e_range=energy_plotting_range, lab_testing=lab_testing)
    print(f'Batch {batch} DONE')

#the main thing that goes through each CdTe
def process_cdte(cdte, cdte_folder, rn, directory, date, voltage, temp, source, save_source, fancy_temp, lab_testing, energy_plotting_range, parallel_batches=True):
    """Process one CdTe detector across all batches"""
    path = os.path.join(directory, cdte_folder, cdte)
    print(f'DOING {cdte}')

    # First split into batches
    sf.split_folder_by_size(path)
    batch_list = os.listdir(path)

    run_dict = {
        'cdte': cdte_id_dict[cdte],
        'voltage': voltage,
        'run_temperature': temp,
        'save_source': save_source,
        'date': date,
        'cdteid': rn,
        'source': source,
        'plot_title_name': f'{cdte} {date} 2025 \n {rn}, {source}, {fancy_temp}, {voltage}'
    }

    # Decide whether to parallelize batches or not
    if parallel_batches:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_batch, batch, path, run_dict,
                                       energy_plotting_range, fancy_temp, voltage, lab_testing)
                       for batch in batch_list]
            for f in as_completed(futures):
                f.result()
    else:
        for batch in batch_list:
            process_batch(batch, path, run_dict, energy_plotting_range, fancy_temp, voltage, lab_testing)

    # Combine and re-plot
    print(f'COMBINING {cdte} NOW!!')
    sf.combine_paths(path, run_dict)
    sf.do_plots(path, run_dict, e_range=energy_plotting_range)
    print(f'CdTe {cdte} DONE')

if __name__ == "__main__":
    cpu_count = os.cpu_count() or 1
    n_cdte = len(cdte_list)

    #if CPUs >= 2 × detectors, allow inner batch parallelism
    allow_inner_parallel = cpu_count >= (n_cdte * 2)

    print(f"CPU count = {cpu_count}. Parallelizing batches={allow_inner_parallel}")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_cdte, cdte, cdte_folder, rn, directory, date,
                                   voltage, temp, source, save_source, fancy_temp,
                                   lab_testing, energy_plotting_range, allow_inner_parallel)
                   for cdte, cdte_folder, rn in zip(cdte_list, cdte_folder_list, run_number)]
        for f in as_completed(futures):
            f.result()

    print("ALL CdTe runs completed!!")