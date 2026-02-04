import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.special import erfc, erf
from scipy.optimize import minimize

class MakeEfficiencyPlot_try2:

    #VTH_list = [0, 10, 12, 14, 16, 18, 2, 4, 5, 6, 8]

    def __init__(self, directory, save_directory, cdte, platinum_cutoff, chosen_vth, VTH_list):
        self.directory = directory
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)
        self.platinum_cutoff = platinum_cutoff
        self.path2_files = sorted([f for f in os.listdir(self.directory) if f.endswith('_PATH2.fits')])
        print(self.path2_files)
        self.cdte = cdte
        self.chosen_vth = chosen_vth
        self.VTH_list = VTH_list

    def get_al_singlehit_counts(self, vth_data):
        '''Masks the path2 file so that only counts with a single hit on the Al side, and any hits on the Pt side are saved.'''
        vth_data_masked = vth_data[vth_data['al_nhit'] == 1]
        vth_data_masked = vth_data_masked[vth_data_masked['pt_nhit'] == 1]
        vth_data_masked = vth_data_masked[vth_data_masked['pt_merged_energy_list'][:, 0] > self.platinum_cutoff]
        return vth_data_masked

    def calculate_run_time(self, vth_data, livetime_correction):
        '''Calculates total run time.'''
        if livetime_correction:
            ti_run_seconds = np.sum(vth_data['livetime']) * 1e-8
        else:
            start_ti = vth_data['ti'][0]
            end_ti = vth_data['ti'][-1]
            ti_run_seconds = (end_ti - start_ti) * 160e-9
        return ti_run_seconds
    
    def calculate_counts(self, vth_data):
        '''Calculates the count rate for the path2 file using the ti run time.'''
        total_counts = len(vth_data['ti'])
        total_count_error = np.sqrt(total_counts)
        return total_counts, total_count_error

    def do_single_vth(self, vth_path2_file, vth, ax, livetime_correction):
        vth_data = fits.open(os.path.join(self.directory, vth_path2_file))[1].data
        masked_vth_data = self.get_al_singlehit_counts(vth_data)
        total_counts, total_count_error = self.calculate_counts(masked_vth_data)
        runtime_seconds = self.calculate_run_time(vth_data, livetime_correction)
        ax.errorbar(vth, total_counts, yerr=total_count_error, fmt='o', c='k')
        return ax, total_counts, runtime_seconds
    
    def model(self, vth, runtimes, C0, G, sigma):
        vth = np.asarray(vth)
        runtimes = np.asarray(runtimes)
        return 0.5 * C0 * runtimes * erfc((G * vth - 5.9) / (np.sqrt(2) * sigma))

    def poisson_nll(self, params, vth, counts, runtimes):
        '''calculates the poisson negative log likelihood'''
        C0, G, sigma = params
        mu = self.model(vth, runtimes, C0, G, sigma)
        mu = np.clip(mu, 1e-10, None)
        return np.sum(mu - counts * np.log(mu))
        
    def calculate_model_fit(self, vth, counts, runtimes):
        initial_guess = [60, 0.5, 1.0]
        bounds = [(0, None), (0, 1), (1e-6, 5)]
        result = minimize(self.poisson_nll, initial_guess, args=(vth, counts, runtimes), bounds=bounds)
        if result.success:
            fitted_C0, fitted_G, fitted_sigma = result.x
            print(f"C0: {fitted_C0}")
            print(f"G: {fitted_G}")
            print(f"Sigma: {fitted_sigma}")
            return fitted_C0, fitted_G, fitted_sigma
        else:
            print("Optimization failed:", result.message)
            return None, None, None
        
    def efficiency_model(self, E_array, vth, G, sigma):
        '''Calculates the efficiency curve for a given vth, with the fitted G and sigma parameters.'''
        return 0.5 * (erf((E_array - G*vth)/(np.sqrt(2) * sigma)) + 1)

    def make_plots(self, livetime_correction=False):
        final_counts = []
        runtimes = []
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_xlabel('Vth')
        ax.set_ylabel('Counts')
        for vth_path2_file, vth in zip(self.path2_files, self.VTH_list):
            print(f'doing {vth_path2_file} with {vth}')
            _, counts, runtime_seconds = self.do_single_vth(vth_path2_file, vth, ax, livetime_correction)
            runtimes.append(runtime_seconds)
            final_counts.append(counts)
        #sorting the values so that they are in order
        sorted_indices = np.argsort(self.VTH_list)
        vth_sorted = np.array(self.VTH_list)[sorted_indices]
        counts_sorted = np.array(final_counts)[sorted_indices]
        runtimes_sorted = np.array(runtimes)[sorted_indices]
        print(vth_sorted)
        print(runtimes_sorted)
        print(counts_sorted)
        #doing the model fit
        C0, G, sigma = self.calculate_model_fit(vth_sorted, counts_sorted, runtimes_sorted)

        #doing the counts plot
        if C0 is not None:
            print('vth:', vth_sorted)
            print('runtimes:', runtimes_sorted)
            model_counts = self.model(vth_sorted, runtimes_sorted, C0, G, sigma)
            ax.plot(vth_sorted, model_counts, c='r', label=f'Fitted Model: \nC0: {C0:.2f} \nG: {G:.2f} \nSigma: {sigma:.2f}')
        if livetime_correction:
            ax.set_title(f'Counts for Varying Vth \n {self.cdte}, Pt Energy Cutoff={self.platinum_cutoff}, LIVETIME CORRECTED')
        else:
            ax.set_title(f'Counts for Varying Vth \n {self.cdte}, Pt Energy Cutoff={self.platinum_cutoff}')
        ax.set_xlim(-1, 19)
        ax.set_xticks(range(0, 19))
        ax.legend()
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle=':')
        if livetime_correction:
            plt.savefig(os.path.join(self.save_directory, f'{self.cdte}_counts_poissonfit_livetimecorrected.png'), dpi=250, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_directory, f'{self.cdte}_counts_poissonfit.png'), dpi=250, bbox_inches='tight')

        #add in a count rate plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_xlabel('Vth')
        ax.set_ylabel('Count Rate')
        ax.grid(True, which='major', linestyle=':', zorder=1)
        ax.scatter(vth_sorted, counts_sorted/runtimes_sorted, c='k', zorder=3)
        ax.plot(vth_sorted, model_counts/runtimes_sorted, c='r', label=f'Fitted Model: \nC0: {C0:.2f} \nG: {G:.2f} \nSigma: {sigma:.2f}', zorder=3)
        if livetime_correction:
            ax.set_title(f'Count Rate for Varying Vth \n {self.cdte}, Pt Energy Cutoff={self.platinum_cutoff}, LIVETIME CORRECTED')
        else:
            ax.set_title(f'Count Rate for Varying Vth \n {self.cdte}, Pt Energy Cutoff={self.platinum_cutoff}')
        ax.set_xlim(-1, 19)
        ax.set_xticks(range(0, 19))
        ax.legend()
        ax.minorticks_on()
        if livetime_correction:
            plt.savefig(os.path.join(self.save_directory, f'{self.cdte}_countrate_poissonfit_livetimecorrected.png'), dpi=250, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_directory, f'{self.cdte}_countrate_poissonfit.png'), dpi=250, bbox_inches='tight')
        
        #doing the efficiency plot
        energies = np.linspace(0, 10, 21)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        for chosen_vth in self.VTH_list:
            efficiency_curve = self.efficiency_model(energies, chosen_vth, G, sigma)
            line = ax.plot(energies, efficiency_curve)
            sorted_indices = np.argsort(efficiency_curve)
            eff_sorted = efficiency_curve[sorted_indices]
            energies_sorted = energies[sorted_indices]

            if eff_sorted[0] <= 0.5 <= eff_sorted[-1]:
                energy_05 = np.interp(0.5, eff_sorted, energies_sorted)
            else:
                energy_05 = np.nan
            eff_color = line[0].get_color()
            ax.vlines(energy_05, ymin=0, ymax=0.5, colors=eff_color, linestyles='dashed', label=f'{chosen_vth} 50% Efficiency = {energy_05:.2f}')
        ax.axhline(0.5, color='black', ls = '--')
        ax.set_xlim(0, 10)
        ax.set_ylabel('Efficiency')
        ax.set_xlabel('Energy [keV]')
        ax.set_xticks(range(0, 11))
        ax.set_ylim(0, 1.1)
        ax.minorticks_on()
        ax.legend()
        ax.grid(True, which='both')  # 'both' includes both major and minor grid lines
        ax.grid(True, which='minor', linestyle=':', color='gray')
        if livetime_correction:
            ax.set_title(f'Efficiency Curves for Varying Vth, Pt Energy Cutoff={self.platinum_cutoff}, LIVETIME CORRECTED COUNTS')
        else:
            ax.set_title(f'Efficiency Curves for Varying Vth, Pt Energy Cutoff={self.platinum_cutoff}')
        if livetime_correction:
            plt.savefig(os.path.join(self.save_directory, f'{self.cdte}_efficiency_allvth_poissonfit_livetimecorrected.png'), dpi=250, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_directory, f'{self.cdte}_efficiency_allvth_poissonfit.png'), dpi=250, bbox_inches='tight')
