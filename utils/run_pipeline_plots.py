from utils import saving_FITS as sf
from utils import lab_plotting as lp
import matplotlib.pyplot as plt
import os
import importlib
importlib.reload(lp)
importlib.reload(sf)

########## PLOTTING NEEDS UPDATING NOW THAT I HAVE CHANGED THE PLOTTING CODE!!!!!!!!!

class SavingAndPlotting:

    def __init__(self, cdte_fits, save_path, run_dict, cdte_fits_folder):
        self.cdte_fits = cdte_fits
        self.cdte_path_savename = os.path.join(cdte_fits_folder, f'foxsi_{run_dict["cdte"]}_{run_dict["cdteid"]}')
        self.run_dict = run_dict

        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        keys = list(self.run_dict.values())
        self.savefile = '_'.join(keys[:-2])
        self.saving_cdte = sf.SaveFITSFromRaw(cdte_fits, run_dict['cdte'], self.cdte_path_savename, frame_data=False)
        self.saving_cdte.get_fits_path1()
        self.saving_cdte.make_manual_adc_cmn()
        self.saving_cdte.save_cdte_fits_path1()

    def check_for_readout_type(self):
        '''Checks to see if the PATH1 plots are sparse or full readout'''
        if self.saving_cdte.path1_data['hitnum_pt'][0] > 100:
            return 'full'
        else:
            return 'sparse'
        
    def make_path1_plots(self, pedestal_top):
        self.readout_type = self.check_for_readout_type()
        test = lp.Path1Plots(self.cdte_path_savename + '_PATH1.fits')

        #ADC Image Plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        test.make_image(ax)
        ax.set_title(self.run_dict['plot_title_name'])
        plt.savefig(os.path.join(self.save_path, self.savefile) + f'_adcimage_{self.readout_type}.png')

        #PEDESTAL/SPECTROGRAM Plots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        test.plot_pedestal(axs[0], axs[1], pedestal_top)
        plt.suptitle(self.run_dict['plot_title_name'])
        fig = plt.gcf()  # Get current figure
        fig.colorbar(test.pt[3], ax=axs[0], orientation='vertical')
        fig.colorbar(test.al[3], ax=axs[1], orientation='vertical')
        plt.savefig(os.path.join(self.save_path, self.savefile) + f'_manualpedestal_{self.readout_type}.png')

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        test.plot_pedestal(axs[0], axs[1], pedestal_top, manual=False)
        plt.suptitle(self.run_dict['plot_title_name'])
        fig = plt.gcf()  # Get current figure
        fig.colorbar(test.pt[3], ax=axs[0], orientation='vertical')
        fig.colorbar(test.al[3], ax=axs[1], orientation='vertical')
        plt.savefig(os.path.join(self.save_path, self.savefile) + f'_pedestal_{self.readout_type}.png')

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        test.plot_nocmn_pedestal(axs[0], axs[1], pedestal_top)
        plt.suptitle(self.run_dict['plot_title_name'])
        fig = plt.gcf()  # Get current figure
        fig.colorbar(test.pt[3], ax=axs[0], orientation='vertical')
        fig.colorbar(test.al[3], ax=axs[1], orientation='vertical')
        plt.savefig(os.path.join(self.save_path, self.savefile) + f'_nocmnpedestal_{self.readout_type}.png')

        #TIMING PLOT
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        test.time_plots(axs[0], axs[1])
        plt.suptitle(self.run_dict['plot_title_name'])
        plt.savefig(os.path.join(self.save_path, self.savefile) + f'_time_{self.readout_type}.png')

    def make_path2_plots(self, e_range):
        '''Makes all count and single count spectra from merged energy hit lists, using PATH2 FITS.'''
        test = lp.Path2Plots(self.cdte_path_savename + '_PATH2.fits', self.run_dict['source'])

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        test.make_spectra(ax, e_range)
        ax.set_title(self.run_dict['plot_title_name'] + ', All Events')
        plt.savefig(os.path.join(self.save_path, self.savefile) + f'_alleventspectra_{self.readout_type}.png')

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        test.make_spectra(ax, e_range, single_hit=True)
        ax.set_title(self.run_dict['plot_title_name'] + ', Single Strip Events')
        plt.savefig(os.path.join(self.save_path, self.savefile) + f'_singleeventspectra_{self.readout_type}.png')

        ## also doing the gain shifting plots!!
        test2 = lp.GainShiftingPlotting(self.cdte_path_savename + '_PATH2.fits', self.run_dict['source'], self.run_dict['cdte'], os.path.join(self.save_path, self.savefile) + f'_peakdifference_{self.readout_type}.png', 
                os.path.join(self.save_path, self.savefile) + f'_shiftcomparison_{self.readout_type}.png')
        test2.find_peak_shifts(self.cdte_path_savename + '_PATH2.fits', self.run_dict['source'])

    def do_runs(self, pedestal_top=1000, e_range=(10, 80), lab_testing=False):
        '''What we run to do the automated path pipelines and plotting.
        Inputs: 
        --------------
        pedestal_top: optional manual input for the maximum plotted pedestal/spectrograph value
        all_event_ylim: optional manual input for the ylim on the all events spectra.
        single_event_ylim: optional manual input for the ylim on the single events spectra.

        Plots are saved to the specified file path given by self.save_path.
        '''
        #self.make_path1_plots(pedestal_top)
        print(f'Plots saved to {self.save_path}')
        if not self.run_dict['save_source'] == 'nosource':
            print('Now starting PATH2.')
            self.saving_cdte.save_cdte_fits_path2()
            #self.make_path2_plots(e_range)
            print(f'Plots saved to {self.save_path}')
            if not lab_testing:
                print('Now starting PATH3')
                self.saving_cdte.save_cdte_fits_path3()
        else:
            print('No source, so not doing PATH2 or PATH3.')

        
    