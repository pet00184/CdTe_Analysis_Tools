import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d

#functions to be used for plotting summary
def define_path_type(fits_file):
    return int(fits_file[-6])

def find_readout_mode(cdte_data):
    if cdte_data['hitnum_pt'][0] > 100:
        return 'full'
    else:
        return 'sparse'
    
class Path1Plots:
    ''' Plots for looking at CdTe data that has been through path 1.

    Plots: 
    - ADC Image Plot
    - Pedestals (with and without common mode subtraction)
    - Livetime histogram with expected distribution
    - ADC common mode vs. livetime 2D histogram
    '''

    def __init__(self, cdte_data, filter_pseudo_triggers=False, keep_only_single_clumps=False):
        '''
        Input:
        cdte_fits (.fits) = cdte_fits can either be a PATH1 or PATH2 .fits file.
        filter_pseudo_triggers (True/False) = if True, keeps only real triggers for plotting.
        keep_only_single_clumps = if True, keeps only triggers that resulted in a single clump on both 
           the Pt and Al sides. Setting this to true will help get rid of noise in analysis, but is not useful if trying to look at noise levels or compare to what we will see realtime with the GSE.
        '''
        self.cdte_data = cdte_data
        if filter_pseudo_triggers:
            pseudo_mask = self.cdte_data['flag_pseudo'] == 0
            self.cdte_data = self.cdte_data[pseudo_mask]
        if keep_only_single_clumps:
            self.keep_single_clumps()
        self.calculate_livetime_and_countrates()

    def keep_single_clumps(self):
        mask_al = self.cdte_data['al_merged_nhit'] == 1
        self.cdte_data = self.cdte_data[mask_al]
        mask_pt = self.cdte_data['pt_merged_nhit'] == 1
        self.cdte_data = self.cdte_data[mask_pt]

    def calculate_livetime_and_countrates(self):
        ''' Calculates the livetime and total count rates for the observation. This is done at the beginning so that I can use global variables for the two livetime plots, and so that the countrate can be included in pedestal plotting (still need to add that in when finalizing the summary info).
        '''
        self.livetime = self.cdte_data['livetime'] * 1e-8 #to get into seconds
        self.total_runtime = np.sum(self.livetime)
        self.total_counts = len(self.cdte_data['ti'])
        self.count_rate = self.total_counts/self.total_runtime

    def take_out_edge_channels(self):
        '''Sets everything in the guard rail remapped channels to their null values. For all trigger hits that occur within channels 0-5 and 124-127, the remapped channel is set to 128 and the cmn_adc_al value is set to 0. This is done to avoid mapping only the edges when making an image incorporating maximum adc values for each trigger.'''
        remapped_al = self.cdte_data['remapch_al'].copy()
        cmn_adc_al = self.cdte_data['adc_cmn_al'].copy()
        idx = np.where((remapped_al > 123) | (remapped_al < 6))
        remapped_al[idx] = 128 
        cmn_adc_al[idx] = 0

        remapped_pt = self.cdte_data['remapch_pt'].copy()
        cmn_adc_pt = self.cdte_data['adc_cmn_pt'].copy()
        idx = np.where((remapped_pt > 123) | (remapped_pt < 6))
        remapped_pt[idx] = 128 
        cmn_adc_pt[idx] = 0

        return remapped_al, cmn_adc_al, remapped_pt, cmn_adc_pt

    def find_max_adc(self, cmn_adc_arr, remapch_arr):
        '''Finds the maximum ADC index for each trigger. (This is the index not the actual value!)'''
        max_indx = np.argmax(cmn_adc_arr, axis=1)
        max_values = cmn_adc_arr.copy()[np.arange(cmn_adc_arr.shape[0]), max_indx]
        max_remapch_values = remapch_arr.copy()[np.arange(remapch_arr.shape[0]), max_indx]
        return max_values, max_remapch_values
    
    def find_zero_adcs(self, max_adc_al, max_adc_pt):
        ''' Finds the overall indexes where either Pt or Al has 0 as its maximum value, meaning that 
        the trigger happened solely in the guard rail area.'''
        return np.where((max_adc_al == 0) | (max_adc_pt==0))
    
    def make_image(self, ax):
        '''Making the Image!'''
        remapped_al, cmn_adc_al, remapped_pt, cmn_adc_pt = self.take_out_edge_channels()

        #find the vactual ADC value and the remapch channel where the maximum ADC happens for that trigger
        max_adc_al_vals, max_remapch_al_vals = self.find_max_adc(cmn_adc_al, remapped_al)
        max_adc_pt_vals, max_remapch_pt_vals = self.find_max_adc(cmn_adc_pt, remapped_pt)
        
        # #removing zeros from the maximums
        zeros = self.find_zero_adcs(max_adc_al_vals, max_adc_pt_vals) 
        self.max_adc_al_vals = np.delete(max_adc_al_vals.copy(), zeros[0])
        self.max_remapch_al_vals = np.delete(max_remapch_al_vals.copy(), zeros[0])
        self.max_adc_pt_vals = np.delete(max_adc_pt_vals.copy(), zeros[0])
        self.max_remapch_pt_vals = np.delete(max_remapch_pt_vals.copy(), zeros[0])
        
        #making the image
        ax.hist2d(self.max_remapch_al_vals, self.max_remapch_pt_vals, range=[[6, 123], [6, 123]], bins=[117, 117], cmin=1)
        ax.set_xlabel('Remapped Channels (Al)')
        ax.set_ylabel('Remapped Channels (Pt)')
        ax.set_xlim(0, 128)
        ax.set_ylim(0, 128)
        return ax
    
    def plot_pedestal(self, ax1, ax2, top_range, cmn_subtracted=True):
        '''Plots a pedestal for a sindle CdTe detector, either with or without the common mode subtracted for each ASIC.
        Input: 
        ax1, ax2 (matplotlib axs objects) = axis for the Pt (ax1) and Al (ax2) plots
        
        Returns: 
        ax1, ax2 
        '''
        
        idx = np.where(self.cdte_data['remapch_al'] < 128)
        remapch_al = self.cdte_data['remapch_al'][idx]
        if cmn_subtracted:
            adc_cmn_al = self.cdte_data['adc_cmn_al'][idx]
        else:
            adc_cmn_al = self.cdte_data['adc_al'][idx]

        idx = np.where(self.cdte_data['remapch_pt'] < 128)
        remapch_pt = self.cdte_data['remapch_pt'][idx]
        if cmn_subtracted:
            adc_cmn_pt = self.cdte_data['adc_cmn_pt'][idx]
        else:
            adc_cmn_pt = self.cdte_data['adc_pt'][idx]
        
        #fig, [ax1, ax2] = plt.subplots(1, 2)
        self.pt = ax1.hist2d(remapch_pt, adc_cmn_pt, bins=[130, 1000], range=[[-1, 129], [0, 1000]], cmap='jet', cmin=1, vmin=1, vmax=2000)
        ax1.axhline(0, ls='--', c='k', lw=0.5)
        ax1.set_xlabel('Remapped Channels (Pt)')
        ax1.set_ylabel('ADC Values (Pt)')
        if cmn_subtracted:
            ax1.set_ylabel('ADC Common Mode Subtracted Values (Pt)')
        ax1.set_title('Pt')
        ax1.set_ylim(-10, top_range)

        self.al = ax2.hist2d(remapch_al, adc_cmn_al, bins=[130, 1000], range=[[-1, 129], [0, 1000]], cmap='jet', cmin=1, vmin=1, vmax=2000)
        ax2.axhline(0, ls='--', c='k', lw=0.5)
        ax2.set_xlabel('Remapped Channels (Al)')
        ax2.set_ylabel('ADC Values (Al)')
        if cmn_subtracted:
            ax2.set_ylabel('ADC Common Mode Subtracted Values (Al)')
        ax2.set_title('Al')
        ax2.set_ylim(-10, top_range)

        return ax1, ax2
    
    def time_plots(self, ax1, ax2):
        ''' Plots a histogram of the 'ti' values for each trigger, as well as a plot of the Ti vs. unix time. As Ti rolls over, this should look like a sawtooth plot.
        '''
        count_rate = np.histogram(self.cdte_data['ti'], bins=100)
        ax1.stairs(*count_rate)
        ax1.set_title('Count Rate')
        ax1.set_xlabel('Ti')
        ax1.set_ylabel('Number of Triggers')

        ax2.plot(self.cdte_data['unixtime'], self.cdte_data['ti'])
        ax2.set_title('Unix time vs. Ti')
        ax2.set_ylabel('Ti')
        ax2.set_xlabel('Unixtime')

    def livetime_histogram(self, ax, livetime_plot_cutoff):
        ''' Plots a histogram of the livetimes, with the expected exponential waiting time distrubution overplotted. The exponential distrubution is based on the expected waiting time for another count if the counts follow a Poisson distribution, given by: 

                I(t)dt = r exp(-rt)dt 
        where r is the count rate (ct/s) and so rt are the total counts.

        Input:
        ax = matplotlib axis object for the histogram
        livetime_plot_cutoff = the maximum x range for the plot

        Returns:
        ax
        '''
        # Histogram of the counts (not normalized)
        counts, bins, _ = ax.hist(self.livetime, bins=80, density=False, label="Livetime histogram")
        
        # Exponential curve scaled to counts, and accounting for bin width
        bin_width = bins[1] - bins[0]
        t = np.linspace(0, self.livetime.max(), 500)
        expected_counts = len(self.livetime) * self.count_rate * np.exp(-self.count_rate * t) * bin_width

        ax.plot(t, expected_counts, label=r"$N r e^{-rt}$")
        ax.set_xlim(0, livetime_plot_cutoff)
        # Labels
        ax.set_xlabel("Event Livetime (s)")
        ax.set_ylabel("Counts per bin")
        ax.set_title("Histogram of Livetime Distribution")
        ax.legend(loc="upper right")

        ax.text(
            0.98, 0.8,  # x, y in axes coords (near top right)
            f"Incident count rate = {self.count_rate:.1f} ct/s",
            transform=ax.transAxes,
            fontsize=10,
            horizontalalignment='right',
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        return ax

    def cmnmode_vs_livetime(self, axes, livetime_plot_cutoff, common_mode_range):
        '''2D histogram of the common mode vs. livetime. This is based off connections Yixian has found between commond mode values changing (either increasing or decreasing) for very low livetimes. Note that Yixian did not see a difference in gain shifting when taking out low livetime or outlier common mode values.

        Input:
        ax = matplotlib axis object for the histogram
        livetime_plot_cutoff = the maximum x range for the plot
        common_mode_range [min, max] = range to plot common modes for all ASICs

        Returns:
        ax
        '''
        ASIC0 = self.cdte_data['cmn_pt'][:, 0]
        ASIC1 = self.cdte_data['cmn_pt'][:, 1]
        ASIC2 = self.cdte_data['cmn_al'][:, 0]
        ASIC3 = self.cdte_data['cmn_al'][:, 1]

        asics = [ASIC0, ASIC1, ASIC2, ASIC3]
        titles = ["ASIC 0 (Pt)",
                "ASIC 1 (Pt)",
                "ASIC 2 (Al)",
                "ASIC 3 (Al)"]

        for ax, asic, title in zip(axes.flat, asics, titles):
            h = ax.hist2d(
                self.livetime,
                asic,
                bins=[500, 500]
            )
            ax.set_xlim(0, livetime_plot_cutoff)
            ax.set_ylim(common_mode_range)
            ax.set_title(title)
            ax.set_xlabel("Livetime (s)")
            ax.set_ylabel("Common Mode (ADC)")

            # Colorbar per subplot (optional but useful)
            plt.colorbar(h[3], ax=ax)

        return axes
    
    
class Path2Plots:
    ''' Plots for looking at CdTe data that has been through path 2. 

    Plots available to make: 
    - Single hit, double hit and all hit spectra (with option to keep just single clumps as well). 0.2 keV bin widths are used for these, following Yixian's thesis work.
    - Energy vs. livetime 2D histogram
    '''

    sourcelines_dict = {
            'Am241': [59.5, 26.3, 13.9, 17.7, 20.7],
            'Am241 and Fe55': [59.5, 26.3, 13.9, 17.7, 20.7, 5.9],
            'Fe55': [5.9],
            'Ba133': [7.8, 11.9, 30.9, 35.1]
        }

    def __init__(self, cdte_data, source_list, filter_pseudo_triggers=False, keep_only_single_clumps=False):
        '''
        Inputs:
        cdte_data (fits.open(cdte_data)[1].data) = cdte_data must be through PATH2 or higher. 
        source_list = defined source(s) used in test. Will be one of the keywords in sourcelines_dict.
        filter_pseudo_triggers (True/False) = if True, keeps only real triggers for plotting.
        keep_only_single_clumps = if True, keeps only triggers that resulted in a single clump on both 
           the Pt and Al sides. Setting this to true will help get rid of noise in analysis, and is done automatically when looking at charge sharing. (This makes no difference for single hit spectra.)
        '''
        self.path2_data = cdte_data
        self.source_list = source_list
        if filter_pseudo_triggers:
            pseudo_mask = self.path2_data['flag_pseudo'] == 0
            self.path2_data = self.path2_data[pseudo_mask]
        if keep_only_single_clumps:
            self.keep_single_clumps()
        self.calculate_livetime_and_countrates()

    def keep_single_clumps(self):
        mask_al = self.path2_data['al_merged_nhit'] == 1
        self.path2_data = self.path2_data[mask_al]
        mask_pt = self.path2_data['pt_merged_nhit'] == 1
        self.path2_data = self.path2_data[mask_pt]

    def single_hits(self):
        mask_al = self.path2_data["al_nhit"]==1 
        mask_pt = self.path2_data["pt_nhit"]==1
        cdte_data_ge_al = self.path2_data[mask_al]
        cdte_data_ge_pt = self.path2_data[mask_pt]
        return cdte_data_ge_al, cdte_data_ge_pt
    
    def double_hits(self):
        mask_al = self.path2_data["al_nhit"]==2 
        mask_pt = self.path2_data["pt_nhit"]==2
        cdte_data_ge_al = self.path2_data[mask_al]
        cdte_data_ge_pt = self.path2_data[mask_pt]
        return cdte_data_ge_al, cdte_data_ge_pt
    
    def no_masking(self):
        return self.path2_data, self.path2_data
    
    def calculate_livetime_and_countrates(self):
        ''' Calculates the livetime and total count rates for the observation. This is done at the beginning so that I can use global variables for the two livetime plots, and so that the countrate can be included in pedestal plotting (still need to add that in when finalizing the summary info).
        '''
        self.livetime = self.path2_data['livetime'] * 1e-8 #to get into seconds
        self.total_runtime = np.sum(self.livetime)
        self.total_counts = len(self.path2_data['ti'])
        self.count_rate = self.total_counts/self.total_runtime

    def make_spectra(self, ax, e_range, single_hit=False, double_hit=False):
        ''' Produces either a single hit, double hit or all event specttra (keep single clump filtering in mind). 

        Input:
        ax = matplotlib axis object
        e_range = energy range for creating bins and to plot the spectra over. 
        single_hit = if True, only single hits on both side are included.
        double_hit = if True, only double hits on both side are included.
        '''
        cdte_data_ge_al, cdte_data_ge_pt = self.no_masking()
        plotname = "All Event Spectra"
        if single_hit:
            cdte_data_ge_al, cdte_data_ge_pt = self.single_hits()
            plotname = "Single Hit Spectra"
        if double_hit:
            cdte_data_ge_al, cdte_data_ge_pt = self.double_hits()
            plotname = "Double Hit Spectra, No Gap Loss Correction"

        lines = self.sourcelines_dict[self.source_list] 

        # --- define 0.2 keV bins ---
        bin_width = 0.2  # keV
        bins = np.arange(e_range[0], e_range[1] + bin_width, bin_width)

        ax.set_xlim(e_range)
        pc, pb = np.histogram(cdte_data_ge_pt["pt_merged_energy_list"][:,0], bins=bins)
        ac, ab = np.histogram(cdte_data_ge_al["al_merged_energy_list"][:,0], bins=bins)
        ax.stairs(pc, pb, label="Pt")
        ax.stairs(ac, ab, label="Al")
        ax.legend(fontsize=9)
        for line in lines:
            ax.axvline(line, color="k", ls=":")
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("Counts (0.2 keV bins)")
        ax.set_title(plotname)
        return ax

    def energy_vs_livetime(self, axes, source, livetime_plot_cutoff, e_range, single_hit=False, double_hit=False):
        '''2D histogram of the energy vs. livetime, to help check for any energy gain shifting due to low livetimes or pileup.

        Input:
        axes = matplotlib axis object, for a 1x2 subplot.
        livetime_plot_cutoff = maximum livetime included in x-range when plotting.
        e_range [min, max] = energy range included in y-axis plotting.
        single_hit = if True, only single hits on both side are included.
        double_hit = if True, only double hits on both side are included.
        '''
        livetime = self.path2_data['livetime'] * 1e-8
        cdte_data_ge_al, cdte_data_ge_pt = self.no_masking()
        plotname = "All Events"
        # if single_hit:
        #     cdte_data_ge_al, cdte_data_ge_pt = self.single_hits()
        #     plotname = "Single Events"
        # if double_hit:
        #     cdte_data_ge_al, cdte_data_ge_pt = self.double_hits()
        #     plotname = "Double Events"

        lines = self.sourcelines_dict[self.source_list] 
        energy_bins = np.arange(e_range[0], e_range[1] + 0.2, 0.2) #keeping 0.2 keV bins
        livetime_bins = 500

        #Pt energies
        h0 = axes[0].hist2d(
            livetime,
            cdte_data_ge_pt["pt_merged_energy_list"][:,0],
            bins=[livetime_bins, energy_bins]
        )
        axes[0].set_title(f"Pt Energies vs Livetime, {plotname}")
        axes[0].set_xlabel("Livetime (s)")
        axes[0].set_ylabel("Energy (keV)")
        axes[0].set_xlim(0, livetime_plot_cutoff)
        axes[0].set_ylim(e_range)
        plt.colorbar(h0[3], ax=axes[0])

        # Al energies
        h1 = axes[1].hist2d(
            livetime,
            cdte_data_ge_al["al_merged_energy_list"][:,0],
            bins=[livetime_bins, energy_bins]
        )
        axes[1].set_title(f"Al Energies vs Livetime, {plotname}")
        axes[1].set_xlabel("Livetime (s)")
        axes[1].set_xlim(0, livetime_plot_cutoff)
        axes[1].set_ylim(e_range)
        plt.colorbar(h1[3], ax=axes[1])

        # --- Overplot source lines ---
        for ax in axes:
            for E in self.sourcelines_dict[source]:
                ax.axhline(
                    E,
                    color='red',
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.7
                )
        return axes





class GainShiftingPlotting:
    '''Plotting class to specifically measure gain shifting for CdTe spectra. This so far has only been done for single hit spectra, but double hit capabilities are written (without gap loss correction, so not sure how useful that is).'''

    sourcelines_dict = {
            'Am241': [59.5, 26.3, 13.9, 17.7, 20.7],
            'Am241 and Fe55': [59.5, 26.3, 13.9, 17.7, 20.7, 5.9],
            'Fe55': [5.9],
            'Ba133': [7.8, 11.9, 30.9, 35.1]
        }

    def __init__(self, cdte_data, source_list, filter_pseudo_triggers=True, keep_only_single_clumps=True, double=False):
        '''
        Inputs:
        cdte_data (fits.open(cdte_data)[1].data) = cdte data that has been through at least PATH2.
        source_list (str) = source(s) used for test, which will be one of the keywords in 
            sourcelines_dict
        filter_pseudo_triggers (True/False) = takes out the pseudo triggers for analysis
        keep_only_single_clumps (True/False) = keeps only single clump events for analysis. This will always happen with single hit events.
        '''
        self.cdte_data = cdte_data
        self.source = source_list
        self.double = double
        self.bin_width = 0.2  # keeping with 0.2 keV bin sizes
        self.sigma_bin = self.bin_width / np.sqrt(12) #bin standard deviation error, assuming a uniform distribution across all bins


        #dong any psuedo filter or single clump filtering
        if filter_pseudo_triggers:
            pseudo_mask = self.cdte_data['flag_pseudo'] == 0
            self.cdte_data = self.cdte_data[pseudo_mask]
        if keep_only_single_clumps:
            self.keep_single_clumps()
        self.calculate_livetime_and_countrates()

        #find the shifts here:
        self.find_peak_shifts() 

    def calculate_livetime_and_countrates(self):
        ''' Calculates the livetime and total count rates for the observation. This is done at the beginning so that I can use global variables for the two livetime plots, and so that the countrate can be included in pedestal plotting (still need to add that in when finalizing the summary info).
        '''
        self.livetime = self.cdte_data['livetime'] * 1e-8 #to get into seconds
        self.total_runtime = np.sum(self.livetime)
        self.total_counts = len(self.cdte_data['ti'])
        self.count_rate = self.total_counts/self.total_runtime

    def keep_single_clumps(self):
        mask_al = self.cdte_data['al_merged_nhit'] == 1
        self.cdte_data = self.cdte_data[mask_al]
        mask_pt = self.cdte_data['pt_merged_nhit'] == 1
        self.cdte_data = self.cdte_data[mask_pt]

    def single_hits(self, cdte_data):
        '''Keeps only triggers with one hit on each side of the detector. Also doing an energy cutoff below 80 keV to clean up data.
        '''
        cdte_data_mask = cdte_data['al_merged_energy_list'][:, 0] < 80
        cdte_data = cdte_data[cdte_data_mask]
        mask_al = cdte_data["al_nhit"] == 1 
        mask_pt = cdte_data["pt_nhit"] == 1
        cdte_data_ge_al = cdte_data[mask_al]
        cdte_data_ge_pt = cdte_data[mask_pt]
        return cdte_data_ge_al, cdte_data_ge_pt
    
    def double_hits(self, cdte_data):
        '''Keeps only triggers with two hits on each side of the detector. Also doing an energy cutoff below 80 keV to clean up data.
        '''
        cdte_data_mask = cdte_data['al_merged_energy_list'][:, 0] < 80
        cdte_data = cdte_data[cdte_data_mask]
        mask_al = cdte_data["al_nhit"] == 2 
        mask_pt = cdte_data["pt_nhit"] == 2
        cdte_data_ge_al = cdte_data[mask_al]
        cdte_data_ge_pt = cdte_data[mask_pt]
        return cdte_data_ge_al, cdte_data_ge_pt

    def _analyze_peaks(self, cdte_energy, smoothing_sigma, prominence, height, match_tol):
        ''' Do the peak finding analysis, to end up with the expected peaks, observed peaks, and the error for the measured peak values. 

        Error includes both the error based on assuming a uniform distribution across the energy bins, as well as error on the calculated peak location (assuming a Gaussian distribution for peaks). The two errors are as follows: 

            sigma_bin = 0.2 keV / sqrt(12)
            sigma_centroid = sigma_peak / sqrt(N_peak) = (FWHM/2.55) / sqrt(N/0.76)

        Input:
        cdte_energy (arr) = array of CdTe energy values. Do hitnumber masking before this.
        smoothing_sigma (int) = smoothing integer for gaussian filter (dont need to tweak this often)
        prominence, height (int) = values for the peak finding function (dont need to tweak this often)
        match_tol (keV) = maximum distance a measured peak can be from an expected peak to continue with analysis.

        Returns:
        matching (dict) = dictionary with a keyword for each expected line, that includes the observed peak location, shift value and error value in keV.
        table_data = same information, but in a nicely written table format for plotting.
        '''
        #calculating the histogram with defined energy bins
        bins = np.arange(0, 80 + self.bin_width, self.bin_width) #keeping a large distrubution to hist over
        hist, edges = np.histogram(cdte_energy, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        #smoothing the histogram to find the peak more easily
        hist_smooth = gaussian_filter1d(hist, smoothing_sigma)

        #finding the peaks
        peaks, props = find_peaks(
            hist_smooth, prominence=prominence, height=height
        )

        #finds the FWHM in keV, as well as the edges
        widths = peak_widths(hist_smooth, peaks, rel_height=0.5)
        fwhm_keV = widths[0] * self.bin_width #NOTE, this must be assumed baseline 0 and gaussian to truly be a FWHM, so its an assumption
        left = widths[2] #left poing of width in energy
        right = widths[3] #right point of width in energy

        #finding the counts inside the FWHM region to compute error on the centroid calculation (the mean)
        peak_counts = []
        for l, r in zip(left, right):
            peak_counts.append(np.sum(hist[int(l):int(r)+1]))
        peak_counts = np.array(peak_counts)
        sigma_centroid = sigma_centroid = (fwhm_keV / 2.355) / np.sqrt(peak_counts / 0.76) #again, assuming a Gaussian distribution
        sigma_total = np.sqrt(self.sigma_bin**2 + sigma_centroid**2) #combining this with the bin error

        #the actual energies of the peaks!
        peak_E = centers[peaks]

        #define what lines should be there
        expected_lines = np.array(self.sourcelines_dict[self.source])
        matched = {}
        table_data = []

        for expected in expected_lines:
            diffs = np.abs(peak_E - expected)
            #only move ahead with lines that fall within the expected match tolerance (5keV usually)
            if np.any(diffs <= match_tol):
                i = np.argmin(diffs) #take the closest peak that is less than 5 keV away
                observed = peak_E[i]
                shift = observed - expected
                err = sigma_total[i]

                matched[expected] = {
                    "observed": observed,
                    "shift": shift,
                    "error": err
                }

                table_data.append([
                    f"{expected:.1f}",
                    f"{observed:.2f}",
                    f"{shift:+.2f}",
                    f"±{err:.2f}"
                ])
            else:
                table_data.append([f"{expected:.1f}", '—', '—', '—'])

        return {"hist": hist, "hist_smooth": hist_smooth, "edges": edges, "peak_energy": peak_E, "expected_lines": expected_lines, "table_data": table_data}

    def find_peak_shifts(self, smoothing_sigma=2, prominence=10, height=5, match_tol=5):
        ''' Starting with the CdTe .fits file, does specified hitnumber masking, calls the _analyze_peaks function and ends with the necessary information for plotting.
        '''
        if self.double:
            cdte_data_ge_al, cdte_data_ge_pt = self.double_hits(self.cdte_data)
        else:
            cdte_data_ge_al, cdte_data_ge_pt = self.single_hits(self.cdte_data)
        
        #doing the shift analysis
        self.al_shift_analysis = self._analyze_peaks(cdte_data_ge_al['al_merged_energy_list'][:,0], smoothing_sigma, prominence, height, match_tol)
        self.pt_shift_analysis = self._analyze_peaks(cdte_data_ge_pt['pt_merged_energy_list'][:,0], smoothing_sigma, prominence, height, match_tol)
        

    def make_spectral_shift_plot(self, axes):
        """
        Plots spectrum, detected peaks, expected peaks, and summary table.

        Input:
        axes = matploblib axis object, for a 1x2 subplot.
        """
        analysis_list = [self.al_shift_analysis, self.pt_shift_analysis]
        side_labels = ['Al', 'Pt']

        for ax, analysis, side_label in zip(axes, analysis_list, side_labels):
            hist = analysis["hist"]
            hist_smooth = analysis["hist_smooth"]
            edges = analysis["edges"]
            peak_E = analysis["peak_energy"]
            expected_lines = analysis["expected_lines"]
            table_data = analysis["table_data"]

            label = "Double Hit Spectrum" if self.double else "Single Hit Spectrum"

            # ---- Spectrum ----
            ax.stairs(hist, edges, label=label)
            ax.stairs(hist_smooth, edges, label="Smoothed", lw=2)

            # ---- Observed peaks ----
            for p in peak_E:
                ax.axvline(p, color="k", alpha=0.4)

            # ---- Expected peaks ----
            for i, Eexp in enumerate(expected_lines):
                ax.axvline(
                    Eexp,
                    color="r",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                    label="Expected Peak" if i == 0 else None,
                )

            ax.set_xlabel("Energy [keV]")
            ax.set_ylabel("Counts")
            ax.legend()

            # ---- Table ----
            col_labels = [
                "Expected [keV]",
                "Observed [keV]",
                "Shift [keV]",
                "Total Error [keV]",
            ]

            table = ax.table(
                cellText=table_data,
                colLabels=col_labels,
                cellLoc="center",
                bbox=[1.05, 0.05, 0.9, 0.5],
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
        return axes

    def make_energyvsshift_plots(self, axes):
        ''' Plots the energy shift (measured - expected) vs. energy for all source lines.

        Input:
        axes = matplotlib axis object, for a 1x2 subplot.
        '''
        labels = ['Al', 'Pt']
        title = "Double Strip Events Only" if self.double else "Single Strip Events Only"
        table_data_list = [self.al_shift_analysis['table_data'], self.pt_shift_analysis['table_data']]
        for ax, table, label in zip(axes, table_data_list, labels):

            E = np.array([float(r[0]) for r in table if r[1] != '—'])
            shift = np.array([float(r[2]) for r in table if r[1] != '—'])
            err = np.array([float(r[3].replace('±', '')) for r in table if r[1] != '—'])

            ax.errorbar(E, shift, yerr=err, fmt='o', capsize=3)
            ax.axhline(0, color='k', ls='--')

            # --- Linear fit ---
            valid = ~np.isnan(shift)
            if np.sum(valid) >= 2:  # need at least 2 points to fit
                slope, intercept = np.polyfit(E[valid], shift[valid], 1)
                fit_line = slope * E + intercept
                ax.plot(E, fit_line, 'b--', lw=1, label='Linear Fit')

                # Annotate fit parameters in upper left corner
                ax.text(
                    0.02, 0.95,
                    f"y = {slope:.2f}x + {intercept:.2f}",
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
                )
            
            ax.set_xlabel("Energy [keV]")
            ax.grid(alpha=0.3)
            ax.set_title(f'{label}, {title}')

        axes[0].set_ylabel("Measured − Expected [keV]")
        return axes
        



class ChargeSharingPlotting:
    """
    End-to-end charge sharing analysis for CdTe detectors:
      - single-clump filtering
      - observed peak finding
      - FWHM-based peak windows
      - multi-strip fraction vs energy per strip pitch
      - plotting
    """
    sourcelines_dict = {
            'Am241': [59.5, 26.3, 13.9, 17.7, 20.7],
            'Am241 and Fe55': [59.5, 26.3, 13.9, 17.7, 20.7, 5.9],
            'Fe55': [5.9],
            'Ba133': [7.8, 11.9, 30.9, 35.1]
        }

    def __init__(self, cdte_data, source_lines, filter_pseudo_triggers=True,
                 bin_width=0.2, smoothing_sigma=2,
                 match_tol=5.0, energy_max=80.0):
        '''
        Inputs:
        cdte_fits (fits.open(cdte_data)[1].data) = CdTe file that has at least gone through process to PATH2.
        source_lines (str) = source line(s) used in test. Will be one of the keywords in 
            sourcelines_dict.
        filter_pseudo_triggers (True/False) = filters out pseudo triggers before analysis.
        bin_width (keV) = size of bin with for energy/spectral binning.
        smoothing_sigma = used for gaussian smoothing of data, usually no need for tweaking this.
        match_tol (keV) = difference between found peak and expected peak that will be allowed when   
           assigning a measured peak to an expected one.
        energy_max = maximum energy used for binning-- I am keeping this a ways above the actual peak so 
            that we don't have any energy buildup at the ends of our range.
        '''
        self.cdte_data = cdte_data
        self.source_lines = self.sourcelines_dict[source_lines]
        self.bin_width = bin_width
        self.smoothing_sigma = smoothing_sigma
        self.match_tol = match_tol
        self.energy_max = energy_max

        #filter out pseudo triggers:
        if filter_pseudo_triggers:
            pseudo_mask = self.cdte_data['flag_pseudo'] == 0
            self.cdte_data = self.cdte_data[pseudo_mask]
        self.calculate_livetime_and_countrates()

        # Results containers
        self.peak_results = {}
        self.multistrip_fractions = {}

        #making it so we only have single clumps for charge sharing analysis:
        self.keep_single_clumps()
        #doing computations on initialization so I only need to care about the plotting
        self.compute_peaks()
        self.compute_all_multistrip_fractions()

    def calculate_livetime_and_countrates(self):
        ''' Calculates the livetime and total count rates for the observation. This is done at the beginning so that I can use global variables for the two livetime plots, and so that the countrate can be included in pedestal plotting (still need to add that in when finalizing the summary info).
        '''
        self.livetime = self.cdte_data['livetime'] * 1e-8 #to get into seconds
        self.total_runtime = np.sum(self.livetime)
        self.total_counts = len(self.cdte_data['ti'])
        self.count_rate = self.total_counts/self.total_runtime

    def keep_single_clumps(self):
        ''' For this analysis, Yixian' kept only events that had a single clump on each side, and I am doing the same to help filter out noise.
        '''
        mask = (
            (self.cdte_data['al_merged_nhit'] == 1) &
            (self.cdte_data['pt_merged_nhit'] == 1)
        )
        self.cdte_data = self.cdte_data[mask]

    #helping functions for finding spectral peak regions
    def _histogram(self, energies):
        ''' Makes histogram of spectra with defined energy bins and energy range (from 0 to self.energy_max)
        '''
        bins = np.arange(0, self.energy_max + self.bin_width, self.bin_width)
        hist, edges = np.histogram(energies, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return hist, edges, centers

    def _find_peaks(self, hist):
        ''' Uses input histogram and makes a smooth version, and finds its peak and width at half max. This will be used for deciding what counts to include when looking at charge sharing for specific peaks.
        '''
        smooth = gaussian_filter1d(hist, self.smoothing_sigma)
        peaks, _ = find_peaks(smooth, prominence=5, height=5)
        widths = peak_widths(smooth, peaks, rel_height=0.5)
        return smooth, peaks, widths

    # getting the peak counts
    def get_peak_counts_per_side(self, side):
        ''' Histograms single clump spectra, smooths and finds its peaks and widths. For each peak that is close to an expected peak, the counts, counting error (sqrt(N)), and peak properties are saved. 

        Note: I am not currently propagating the FWHM error into the fractional peak, but instead treating it like a systematic effect (not a statistical error) because the fraction is dependent on this. Can edit this in the future!

        Input:
        side ('Al' or 'Pt') = side of detector used for getting peak counts.
        '''
        energy_col = f"{side.lower()}_merged_energy_list"
        energies = self.cdte_data[energy_col][:, 0]

        hist, edges, centers = self._histogram(energies)
        smooth, peaks, widths = self._find_peaks(hist)

        fwhm_keV = widths[0] * self.bin_width
        left_bin = widths[2].astype(int)
        right_bin = widths[3].astype(int)
        peak_centers = centers[peaks]

        results = {}

        for expected in self.source_lines:
            diffs = np.abs(peak_centers - expected)

            if np.any(diffs <= self.match_tol): #keeping same 5 keV match tolerance
                idx = np.argmin(diffs)
                observed = peak_centers[idx]
                l, r = left_bin[idx], right_bin[idx]
                counts = np.sum(hist[l:r+1])
                fwhm = fwhm_keV[idx]
            else:
                observed = expected
                counts = np.sum(
                    (energies > expected - 1.0) &
                    (energies < expected + 1.0)
                )
                fwhm = 2.0

            results[expected] = {
                "observed": observed,
                "counts": counts,
                "error": np.sqrt(counts), #basic counting statistics error, this is just used for printing out these values/a sanity check! This error is NOT propagated into the fraction error, because the fraction is dependent on the choice of peak.
                "fwhm": fwhm
            }

        return results, hist, edges, centers, smooth

    def compute_peaks(self):
        ''' Actually computes and saved the peak results/count results for each side of the detector.
        '''
        self.peak_results['Al'] = self.get_peak_counts_per_side('Al')
        self.peak_results['Pt'] = self.get_peak_counts_per_side('Pt')

    # getting the fractions of charge-sharing events for each peak
    def compute_multistrip_fraction(self, side):
        ''' Finds the fractions of multi-strip events for each peak.
        Input:
        Side ('Al' or 'Pt') = side of the detector used to compute the fractions.

        Returns:
        fractions (dict) = dictionary, with an input for each expected peak. For each expected peak, the fraction of multi-strip events (and error) is saved for each strip pitch width. The error is calculated by assuming a binomial random distribution, given that the fractions are fairly large (assumes that we either have single or multi, with some unknown true probability p). The error is then largest with fractions near 0.5.
        '''
        peak_results = self.peak_results[side][0]
        fractions = {}

        if side == 'Pt':
            channel = self.cdte_data['pt_merged_position_list'][:, 0]
            pitches = {100: (4, 27), 80: (28, 47), 60: (48, 63)}
            nhit_col = 'pt_nhit'
            energy_col = 'pt_merged_energy_list'
        else:
            channel = self.cdte_data['al_merged_position_list'][:, 0]
            pitches = {60: (64, 79), 80: (80, 99), 100: (100, 123)}
            nhit_col = 'al_nhit'
            energy_col = 'al_merged_energy_list'

        for expected, info in peak_results.items():
            obs = info['observed']
            fwhm = info['fwhm']

            #keep only the energy range that is within the FWHM of the observed peak
            mask_energy = (
                (self.cdte_data[energy_col][:, 0] > obs - fwhm/2) &
                (self.cdte_data[energy_col][:, 0] < obs + fwhm/2)
            )

            fractions[expected] = {}

            for pitch, (cmin, cmax) in pitches.items():
                #filter out only the events within the specific pitch region
                mask_channel = (channel >= cmin) & (channel <= cmax)
                mask = mask_energy & mask_channel

                #get the total counts within the correct energy range and strip
                N = np.sum(mask)
                if N == 0:
                    frac, err = np.nan, np.nan
                else:
                    #the number of counts that are multi-strip within the mask
                    N_multi = np.sum(self.cdte_data[nhit_col][mask] > 1)
                    frac = N_multi / N
                    err = np.sqrt(frac * (1 - frac) / N) #comes from assuming a binomial distribution

                fractions[expected][pitch] = (frac, err)
        return fractions

    def compute_all_multistrip_fractions(self):
        ''' Actually computes the fractions.
        '''
        self.multistrip_fractions['Al'] = self.compute_multistrip_fraction('Al')
        self.multistrip_fractions['Pt'] = self.compute_multistrip_fraction('Pt')

    # Plots!!
    def plot_spectrum(self, ax, e_range):
        ''' Plots the spectrum, with shaded regions denoting where the counts are being included to compute the charge sharing fractions. This is mostly a sanity check plot, to make sure the data is clean enough to make the charge sharing plot worth interpeting.

        Input:
        ax = matplotlib axis object.
        '''
        for side, color in zip(['Al', 'Pt'], ['blue', 'green']):
            peak_info, hist, edges, centers, smooth = self.peak_results[side]

            ax.stairs(hist, edges, color=color, alpha=0.3,
                      label=f'{side} Histogram')
            ax.plot(centers, smooth, color=color, lw=1.5,
                    label=f'{side} Smoothed')

            for info in peak_info.values():
                ax.axvspan(info['observed'] - info['fwhm']/2,
                           info['observed'] + info['fwhm']/2,
                           color=color, alpha=0.1)

        for i, E in enumerate(self.source_lines):
            ax.axvline(E, color='red', ls='--', alpha=0.7,
                       label='Expected Peaks' if i == 0 else None)

        ax.set_xlabel('Energy [keV]')
        ax.set_ylabel('Counts')
        ax.set_title('Single Clump Merged Energy Spectrum with Observed Peak Windows')
        ax.set_xlim(e_range)
        ax.legend()
        return ax

    def plot_multistrip_fraction(self, axes):
        ''' Plots the fraction of multi-strip single-clump events for both sides of the detector, for each strip pitch width.

        Input:
        axes = matplotlib axis object, for a 2x1 subplot.
        '''
        colors = {60: 'blue', 80: 'green', 100: 'orange'}
        labels = {60: '60 µm', 80: '80 µm', 100: '100 µm'}

        for ax, side in zip(axes, ['Al', 'Pt']):
            fractions = self.multistrip_fractions[side]

            for pitch in [60, 80, 100]:
                energies, fracs, errs = [], [], []

                for E in sorted(fractions):
                    if pitch in fractions[E]:
                        f, e = fractions[E][pitch]
                        if not np.isnan(f):
                            energies.append(E)
                            fracs.append(f)
                            errs.append(e)

                ax.errorbar(energies, fracs, yerr=errs,
                            fmt='o-', color=colors[pitch],
                            label=labels[pitch])

            ax.set_title(f'{side} Side')
            ax.set_xlabel('Energy [keV]')
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.legend()

        axes[0].set_ylabel('Fraction of Multi-Strip Hits')
        return axes



########################################################################################################

                                  ##### THE GRAVEYARD #####

########################################################################################################

#     def find_differences_and_make_plot(self, energies, ax, plot_title, source, smoothing_sigma, prominence, height, match_tol, double):
#         # Histogram the energies
#         bins = np.linspace(0, 80, 800)
#         hist, bin_edges = np.histogram(energies, bins=bins)
#         bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

#         # Smooth histogram to suppress noise
#         hist_smooth = gaussian_filter1d(hist, sigma=smoothing_sigma)

#         # Find peaks with tuned sensitivity
#         peaks, properties = find_peaks(hist_smooth, prominence=prominence, height=height)
#         peak_energies = bin_centers[peaks]

#         # Match to expected lines within a tolerance
#         expected_lines = np.array(self.sourcelines_dict[source])
#         matched_peaks = {}

#         for expected in expected_lines:
#             diffs = np.abs(peak_energies - expected)
#             if np.any(diffs <= match_tol):
#                 best_idx = np.argmin(diffs)
#                 observed = peak_energies[best_idx]
#                 shift = observed - expected
#                 matched_peaks[expected] = {
#                     "observed": observed,
#                     "shift": shift,
#                     "prominence": properties['prominences'][best_idx],
#                     "height": properties['peak_heights'][best_idx]
#                 }
#             else:
#                 matched_peaks[expected] = None  # No matching peak found

#         #do a little plot to look at the smoothed histogram!
#         if double:
#             ax.stairs(hist, bin_edges, label='Double Hit Spectra')
#         else:
#             ax.stairs(hist, bin_edges, label='Single Hit Spectra')
#         ax.stairs(hist_smooth, bin_edges, label='Gaussian Smoothed Spectra')
#         for p in peak_energies:
#             ax.axvline(p, c='k')
#         ax.set_title(plot_title, fontsize=16)
#         # Build table data
#         table_data = []
#         for expected in sorted(self.sourcelines_dict[source]):
#             result = matched_peaks[expected]
#             if result:
#                 observed = f"{result['observed']:.2f}"
#                 shift = f"{result['shift']:+.2f}"
#             else:
#                 observed = '—'
#                 shift = '—'
#             table_data.append([f"{expected:.1f}", observed, shift])

#         # Add table to plot
#         col_labels = ["Expected Peak [keV]", "Observed Peak [keV]", "Shift [keV]"]
#         table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center',
#                         loc='right', bbox=[1.05, 0, 0.8, 0.5], fontsize=14)  # Adjust bbox for spacing
#         table.auto_set_font_size(False)
#         ax.set_xlabel('Energy [keV]', fontsize=14)
#         if double:
#             ax.set_ylabel('Double Strip Counts', fontsize=14)
#         else:
#             ax.set_ylabel('Single Strip Counts', fontsize=14)
#         ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

#         # # Report
#         # print(f"Matched peaks for {source}:")
#         # for expected, result in matched_peaks.items():
#         #     if result:
#         #         print(f"Expected: {expected:.1f} keV → Observed: {result['observed']:.2f} keV (Shift: {result['shift']:+.2f}, Prom: {result['prominence']:.1f}, Height: {result['height']:.1f})")
#         #     else:
#         #         print(f"Expected: {expected:.1f} keV → No peak found within ±{match_tol} keV")

#         return matched_peaks, ax, table_data

#     def make_energyvsshift_plots(self, table_data_list, double):

#         datalist = ['Al Data', 'Pt Data']
#         fig, axes = plt.subplots(1, len(table_data_list), figsize=(16, 8), sharey=True)

#         if len(table_data_list) == 1:
#             axes = [axes]  # make iterable

#         for idx, (table_data, ax) in enumerate(zip(table_data_list, axes)):
#             # Extract expected, observed, shifts
#             expected = [float(item[0]) for item in table_data]
#             observed = [float(item[1]) if item[1] != '—' else np.nan for item in table_data]
#             shifts = [float(item[2]) if item[2] != '—' else np.nan for item in table_data]

#             print(f"[Dataset {idx}] Expected: {expected}")
#             print(f"[Dataset {idx}] Observed: {observed}")

#             # Plot Expected vs Observed
#             ax.plot(expected, observed, 'go', label='Observed vs Expected')
#             ax.plot([0, 65], [0, 65], 'k--')

#             # Labels and title
#             ax.set_xlabel('Expected Energy [keV]', fontsize=14)
#             if idx == 0:
#                 ax.set_ylabel('Observed Energy [keV]', fontsize=14)

#             ax.set_title(f'{self.cdte} Peak Shifts for {datalist[idx]}', fontsize=16)
#             ax.set_xlim(0, 65)
#             ax.set_ylim(0, 65)

#             # Table data for the subplot
#             col_labels = ["Expected\n[keV]", "Observed\n[keV]", "Shift\n[keV]"]
#             table_data_for_table = [
#                 [f"{e:.1f}", f"{o:.2f}" if not np.isnan(o) else '—', f"{s:.2f}" if not np.isnan(s) else '—']
#                 for e, o, s in zip(expected, observed, shifts)
#             ]

#             table = ax.table(
#                 cellText=table_data_for_table,
#                 colLabels=col_labels,
#                 cellLoc='center',
#                 loc='upper left',
#                 bbox=[0.02, 0.65, 0.6, 0.33]
#             )
#             table.auto_set_font_size(False)

#             # Linear fit
#             if not all(np.isnan(observed)):
#                 slope, intercept = np.polyfit(np.array(expected), np.array(observed), 1)
#                 ax.plot(np.array([0, 65]), slope * np.array([0, 65]) + intercept, 'b-', lw=1,
#                         label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')
#                 ax.legend(loc='lower right')
#                 print(f"[Dataset {idx}] Slope: {slope:.2f}, Intercept: {intercept:.2f}")
#         if double:
#             plt.suptitle(f'Double Strip Events Only, {self.cdte}')
#         else:
#             plt.suptitle(f'Single Strip Events Only, {self.cdte}')
#         plt.tight_layout()
#         plt.savefig(self.savename2, dpi=250, bbox_inches='tight')

# class ChargeSharing:
#     ''' Takes in a list of CdTe FITS paths from 60-200 V runs. The goal of this class is to look at how the charge sharing is affected by applying different voltages to the detector.
    
#     input:
#     -------------------------------
#     cdte_fits_list = [cdte_0V_fits_path, cdte_60V_fits_path, cdte_200_fits_path]
#     '''

#     def __init__(self, cdte_fits_list, cdte):
#         self.cdte_data_60V = fits.open(cdte_fits_list[0])[1].data
#         self.cdte_data_100V = fits.open(cdte_fits_list[1])[1].data
#         self.cdte_data_200V = fits.open(cdte_fits_list[2])[1].data
#         self.cdte = cdte

#     def keep_single_clumps(self, cdte_data):
#         mask_al = cdte_data['al_merged_nhit'] == 1
#         cdte_data = cdte_data[mask_al]
#         mask_pt = cdte_data['pt_merged_nhit'] == 1
#         cdte_data = cdte_data[mask_pt]
#         return cdte_data
    
#     def find_energy_diffs(self, cdte_data):
#         '''Finds the energy differences for specific CdTe. We are only looking at specific spectral lines,
#         so make sure that the spectral line is filtered out first!'''
#         energy_diff = np.abs(cdte_data['pt_merged_energy_list'][:,0] - cdte_data['al_merged_energy_list'][:,0])
#         #one issue is that this looks to be wayyy too large of a range
#         return energy_diff
    
#     def filter_for_spectral_line(self, cdte_data, spectral_line, range):
#         '''Filters the CdTe data for a specific line', given a specified keV range.'''
#         lowest_energy = spectral_line - range
#         highest_energy = spectral_line + range
#         cdte_data = cdte_data[(cdte_data['al_merged_energy_list'][:,0] > lowest_energy) & (cdte_data['al_merged_energy_list'][:,0] < highest_energy)]
#         cdte_data = cdte_data[(cdte_data['pt_merged_energy_list'][:,0] > lowest_energy) & (cdte_data['pt_merged_energy_list'][:,0] < highest_energy)]
#         return cdte_data
    
#     def find_ratios(self, cdte_data, energy_diffs, binnumber, goal_hitnumber):
#         '''Finds the ratios for binned energy differences, given a specified number of bins. Hitnumber defines the amount of hits you want looked at for the ratio (single=1, double=2, etc)'''
#         energy_bins = np.linspace(np.min(energy_diffs), np.max(energy_diffs), binnumber)
#         energy_bin_indices = np.digitize(energy_diffs, energy_bins)
#         ratios_al = []
#         ratios_pt = []
#         nhit_al, nhit_pt = cdte_data['al_nhit'], cdte_data['pt_nhit']
#         for nhit, rat in zip([nhit_al, nhit_pt], [ratios_al, ratios_pt]):
#             for i in range(1, len(energy_bins)):
#                 #checking to see which triggers are within the bin
#                 in_bin = energy_bin_indices == i
#                 triggers_in_bin = nhit[in_bin]
#                 if len(triggers_in_bin) > 0: 
#                     #checking to see how many of those triggers is the hitnum we want
#                     right_hitnum = np.sum(triggers_in_bin == goal_hitnumber)
#                     rat.append(right_hitnum/len(triggers_in_bin))
#                 else:
#                     rat.append(0)
#         return ratios_al, ratios_pt, energy_bins

#     def do_single_voltage(self, cdte_data, spectral_line, binnumber, goal_hitnumber, ax1, ax2, voltage):
#         single_clump = self.keep_single_clumps(cdte_data)
#         lineemission = self.filter_for_spectral_line(single_clump, spectral_line, 4)
#         energy_diffs = self.find_energy_diffs(lineemission)
#         ratios_al, ratios_pt, energy_bins = self.find_ratios(lineemission, energy_diffs, binnumber, goal_hitnumber)
#         ax1.stairs(ratios_pt, energy_bins, label=voltage)
#         ax2.stairs(ratios_al, energy_bins, label=voltage)

#     def make_plot(self, spectral_line, binnumber, goal_hitnumber, plottitle, savefile):
#         '''Makes a plot that looks at the Al and Pt ratios vs. depth of interaction for all three voltages.'''
#         fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
#         data_list = [self.cdte_data_60V, self.cdte_data_100V, self.cdte_data_200V]
#         voltage_list = ['60', '100', '200']
#         for d, voltage in zip(data_list, voltage_list):
#             self.do_single_voltage(d, spectral_line, binnumber, goal_hitnumber, ax1, ax2, voltage)
#         ax1.set_ylabel(f'Ratio of {goal_hitnumber} Strip Events')
#         ax1.set_xlabel('|Pt Energy - Al Energy| [keV]')
#         ax1.set_title('Pt (Cathode) Side')
#         ax2.set_xlabel('|Pt Energy - Al Energy| [keV]')
#         ax2.set_title('Al (Anode) side')
#         ax1.legend()
#         ax2.legend()
#         plt.suptitle(plottitle)
#         plt.savefig(savefile, dpi=250, bbox_inches='tight')

