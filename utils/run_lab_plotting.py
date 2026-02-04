#importing all necessary packges
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from utils import lab_plotting as lp
# import importlib
# importlib.reload(lp)

def write_cdte_run_txt(filename, sections):
    """
    Writes a readable run summary .txt file from a set of dictionaries.

    Inputs:
    filename (str) = name of file to save

    sections : dict
    """

    section_order = [
        "Summary",
        "PATH1",
        "PATH2",
        "Gain Shifting",
        "Charge Sharing"
    ]

    with open(filename, "w") as f:
        f.write("CdTe Run Report\n")
        f.write("================\n\n")

        for section_name in section_order:
            if section_name not in sections:
                continue  # skip missing ones

            d = sections[section_name]

            f.write(f"--- {section_name} ---\n")

            for key, value in d.items():
                f.write(f"{key:<20}: {value}\n")

            f.write("\n")  

    print(f"Run report written to {filename}")

def make_path1_plots(cdte_data, summary_dict, path1_plotconfigs):

    plotter = lp.Path1Plots(cdte_data, filter_pseudo_triggers=path1_plotconfigs["FilterPseudo"], keep_only_single_clumps=path1_plotconfigs["KeepSingleClump"])

    info_text = (
        f"Source: {summary_dict['Source']}\n"
        f"Run #: {summary_dict['RunNumber']}\n"
        f"Voltage: {summary_dict['Voltage']}\n"
        f"Temp: {summary_dict['Temperature']}\n"
        f"Readout Mode: {summary_dict['ReadoutMode']}\n"
        f"Count Rate: {plotter.count_rate:.1f}ct/s\n"
        f"Total Livetime: {plotter.total_runtime:.1f} s\n"
        f"Excluding Pseudo Triggers: {path1_plotconfigs['FilterPseudo']}\n"
        f"Single Clumps Only: {path1_plotconfigs['KeepSingleClump']}"
        )  
    
    def add_infotext(ax, colorbar=False):
        return ax.text(
        1.28 if colorbar else 1.02, 0.95,  # x, y in axes coordinates
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    title = f'{summary_dict["CdTe"]}, {summary_dict["Location"]}, {summary_dict["Date"]}'
    savename = os.path.join(summary_dict["SaveName"], f'run{summary_dict["RunNumber"]}')
    os.makedirs(summary_dict["SaveName"], exist_ok=True)

    #ADC Image Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plotter.make_image(ax)
    add_infotext(ax)
    ax.set_title(title)
    plt.savefig(f'{savename}_adcimage.png', dpi=250, bbox_inches='tight')

    #Pedestal Plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    plotter.plot_pedestal(axs[0], axs[1], path1_plotconfigs["PedestalTopRange"])
    plt.suptitle(title)
    fig = plt.gcf() 
    fig.colorbar(plotter.pt[3], ax=axs[0], orientation='vertical')
    fig.colorbar(plotter.al[3], ax=axs[1], orientation='vertical')
    add_infotext(axs[1], colorbar=True)
    plt.savefig(f'{savename}_cmnspectrograph.png', dpi=250, bbox_inches='tight')

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    plotter.plot_pedestal(axs[0], axs[1], path1_plotconfigs["PedestalTopRange"], cmn_subtracted=False)
    plt.suptitle(title)
    fig = plt.gcf() 
    fig.colorbar(plotter.pt[3], ax=axs[0], orientation='vertical')
    fig.colorbar(plotter.al[3], ax=axs[1], orientation='vertical')
    add_infotext(axs[1], colorbar=True)
    plt.savefig(f'{savename}_nocmnspectrograph.png', dpi=250, bbox_inches='tight')

    #timing plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    plotter.time_plots(axs[0], axs[1])
    plt.suptitle(title)
    add_infotext(axs[1])
    plt.savefig(f'{savename}_time.png', dpi=250, bbox_inches='tight')

    #livetime histogram
    fig, ax = plt.subplots(1,1, figsize=(8, 6))
    plotter.livetime_histogram(ax, path1_plotconfigs["LivetimeTopRange"])
    ax.set_title(title)
    add_infotext(ax)
    plt.savefig(f'{savename}_livetimehist.png', dpi=250, bbox_inches='tight')

    #common mode vs livetime
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    plotter.cmnmode_vs_livetime(axes, path1_plotconfigs["LivetimeTopRange"], path1_plotconfigs["CommonModeRange"])
    plt.suptitle(title)
    add_infotext(axes[0, 1], colorbar=True)
    plt.savefig(f'{savename}_adcvslivetime.png', dpi=250, bbox_inches='tight')
    
def make_path2_plots(cdte_data, summary_dict, path2_plotconfigs):

    plotter = lp.Path2Plots(cdte_data, summary_dict["Source"], filter_pseudo_triggers=path2_plotconfigs["FilterPseudo"], keep_only_single_clumps=path2_plotconfigs["KeepSingleClump"])

    info_text = (
        f"Source: {summary_dict['Source']}\n"
        f"Run #: {summary_dict['RunNumber']}\n"
        f"Voltage: {summary_dict['Voltage']}\n"
        f"Temp: {summary_dict['Temperature']}\n"
        f"Readout Mode: {summary_dict['ReadoutMode']}\n"
        f"Count Rate: {plotter.count_rate:.1f}ct/s\n"
        f"Total Livetime: {plotter.total_runtime:.1f} s\n"
        f"Excluding Pseudo Triggers: {path2_plotconfigs['FilterPseudo']}\n"
        f"Single Clumps Only: {path2_plotconfigs['KeepSingleClump']}"
        )  
    
    def add_infotext(ax, colorbar=False):
        return ax.text(
        1.28 if colorbar else 1.02, 0.95,  # x, y in axes coordinates
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    title = f'{summary_dict["CdTe"]}, {summary_dict["Location"]}, {summary_dict["Date"]}'
    savename = os.path.join(summary_dict["SaveName"], f'run{summary_dict["RunNumber"]}')
    os.makedirs(summary_dict["SaveName"], exist_ok=True)

    #Make single hit spectra
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plotter.make_spectra(ax, path2_plotconfigs["SpectralEnergyRange"], single_hit=True)
    plt.suptitle(title)
    add_infotext(ax)
    plt.savefig(f'{savename}_singlehitspectra.png', dpi=250, bbox_inches='tight')

    #Make double hit spectra
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plotter.make_spectra(ax, path2_plotconfigs["SpectralEnergyRange"], double_hit=True)
    plt.suptitle(title)
    add_infotext(ax)
    plt.savefig(f'{savename}_doublehitspectra_nogaplosscorrection.png', dpi=250, bbox_inches='tight')

    #all event spectra
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plotter.make_spectra(ax, path2_plotconfigs["SpectralEnergyRange"])
    plt.suptitle(title)
    add_infotext(ax)
    plt.savefig(f'{savename}_allhitspectra.png', dpi=250, bbox_inches='tight')

    # #energy vs livetime 2D histogram- Single hit
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    # plotter.energy_vs_livetime(axes, summary_dict["Source"], path2_plotconfigs["LivetimeTopRange"], path2_plotconfigs["SpectralEnergyRange"], single_hit=True)
    # plt.suptitle(title)
    # add_infotext(ax, colorbar=True)
    # plt.savefig(f'{savename}_singlehit_evslivetime.png', dpi=250, bbox_inches='tight')

    # #energy vs livetime 2D histogram- Double hit
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    # plotter.energy_vs_livetime(axes, summary_dict["Source"], path2_plotconfigs["LivetimeTopRange"], path2_plotconfigs["SpectralEnergyRange"], double_hit=True)
    # plt.suptitle(title)
    # add_infotext(ax, colorbar=True)
    # plt.savefig(f'{savename}_doublehit_evslivetime.png', dpi=250, bbox_inches='tight')

    #energy vs livetime 2D histogram- All hits. Only doing this one to get the correct livetime
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    plotter.energy_vs_livetime(axes, summary_dict["Source"], path2_plotconfigs["LivetimeTopRange"], path2_plotconfigs["SpectralEnergyRange"])
    plt.suptitle(title)
    add_infotext(axes[1], colorbar=True)
    plt.savefig(f'{savename}_allevents_evslivetime.png', dpi=250, bbox_inches='tight')

def make_gainshit_plots(cdte_data, summary_dict, gain_plotconfigs):

    plotter = lp.GainShiftingPlotting(cdte_data, summary_dict["Source"], filter_pseudo_triggers=gain_plotconfigs["FilterPseudo"], keep_only_single_clumps=gain_plotconfigs["KeepSingleClump"], double=False)

    info_text = (
        f"Source: {summary_dict['Source']}\n"
        f"Run #: {summary_dict['RunNumber']}\n"
        f"Voltage: {summary_dict['Voltage']}\n"
        f"Temp: {summary_dict['Temperature']}\n"
        f"Readout Mode: {summary_dict['ReadoutMode']}\n"
        f"Count Rate: {plotter.count_rate:.1f}ct/s\n"
        f"Total Livetime: {plotter.total_runtime:.1f} s\n"
        f"Excluding Pseudo Triggers: {gain_plotconfigs['FilterPseudo']}\n"
        f"Single Clumps Only: {gain_plotconfigs['KeepSingleClump']}"
        )  
    
    def add_infotext(ax, colorbar=False):
        return ax.text(
        1.28 if colorbar else 1.02, 0.95,  # x, y in axes coordinates
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    title = f'{summary_dict["CdTe"]}, {summary_dict["Location"]}, {summary_dict["Date"]}'
    savename = os.path.join(summary_dict["SaveName"], f'run{summary_dict["RunNumber"]}')
    os.makedirs(summary_dict["SaveName"], exist_ok=True)
    
    #gain shift plots - doing only single hit rn
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 14))
    plotter.make_spectral_shift_plot([ax1, ax2])
    plt.suptitle(title)
    add_infotext(ax1)
    plt.savefig(f'{savename}_gainshifts.png', dpi=250, bbox_inches='tight')

    #energy shift vs. energy plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    plotter.make_energyvsshift_plots(axes)
    plt.suptitle(title)
    add_infotext(axes[1])
    plt.savefig(f'{savename}_evsshift.png', dpi=250, bbox_inches='tight')

def make_chargesharing_plots(cdte_data, summary_dict, charge_plotconfigs):
    
    plotter = lp.ChargeSharingPlotting(cdte_data, summary_dict["Source"], filter_pseudo_triggers=charge_plotconfigs["FilterPseudo"])

    info_text = (
        f"Source: {summary_dict['Source']}\n"
        f"Run #: {summary_dict['RunNumber']}\n"
        f"Voltage: {summary_dict['Voltage']}\n"
        f"Temp: {summary_dict['Temperature']}\n"
        f"Readout Mode: {summary_dict['ReadoutMode']}\n"
        f"Count Rate: {plotter.count_rate:.1f}ct/s\n"
        f"Total Livetime: {plotter.total_runtime:.1f} s\n"
        f"Excluding Pseudo Triggers: {charge_plotconfigs['FilterPseudo']}\n"
        )  
    
    def add_infotext(ax, colorbar=False):
        return ax.text(
        1.28 if colorbar else 1.02, 0.95,  # x, y in axes coordinates
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    title = f'{summary_dict["CdTe"]}, {summary_dict["Location"]}, {summary_dict["Date"]}'
    savename = os.path.join(summary_dict["SaveName"], f'run{summary_dict["RunNumber"]}')
    os.makedirs(summary_dict["SaveName"], exist_ok=True)

    #spectra with shading
    fig, ax = plt.subplots(figsize=(12, 6))
    plotter.plot_spectrum(ax, charge_plotconfigs["EnergyRange"])
    plt.suptitle(title)
    add_infotext(ax)
    plt.savefig(f'{savename}_chargesharingspectra.png', dpi=250, bbox_inches='tight')

    #charge strip fractions
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    plotter.plot_multistrip_fraction(axes)
    plt.suptitle(title)
    add_infotext(axes[1])
    plt.savefig(f'{savename}_chargesharingfractions.png', dpi=250, bbox_inches='tight')



