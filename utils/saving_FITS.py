#standard packages
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st
import numpy as np
from astropy.table import Table
from astropy.io import fits

#local packages
from cdte_tools_py.io.load_raw import load_raw
from cdte_tools_py.calibration.path1 import get_path1
from cdte_tools_py.calibration.path2 import get_path2
from cdte_tools_py.calibration.path3 import get_path3
from cdte_tools_py.io.fits_tools import save_fits

class SaveFITSFromRaw:
    '''Simple class to save a FITS after putting data through path1 REDO THIS'''

    def __init__(self, raw_file, cdte_de, save_name, frame_data=False):
        self.raw_file = raw_file
        self.cdte_de = cdte_de
        self.frame_data = frame_data
        self.save_name = save_name

    def get_fits_path1(self):
        ''' Loads the raw file and creates a FITS file that has been through path 1. 
        '''
        #self.cdte_fits = load_raw(file=self.raw_file, cdte_det=self.cdte_de, frame_data=self.frame_data)
        if isinstance(self.raw_file, str):
            self.cdte_fits = load_raw(file=self.raw_file, cdte_det=self.cdte_de, frame_data=self.frame_data)
        elif isinstance(self.raw_file, bytes):
            self.cdte_fits = load_raw(raw_data=self.raw_file, cdte_det=self.cdte_de, frame_data=self.frame_data)
        print('load raw completed. now starting path 1')
        self.cdte_fits = get_path1(self.cdte_fits)

    def make_manual_adc_cmn(self):
        '''
        Uses the index and common mode value to create an ADC common mode subtracted column, that doesn't get rid of negative values.

        Input is a .FITS file that has been through path1. 
        Returns: .FITS file with additional columns "adc_cmn_pt_manual" and "adc_cmn_al_manual".'''
        self.path1_data = Table(self.cdte_fits[1].data)

        #separating out the common modes for each trigger between ASICS
        asic0_cmn_arr = self.path1_data['cmn_pt'][:, 0].copy()
        asic1_cmn_arr = self.path1_data['cmn_pt'][:, 1].copy()

        #making a separate ndarray in the same shape as 'adc_pt', but with the respective common modes.
        adc_pt_copy = self.path1_data['adc_pt'].copy()
        cmn_pt = np.where(self.path1_data['index_pt'] < 64, asic0_cmn_arr[:, None], adc_pt_copy)
        cmn_pt = np.where((63 < self.path1_data['index_pt']) & (self.path1_data['index_pt'] < 128), asic1_cmn_arr[:, None], cmn_pt)

        #making a new column with common mode subtracted adc values
        self.path1_data['adc_cmn_pt_manual'] = self.path1_data['adc_pt'] - cmn_pt

        #doing it again for aluminum!
        asic3_cmn_arr = self.path1_data['cmn_al'][:, 0].copy()
        asic4_cmn_arr = self.path1_data['cmn_al'][:, 1].copy()
        adc_al_copy = self.path1_data['adc_al'].copy()
        cmn_al = np.where(self.path1_data['index_al'] < 64, asic3_cmn_arr[:, None], adc_al_copy)
        cmn_al = np.where((63 < self.path1_data['index_al']) & (self.path1_data['index_al'] < 128), asic4_cmn_arr[:, None], cmn_al)
        self.path1_data['adc_cmn_al_manual'] = self.path1_data['adc_al'] - cmn_al

    def save_cdte_fits_path1(self):
        '''Save the few CdTe Path1 FITS file, with the manual adc cmn-subtracted arrays included.'''
        print(self.save_name)
        path1_savename = self.save_name + '_PATH1.fits'
        self.path1_fits = fits.HDUList([self.cdte_fits[0], fits.BinTableHDU(data=self.path1_data)])
        save_fits(self.path1_fits, path1_savename, overwrite=True)
        print(f'fits file saved as: {path1_savename}')

    def save_cdte_fits_path2(self):
        ''' Gets Path2 and saves as a FITS file.'''
        self.path2_fits = get_path2(self.path1_fits)
        path2_savename = self.save_name + '_PATH2.fits'
        save_fits(self.path2_fits, path2_savename, overwrite=True)
        print(f'fits file saved as: {path2_savename}')

    def save_cdte_fits_path3(self):
        ''' Gets Path3 and saves as a FITS file.'''
        self.path3_fits = get_path3(self.path2_fits)
        path3_savename = self.save_name + '_PATH3.fits'
        save_fits(self.path3_fits, path3_savename, overwrite=True)
        print(f'fits file saved as: {path3_savename}')