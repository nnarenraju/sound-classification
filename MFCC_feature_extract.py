#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Project Details:
    MFCC_feature_extract.py 
    Extracting features from signals using MFCC
    Implements for entire training/testing set

Created on Sat Sep 30 22:12:31 2017

__author__      = "nnarenraju"
__copyright__   = "Copyright 2017, Severity Classification"
__credits__     = "nnarenraju"
__license__     = "Apache License 2.0"
__version__     = "1.0.1"
__maintainer__  = "nnarenraju"
__email__       = "nnarenraju@gmail.com"
__status__      = "inUsage"

Github Repository: "https://github.com/nnarenraju/sound-classification"

"""

import math
import pickle
import numpy as np
import scipy.special
import scipy.fftpack
import scipy.io.wavfile
import matplotlib.pyplot as plt

from astropy.io import ascii
from scipy.interpolate import spline

def kaiser_window(alpha, N, visualise = False):
    """ Returns a kaiser window function """
    
    def _calculate_beta(alpha):
        """ Calculates value of Beta """
        if alpha>50.0:
            return 0.1102*(alpha-8.7)
        elif alpha>=21.0 and alpha<=50.0:
            return 0.5842*math.pow(alpha-21.0, 0.4)+0.07886*(alpha-21.0)
        elif alpha<21.0:
            return 0
        else:
            raise ValueError("Invalid alpha value")
    
    def _visualise(w_n, beta):
        """ Visualise the kaiser window function """
        # Time Domain Response of Kaiser Window
        x_range = np.array(range(0, len(w_n)))
        xnew = np.linspace(x_range.min(), x_range.max(), 300)
        power_smooth = spline(x_range, w_n, xnew)
        plt.figure(figsize=(12,8))
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.plot(xnew, power_smooth, label="Interpolated")
        plt.plot(x_range, w_n, label="Actual")
        plt.grid(True)
        plt.legend()
        plt.title(r"Kaiser Window Function in Time Domain ($\beta$={0})".format(beta))
        plt.show()
    
    beta = _calculate_beta(alpha)
    N=int(N)
    frange = list(float(a) for a in range(0, N+1))
    N = float(N)
    num=map(lambda n: beta*math.pow(1.0-math.pow((n-N/2)/(N/2), 2), 0.5), frange)
    num_bessel=scipy.special.i0(num)
    den_bessel=scipy.special.i0(beta)
    w_n = map(lambda n: n/den_bessel, num_bessel)
    if visualise: 
        _visualise(w_n, beta)
    return w_n
    
def MFCC_feature_extract(frame, frame_length, sample_rate, nfft=512):
    """ Computes Mel Frequency Cepstral Coefficients for given signal """
        
    def _perform_fft(frame, n=nfft):
        """ Performs one-dimensional discrete fourier transform using FFT """
        # Performs 512 point FFT and keep first 256 points
        # Computes the absolute values of each complex/real number output
        Si_k=np.fft.fft(frame, n=n)
        # Change 256 points to req generic form
        Si_k=map(np.absolute, Si_k[0:256])
        return Si_k
    
    def _convert2mel(frequency):
        """ Converts Hertz to Mels """
        return 1125.0*np.log(1.0+frequency/700.0)
    
    def _convert2hz(mels):
        """ Converts Mels to Hertz """
        return 700.0*(np.exp((mels/1125.0))-1.0)
    
    def _freq2bin(hi, sample_rate):
        """ Converts frequency to fft bin numbers """
        return math.floor((nfft+1)*hi/sample_rate)
    
    def _calculate_hm_k(f_m_prev, f_m, f_m_next):
        """ Calculates the Mel-filterbank """
        hm_k=[]
        for k in range(nfft):
            if k<f_m_prev:
                hm_k.append(0)
            elif f_m_prev<=k and k<=f_m:
                hm_k.append((k-f_m_prev)/(f_m-f_m_prev))
            elif f_m<=k and k<=f_m_next:
                hm_k.append((f_m_next-k)/(f_m_next-f_m))
            elif k>f_m_next:
                hm_k.append(0)
            else:
                raise ValueError("Invalid value encountered in filterbank calculation")
                
        #Sanity Check 2
        if len(hm_k)!=nfft:
            raise ValueError("Inconsistant filterbank vector")
        return hm_k
    
    # Applying Kaiser Window
    w_n = kaiser_window(alpha=51, N=frame_length)
    kaiser_frame=[a*b for a,b in zip(np.array(frame), np.array(w_n))]
    
    # One set of 12 MFCC coefficients are extracted for each frame
    #Perform Discrete Fourier Transform on each frame and obtain absolute values
    abs_fft_frame=_perform_fft(kaiser_frame)
    
    # Get Generic expression for upper and lower frequencies
    upper_freq=8000
    lower_freq=300
    num_banks=26 #Choose an even number
    # Convert Upper and Lower Frequency to Mels
    upper_mel,lower_mel=_convert2mel(upper_freq),_convert2mel(lower_freq)
    # Calculate the filterbanks
    m_i=np.linspace(lower_mel, upper_mel, num=num_banks+2)
    h_i=map(_convert2hz, m_i)
    
    # Sanity Check 1
    if abs(int(h_i[0])-lower_freq)>2 or abs(int(h_i[-1])-upper_freq)>2:
        raise ValueError("Discrepancy in upper/lower filterbank frequency")
    
    # Change number of points of FFT to requried number
    fft_bin=[]
    for bank in h_i:
        fft_bin.append(_freq2bin(bank, sample_rate))
        
    # Computing the Mel-spaced Filterbanks
    # We Obtain 26 vectors of length 512, mostly filled with zeroes
    final_filterbank=[]
    for i in range(1,len(fft_bin)-1):
        final_filterbank.append(_calculate_hm_k(fft_bin[i-1], fft_bin[i], fft_bin[i+1]))
    
    # Sanity Check 3
    if len(final_filterbank)!=num_banks:
        raise ValueError("Inconsistant final filterbank length")
    
    # Computing Log Filterbank Energies
    filterbank_energy=[]
    final_energy=[]
    temp_energy=[]
    # Number of Filterbanks = num_banks; default=26
    for filterbank in final_filterbank:
        # Multiplication of filterbank(Triangular funciton) with FFT Output
        temp_energy.append([a*b for a,b in zip(np.array(filterbank), np.array(abs_fft_frame))])
    for temp in temp_energy:
        # Adding all the values of above temp value to get energy
        # Apply Logarithm for getting energy
        filterbank_energy.append(np.log(reduce(lambda p,q: p+q, temp)))
    final_energy.append(filterbank_energy)

    # Applying Dicrete Cosine Transform to obtained energy in order to get features
    # Defualt length of DCT = 26
    MFCC = scipy.fftpack.dct(final_energy, n=26)
    
    return MFCC
    
def _plot_MFCC_(MFCC):
    """ Plots feature vectors extracted using MFCC """
    # Leave out the first element as it conveys spectral energy (not required) "Co" value
    x = np.array(range(0, len(MFCC[0][0])-1))
    plt.figure(figsize=(16,9))
    for i in range(len(MFCC)):
        plt.plot(x, MFCC[i][0][1:])
    plt.xlabel("Cepstral Coefficients")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.title(r"Mel Frequency Cepstral Coefficient using Kaiser Window ($\beta$=4.667)")
    plt.show()


def _framing_(signal_loc, frame=20, overlap=10):
    """ Creates seperate frames of data from input signal """
    # Overlap and Frame width given in milliseconds
    def _add_zero_padding(signal, frame_length, frame_step):
        """ Pads signal with required amount of zeroes """
        signal=list(signal)
        zeroes=frame_step-((len(signal)-frame_length)%frame_step)
        signal+=[0]*int(zeroes)
        return signal
    
    #Sanity Check 0
    def _check_length(frame):
        """ Return frames of proper length """
        if len(frame)==frame_length: return True
        else: return False
    
    sample_rate,signal=scipy.io.wavfile.read(signal_loc)
    frame_length=sample_rate*(frame*0.001) # Frame is measured in milli-seconds
    frame_step=sample_rate*(overlap*0.001) # Frame is measured in milli-seconds
    signal=_add_zero_padding(signal, frame_length, frame_step)
    
    # Seperate signal data into individual frames using frame length and step
    frames=map(lambda i: signal[i:i+int(frame_length)], range(0, int(len(signal)-frame_length), \
               int(frame_step)))
    # Sanity Check 0
    if not all(map(_check_length, frames)):
        raise ValueError("Inconsistant length of frame(s)")

    return frames, frame_length, frame_step, sample_rate

def MFCC_init(dataset, set_name, save_features=False, plot_sample=False):
    """ Initialise feature extraction """
    labels = []
    flat_MFCC = []
    MFCC_sample = []
    # Choose expected frames to be sligthly larger than usual
    expected_frames = 1750
    zero_pad = 0
    # Get the locations and iterate through the list of locations
    # Iterating through different classes/labels as shown
    for locations, label in dataset:
        # Framing and MFCC
        for i, signal_loc in enumerate(locations):
            labels.append(label)
            frames, frame_length, _, sample_rate = _framing_(signal_loc)
            
            # Sanity Check 0
            if len(frames)!=expected_frames:
                print "Discrepancy encountered in frame length"
                # Frame length assumed to be 320
                zero_pad = (expected_frames-len(frames))*320
                
            MFCC = []
            for frame in frames:
                MFCC.append(MFCC_feature_extract(frame, frame_length, sample_rate))
            if plot_sample:
                # Plotting the Cepstral Coefficients
                MFCC_sample = MFCC
                _plot_MFCC_(MFCC_sample)
                plot_sample = False
                
            # Flatten MFCC in order to pass as input to CNN
            flat_MFCC.append([item for sublist in MFCC[0] for item in sublist])
            flat_MFCC[i].extend([0]*zero_pad)
            zero_pad = 0
        features = zip(flat_MFCC, labels)
    
    with open("../../Feature_extract/feature_{}.pkl".format(set_name), "wb") as f:
        pickle.dump((features), f, -1)
          
    # Saving features as viewable and easily usable files
    if save_features:
        ascii.write(features, "input_features.csv", delimiter=",", overwrite=True)
        ascii.write(features, "input_features.fits", delimiter=",", overwrite=True)
    
    return features