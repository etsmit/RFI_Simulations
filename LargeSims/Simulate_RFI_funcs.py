import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special
from scipy.signal import firwin, freqz, lfilter

import math
import os


#helps calculate upper SK threshold
def upperRoot(x, moment_2, moment_3, p):
    upper = np.abs( (1 - sp.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2)))-p)
    return upper

#helps calculate lower SK threshold
def lowerRoot(x, moment_2, moment_3, p):
    lower = np.abs(sp.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2))-p)
    return lower

#fully calculates upper and lower thresholds
#M = SK_ints
#default p = PFA = 0.0013499 corresponds to 3sigma excision
def SK_thresholds(M, N = 1, d = 1, p = 0.0013499):
    Nd = N * d
    #Statistical moments
    moment_1 = 1
    moment_2 = float(( 2*(M**2) * Nd * (1 + Nd) )) / ( (M - 1) * (6 + 5*M*Nd + (M**2)*(Nd**2)) )
    moment_3 = float(( 8*(M**3)*Nd * (1 + Nd) * (-2 + Nd * (-5 + M * (4+Nd))) )) / ( ((M-1)**2) * (2+M*Nd) *(3+M*Nd)*(4+M*Nd)*(5+M*Nd))
    moment_4 = float(( 12*(M**4)*Nd*(1+Nd)*(24+Nd*(48+84*Nd+M*(-32+Nd*(-245-93*Nd+M*(125+Nd*(68+M+(3+M)*Nd)))))) )) / ( ((M-1)**3)*(2+M*Nd)*(3+M*Nd)*(4+M*Nd)*(5+M*Nd)*(6+M*Nd)*(7+M*Nd) )
    #Pearson Type III Parameters
    delta = moment_1 - ( (2*(moment_2**2))/moment_3 )
    beta = 4 * ( (moment_2**3)/(moment_3**2) )
    alpha = moment_3 / (2 * moment_2)
    beta_one = (moment_3**2)/(moment_2**3)
    beta_two = (moment_4)/(moment_2**2)
    error_4 = np.abs( (100 * 3 * beta * (2+beta) * (alpha**4)) / (moment_4 - 1) )
    kappa = float( beta_one*(beta_two+3)**2 ) / ( 4*(4*beta_two-3*beta_one)*(2*beta_two-3*beta_one-6) )
    print('kappa: {}'.format(kappa))
    x = [1]
    upperThreshold = sp.optimize.newton(upperRoot, x[0], args = (moment_2, moment_3, p))
    lowerThreshold = sp.optimize.newton(lowerRoot, x[0], args = (moment_2, moment_3, p))
    return lowerThreshold, upperThreshold


# In[3]:


def vco_complex(v_in,f0,K0,ts,tbit,nbits):
    """
    Takes an input voltage and returns a variable frequency waveform
    Inputs:
        v_in : input time-indexed voltages (1D)
        f0   : quiescent frequency of oscillator
        K0   : oscillator sensitivity (Hz / V)
        ts   : sampling time (reciprocal of sample rate)
    Output:
        v_out: output time-indexed voltages
    """
    #define stuff
    x = np.tile(np.arange(tbit),nbits)
    phase = np.empty(len(v_in))
    
    #define instantaneous frequencies and phases
    freq = f0 + K0*v_in
    phase = sp.integrate.cumtrapz(freq,dx=ts,initial=0)
    for i in range(nbits):
        phase[i*tbit:(i+1)*tbit] = phase[i*tbit]
        
    #create waveform
    arg = (2.j*np.pi*x*freq*ts) + 1.j*phase
    v_out = np.exp(arg)
    
    return v_out,freq,phase


#binary freq-shift keying - switch between 2 freqs
def bfsk(nbits,symbol_rate,f0,f1,Ebit=0.0,N0=None,fs=800e6):

    tbit = int( fs / (symbol_rate*1e3) )
    
    #make bit sequence (and turn into seq of +/- 0.5's for VCO)
    bit_seq = np.random.randint(0,high=2,size=nbits)
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse) - 0.5
    
    #lo-pass filter
    hann = np.hanning(int(tbit*0.2))
    #sym_seq = np.convolve(sym_seq,hann,mode='same')
    #sym_seq = sym_seq/(2*np.max(sym_seq))
    
    ts=1/fs
    Ebit_linear = 10**(Ebit/10.0)

    #define VCO f0,K0 based on inputs
    vco_center = (f0+f1)/2
    #assuming sym_seq = 1 corresponds to voltage = 1V
    vco_sens = (f1-f0)
    
    sig,f,p = vco_complex(sym_seq, vco_center, vco_sens, ts, tbit, nbits)

    sig *= np.sqrt(Ebit_linear)

    return sig,sym_seq,f,p



#binary freq-shift keying - switch between 2 freqs
def bfsk_fast(nbits,symbol_rate,f0,f1,wincut,Ebit=0.0,N0=None,fs=800e6):

    tbit = int( fs / (symbol_rate*1e3) )
    
    #make bit sequence (and turn into seq of +/- 0.5's for VCO)
    bit_seq = np.random.randint(0,high=2,size=nbits)
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse) - 0.5
    

    fir_sz = int(0.2*tbit)
    sinc = scipy.signal.firwin(fir_sz, cutoff=wincut/fir_sz, window="rectangular")
    sym_seq = scipy.signal.convolve(sym_seq,sinc,mode='same',method='fft')

    
    ts=1/fs
    Ebit_linear = 10**(Ebit/10.0)

    #define VCO f0,K0 based on inputs
    vco_center = (f0+f1)/2
    #assuming sym_seq = 1 corresponds to voltage = 1V
    vco_sens = (f1-f0)
    
    sig,f,p = vco_complex(sym_seq, vco_center, vco_sens, ts, tbit, nbits)

    sig *= np.sqrt(Ebit_linear)

    return sig,sym_seq,f,p



def ask_mod(x,e_vec,nsym,tbit,wincut,fc,bias,Ebit=0.0,N0=None,fs=800e6):

    #ASK but with power levels between 0 and 1 to test dc stuff
    #tbit = int( fs / (symbol_rate*1e3) )
    symsz = int((len(x)/tbit))+1
    bit_seq = np.random.randint(0,2,size=symsz).astype(np.float16)
    bit_seq[bit_seq==0] = bias
    print(tbit)
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]


    fir_sz = int(0.2*tbit)
    sinc = scipy.signal.firwin(fir_sz, cutoff=wincut/fir_sz, window="rectangular")
    sym_seq = scipy.signal.convolve(sym_seq,sinc,mode='same',method='fft')


    #apply carrier signal
    ts = 1/fs
    #e_vec = np.exp(2.j*np.pi*fc*np.arange(len(sym_seq))*ts)
    #e_vec = np.exp(2.j*np.pi * fs/f_sim * x)
    print(len(sym_seq))
    print(len(e_vec))
    
    sig = sym_seq * e_vec

    return sig,sym_seq,bit_seq



def ask_fast(x,e_vec,nsym,tbit,nbit,wincut,fc,Ebit=0.0,N0=None,fs=800e6):

    #tbit = int( fs / (symbol_rate*1e3) )
    bit_seq = np.random.RandomState().randint(0,2**nbit,size=nsym)
    bit_seq = 2*bit_seq/(2**nbit - 1) - 1
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]


    fir_sz = int(0.2*tbit)
    sinc = scipy.signal.firwin(fir_sz, cutoff=wincut/fir_sz, window="rectangular")
    sym_seq = scipy.signal.convolve(sym_seq,sinc,mode='same',method='fft')



    #apply carrier signal
    ts = 1/fs
    #e_vec = np.exp(2.j*np.pi*fc*x*ts)
    #e_vec = np.exp(2.j*np.pi * fs/f_sim * x)
    
    sig = sym_seq * e_vec

    return sig,sym_seq,bit_seq





    
def SK_EST(a,m,n=1,d=1):
        #make s1 and s2 as defined by whiteboard (by 2010b Nita paper)
        a = a[:,:m]*n
        sum1=np.sum(a,axis=1)
        sum2=np.sum(a**2,axis=1)
        sk_est = ((m*n*d+1)/(m-1))*(((m*sum2)/(sum1**2))-1)                     
        return sk_est
    
def msk(nbits,symbol_rate,fc,Ebit=0.0,N0=None,fs=800e6):
    #minimum-shift keying - 4 levels
    tbit = int( fs / (symbol_rate*1e3) )
    
    bit_seq = np.random.randint(0,high=4,size=nbits)
    pulse = np.ones(2*tbit)
    
    #set a_I, a_Q and pulse to make symbol seq
    a_I = -1*np.ones(nbits)
    a_I[bit_seq>=2]=1
    a_Q = -1*np.ones(nbits)
    a_Q[bit_seq%2==1]=1
    
    sym_I = np.kron(a_I,pulse)
    sym_I = np.roll(sym_I, -1*tbit)
    sym_Q = np.kron(a_Q,pulse)
    
    #time index and mod arg
    x = np.arange(nbits*tbit)
    mod_arg = (np.pi*x)/(2*tbit)
    sig_arg = 2*np.pi*fc*x
    
    sig = sym_I*np.cos(mod_arg)*np.cos(sig_arg) - sym_Q*np.sin(mod_arg)*np.sin(sig_arg)
    return sig





def bpsk_alt(x,e_vec,nbits,symbol_rate,wincut,fc,Ebit,fs=800e6):
    #binary phase shift keyed
    tbit = int( fs / (symbol_rate*1e3) )
    #print('making index range...')
    #x = np.arange(nbits*tbit)
    #print('making bit seq...')
    bit_seq = np.random.randint(0,2,size=(int(len(x)/tbit)+1,))
    bit_seq = (2*bit_seq)-1
    pulse = np.ones(tbit)
    #print('making symbol seq...')
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]
    
    fir_sz = int(0.2*tbit)
    sinc = scipy.signal.firwin(fir_sz, cutoff=wincut/fir_sz, window="rectangular")
    sym_seq = scipy.signal.convolve(sym_seq,sinc,mode='same',method='fft')
    
    #apply carrier signal
    ts = 1/fs
    #print('making carrier signal...')
    #e_vec = np.exp(2.j*np.pi*fc*x*ts)
    #e_vec = np.exp(2.j*np.pi * fs/f_sim * x)
    #print('modulating...')
    
    sig = sym_seq * e_vec
    return sig,sym_seq,bit_seq




def bpsk_fast(x,e_vec,nbits,symbol_rate,fc,Ebit,fs=800e6):
    #binary phase shift keyed
    #x, e_vec are index and carrier wave arrays - reused for expediency

    tbit = int( fs / (symbol_rate*1e3) )
    #print('making index range...')
    #x = np.arange(nbits*tbit)
    bit_seq = np.random.RandomState().randint(0,2,size=(int(len(x)/tbit)+1,))
    #bit_seq = np.random.randint(0,2,size=(int(len(x)/tbit)+1,))
    bit_seq = (2*bit_seq)-1
    #print('num_bits: {}'.format(len(bit_seq)))
    pulse = np.ones(tbit)
    #print('making symbol seq...')
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]

    fir_sz = int(0.2*tbit)
    sinc = scipy.signal.firwin(fir_sz, cutoff=1.0/fir_sz, window="rectangular")
    sym_seq = scipy.signal.convolve(sym_seq,sinc,mode='same',method='auto')
    #sym_seq = np.convolve(sym_seq,sinc,mode='same')
    #apply carrier signal
    ts = 1/fs
    #print('making carrier signal...')
    #e_vec = np.exp(2.j*np.pi*fc*x*ts)
    #e_vec = np.exp(2.j*np.pi * fs/f_sim * x)
    #print('modulating...')
    sig = sym_seq * e_vec

    return sig,sym_seq,bit_seq






def qpsk(x,nbits,symbol_rate,wincut,fc,Ebit,fs=800e6):
    #quad phase shift keyed
    #derived number of samples per symbol (this should go inside each rfi generator)
    tbit = int( fs / (symbol_rate*1e3) )
    
    #x = np.arange(nbits*tbit)
    bit_seq = np.random.randint(1,5,size=(int(len(x)/tbit)+1,))
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]


    fir_sz = int(0.2*tbit)
    sinc = scipy.signal.firwin(fir_sz, cutoff=1.0/fir_sz, window="rectangular")
    sym_seq = scipy.signal.convolve(sym_seq,sinc,mode='same',method='auto')


    #apply carrier signal
    ts = 1/fs
    
    #theres a faster way to optimize qpsk i think but this is fine for overnight
    arg = (2.j*np.pi*fc*x*ts) + (1.j*(np.pi/4)*(2*sym_seq-1))
    sig = np.exp(arg)
    #e_vec = np.exp(2.j*np.pi * fs/f_sim * x)
    
    #sig = e_vec

    return sig,sym_seq,bit_seq


def duty_cycle(y,percent,period,fs=800e6):
    #y: 1D input signal
    #period in ms, given sampling rate fs (in Hz)
    #percent as a fraction in range 0-1
    period_nsamp = period * 1e-3 * fs
    print('number of samples per duty cycle: {}'.format(period_nsamp))
    one_dc = np.ones(int(period_nsamp))
    one_dc[int(percent*period_nsamp):] = 0
    n_dc = (len(y) // period_nsamp) + 1
    print('number of duty cycles: {}'.format(n_dc))
    dc = np.tile(one_dc,int(n_dc))[:len(y)]
    
    return dc*y
    





def SK_master(s,m):
    #do single-scale SK
    #s: 2D array of input data (chan,spectra)
    #m: SK M value
    numSKspectra = s.shape[1]//m

    for i in range(numSKspectra):
        this_s = s[:,i*m:(i+1)*m]
        if (i==0):
            out_sk = SK_EST(this_s,m)
            out_sk = np.expand_dims(out_sk,axis=1)
            out_s = np.expand_dims(np.mean(this_s,axis=1),axis=1)
        else:
            out_sk = np.c_[out_sk,SK_EST(this_s,m)]
            out_s = np.c_[out_s,np.mean(this_s,axis=1)]
    return out_sk,out_s

def ms_SK_EST(s1,s2,m,n=1,d=1):
    #do multi-scale SK using ms-s1,ms-s2
    sk_est = ((m*n*d+1)/(m-1))*(((m*s2)/(s1**2))-1)

    return sk_est

def ms_SK_master(s,m,ms0,ms1,lt,ut):
    #bin up s for multi-scale SK
    #and call ms_SK_EST
    numSKspectra = s.shape[1]//m

    Nchan= s.shape[0]
    n=1
    d=1
    
    ms_binsize = ms0*ms1
    ms_s1 = np.zeros((s.shape[0]-(ms0-1),numSKspectra-(ms1-1)))
    ms_s2 = np.zeros((s.shape[0]-(ms0-1),numSKspectra-(ms1-1)))
    
    #fill single scale s1,s2
    for i in range(numSKspectra):
        this_s = s[:,i*m:(i+1)*m]
        if i==0:
            s1 = np.expand_dims(np.sum(this_s,axis=1),axis=1)
            s2 = np.expand_dims(np.sum(this_s**2,axis=1),axis=1)
            
        else:
            s1 = np.c_[s1,np.sum(this_s,axis=1)]
            s2 = np.c_[s2,np.sum(this_s**2,axis=1)]
                  
    #fill multiscale s1, s2
    for ichan in range(ms0):
        for itime in range(ms1):
            
            ms_s1 += (1./ms_binsize) * (s1[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(numSKspectra-(ms1-1))])
            ms_s2 += (1./ms_binsize) * (s2[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(numSKspectra-(ms1-1))])

    #Multiscale SK
    for k in range(numSKspectra-(ms1-1)):

        sk_spect = ms_SK_EST(ms_s1[:,k],ms_s2[:,k],m,n,d)

        ms_flag_spect = np.zeros((Nchan-(ms0-1)),dtype=np.int8)
        
        ms_flag_spect[sk_spect>ut] = 1
        ms_flag_spect[sk_spect<lt] = 1


        #append to results
        if (k==0):
            ms_sk_block=np.expand_dims(sk_spect,axis=1)
            ms_flags_block = np.expand_dims(ms_flag_spect,axis=1)

        else:
            ms_sk_block=np.c_[ms_sk_block,np.expand_dims(sk_spect,axis=1)]
            ms_flags_block = np.c_[ms_flags_block,np.expand_dims(ms_flag_spect,axis=1)]
    print('----')
    
              
    return ms_sk_block,ms_flags_block,ms_s1


#define where rfi is based on -10dB below noise

#define where rfi is based on -10dB below noise
#x     : input 1D data of signal only
#n     : input 1D noise only
#Nchan : Number of channels for pfb
#SKM   : SK M, size of each averaging bin (to match flag array size to SK mask)
#Nsk   : Number of SK bins (to match flag array size to SK mask)
#M     : number of taps in PFB (M=24 for most VPM modes)
#P     : same as Nchan - size of channelizer output spectra
def power_mask_pfb(x,n,Nchan,SKM,Nsk,M,P):
	#print('making power mask')
	out_f = np.zeros((Nsk,Nchan),dtype=np.int8)

	fb_shape = (Nsk*SKM,Nchan)
	s_shape = (Nsk,Nchan)
    
	xfb = pfb_filterbank(x, M, P)
	nfb = pfb_filterbank(n, M, P)

	xs = np.zeros(fb_shape,dtype=np.complex64)
	xs[:-M,:] = xfb
	xs[-M:,:] = xfb[-1,:]
	xss = np.abs(xs)**2
	xfb = None
	xs = None
    
	ns = np.zeros(fb_shape,dtype=np.complex64)
	ns[:-M,:] = nfb
	ns[-M:,:] = nfb[-1,:]
	nss = np.abs(ns)**2
	nfb = None
	ns = None

	x_ave = np.zeros(s_shape,dtype=np.float64)
	n_ave = np.zeros(s_shape,dtype=np.float64)
	s_db = np.zeros(s_shape,dtype=np.float64)
    
	for i in range(Nsk):
		this_x = xss[SKM*i:SKM*(i+1),:]
		this_n = nss[SKM*i:SKM*(i+1),:]
		x_ave[i,:] = np.mean(this_x,axis=0)
		n_ave[i,:] = np.mean(this_n,axis=0)
		s_db[i,:] = 10*np.log10((x_ave[i,:])/n_ave[i,:])


	out_f[s_db > -10] = 1
	return out_f.T,s_db



#x: input 1d data stream
#win_coeffs: window coefficients (from??)
#M: # of taps
#P: # of branches/points
def pfb_fir_frontend(x, win_coeffs, M, P):
    #print('pfb')
    #print(x.shape[0])
    W = int(x.shape[0] / M / P)
    x_p = x.reshape((W*M, P)).T
    #print(x_p.shape)
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, M * W - M),dtype=np.complex64)
    for t in range(0, M*W-M):
        x_weighted = x_p[:, t:t+M] * h_p
        x_summed[:, t] = x_weighted.sum(axis=1)
    return x_summed.T

def generate_win_coeffs(M, P, window_fn="hann"):
    win_coeffs = scipy.signal.get_window(window_fn, M*P)
    sinc       = scipy.signal.firwin(M * P, cutoff=1.0/P, window="rectangular")
    win_coeffs *= sinc
    return win_coeffs

#oly for real data
#def fft(x_p, P, axis=1):
#    return np.fft.rfft(x_p, P, axis=axis)

def pfb_filterbank(x, M, P, window_fn="hann"):
    win_coeffs = generate_win_coeffs(M, P, window_fn="hann")
    x_fir = pfb_fir_frontend(x, win_coeffs, M, P)
    x_pfb = np.fft.fft(x_fir, axis=1)
    return x_pfb 









