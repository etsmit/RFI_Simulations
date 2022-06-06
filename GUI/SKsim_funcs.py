

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
def bfsk(nbits,symbol_rate,f0,f1,wincut,Ebit=0.0,N0=None,fs=800e6):

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



def ask(nsym,tbit,nbit,wincut,fc,Ebit=0.0,N0=None,fs=800e6):
    #nsym : number of symbols
    #tbit : number of time samples per bit
    #nbit : number of bits per symbol (nbit=2 has 2**nbit=4 levels)

    #tbit = int( fs / (symbol_rate*1e3) )
    x = np.arange(nsym*tbit)
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
    e_vec = np.exp(2.j*np.pi*fc*x*ts)
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


def bpsk(nbits,symbol_rate,wincut,fc,Ebit,fs=800e6):
    #binary phase shift keyed
    #nbits :       number of bits to generate
    #symbol rate : how many kilobits per second (binary=1 bit per symbol)
    #wincut :      cutoff frequency for scipy.signal.firwin
    #fc :          baseband carrier frequency (Hz)
    #Ebit :        Energy per bit (not used, need to implement signal strength in a way that makes sense)
    #fs :          sampling rate (samples per second)
    #================

    #number of samples per bit
    tbit = int( fs / (symbol_rate*1e3) )

    #index array
    x = np.arange(nbits*tbit)

    #generate random bits and extend into symbols
    bit_seq = np.random.randint(0,2,size=(int(len(x)/tbit)+1,))
    bit_seq = (2*bit_seq)-1
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]

    #apply sinc FIR with a width of 20% bit length
    fir_sz = int(0.2*tbit)
    sinc = scipy.signal.firwin(fir_sz, cutoff=wincut/fir_sz, window="rectangular")
    sym_seq = scipy.signal.convolve(sym_seq,sinc,mode='same',method='fft')

    #apply carrier signal
    ts = 1/fs
    e_vec = np.exp(2.j*np.pi*fc*x*ts)

    #binary phase shift means 180deg flip - a simple +/-1 multiply
    sig = sym_seq * e_vec

    #return time samples as well as bit data stream (for verification/sanity)
    return sig,sym_seq,bit_seq


def qpsk(nbits,symbol_rate,fc,Ebit,fs=800e6):
    #quad phase shift keyed
    #derived number of samples per symbol (this should go inside each rfi generator)
    tbit = int( fs / (symbol_rate*1e3) )

    x = np.arange(nbits*tbit)
    bit_seq = np.random.randint(1,5,size=(int(len(x)/tbit)+1,))
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]


    #apply carrier signal
    ts = 1/fs

    arg = (2.j*np.pi*fc*x*ts) + (1.j*(np.pi/4)*(2*sym_seq-1))
    e_vec = np.exp(arg)
    #e_vec = np.exp(2.j*np.pi * fs/f_sim * x)

    sig = sym_seq * e_vec

    return sig,sym_seq,bit_seq


def duty_cycle(y,percent,period,fs=800e6):
    #period in ms, given sampling rate fs (in Hz)
    #percent in fraction 0-1
    period_nsamp = period * 1e-3 * fs
    print('number of samples per duty cycle: {}'.format(period_nsamp))
    one_dc = np.ones(int(period_nsamp))
    one_dc[int(percent*period_nsamp):] = 0
    n_dc = (len(y) // period_nsamp) + 1
    print('number of duty cycles: {}'.format(n_dc))
    dc = np.tile(one_dc,int(n_dc))[:len(y)]

    return dc*y






def SK_master(s,m):
    numSKspectra = s.shape[1]//m
    print(numSKspectra)
    for i in range(numSKspectra):
        this_s = s[:,i*m:(i+1)*m]
        if i==0:
            out_sk = SK_EST(this_s,m)
            out_sk = np.expand_dims(out_sk,axis=1)
            out_s = np.expand_dims(np.mean(this_s,axis=1),axis=1)
        else:
            out_sk = np.c_[out_sk,SK_EST(this_s,m)]
            out_s = np.c_[out_s,np.mean(this_s,axis=1)]
    return out_sk,out_s

def ms_SK_EST(s1,s2,m,n=1,d=1):
    sk_est = ((m*n*d+1)/(m-1))*(((m*s2)/(s1**2))-1)

    return sk_est

def ms_SK_master(s,m,ms0,ms1,lt,ut):
    print('---- MS SK ----')
    print(s.shape)
    numSKspectra = s.shape[1]//m
    print(numSKspectra)
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

    print(s1.shape)
    #fill multiscale s1, s2
    for ichan in range(ms0):
        for itime in range(ms1):

            ms_s1 += (1./ms_binsize) * (s1[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(numSKspectra-(ms1-1))])
            ms_s2 += (1./ms_binsize) * (s2[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(numSKspectra-(ms1-1))])


            #ms_s1 += (s1[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(numSKspectra-(ms1-1))])
            #ms_s2 += (s2[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(numSKspectra-(ms1-1))])
    print(ms_s1.shape)

    #plt.imshow(np.log10(ms_s1),interpolation='nearest',aspect='auto',cmap='hot',vmin=2.5,vmax=3)
    #plt.colorbar()
    #plt.show()


    #Multiscale SK
    for k in range(numSKspectra-(ms1-1)):


        #sk_spect = ms_SK_EST(ms_s1[:,k],ms_s2[:,k],numSKspectra-(ms1-1),n,d)
        sk_spect = ms_SK_EST(ms_s1[:,k],ms_s2[:,k],m,n,d)
        #sk_spect[:,1] = ms_SK_EST(ms_s1[:,k],ms_s2[:,k],numSKspectra-(ms1-1),n,d)


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
def power_mask(x,n,Nchan,SKM,Nsk,M,P):
    print('making power mask')
    out_f = np.zeros((Nsk,Nchan),dtype=np.int8)
    out_bf = np.zeros((Nsk*SKM,Nchan),dtype=np.int8)

    fb_shape = (Nsk*SKM,Nchan)
    s_shape = (Nsk,Nchan)


    xfb = pfb_filterbank(x, M, P)
    nfb = pfb_filterbank(n, M, P)

    xs = np.zeros(fb_shape,dtype=np.complex64)
    xs[:-M,:] = xfb
    xs[-M:,:] = xfb[-1,:]
    xss = np.abs(xs)**2

    ns = np.zeros(fb_shape,dtype=np.complex64)
    ns[:-M,:] = nfb
    ns[-M:,:] = nfb[-1,:]
    nss = np.abs(ns)**2

    x_ave = np.zeros(s_shape,dtype=np.float64)
    n_ave = np.zeros(s_shape,dtype=np.float64)
    s_db = np.zeros(s_shape,dtype=np.float64)

    sbig_db = np.zeros(fb_shape,dtype=np.float64)

    this_nbig_ave = np.mean(nss)
    for i in range(Nsk*SKM):
        sbig_db[i,:] = 10*np.log10((xss[i,:])/this_nbig_ave)

    out_bf[sbig_db > -10] = 1

    for i in range(Nsk):
        this_x = xss[SKM*i:SKM*(i+1),:]
        this_n = nss[SKM*i:SKM*(i+1),:]
        x_ave[i,:] = np.mean(this_x,axis=0)
        n_ave[i,:] = np.mean(this_n,axis=0)
        #print(10*np.log10((x_ave[i,:])/n_ave[i,:]))
        s_db[i,:] = 10*np.log10((x_ave[i,:])/n_ave[i,:])


    out_f[s_db > -10] = 1
    return out_f.T,s_db,sbig_db,out_bf


#x: input 1d data stream
#win_coeffs: window coefficients (from??)
#M: # of taps
#P: # of branches/points
def pfb_fir_frontend(x, win_coeffs, M, P):
    W = int(x.shape[0] / M / P)
    x_p = x.reshape((W*M, P)).T
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, M * W - M),dtype=np.complex64)
    for t in range(0, M*W-M):
        x_weighted = x_p[:, t:t+M] * h_p
        x_summed[:, t] = x_weighted.sum(axis=1)
    return x_summed.T


def generate_win_coeffs(M, P, window_fn="hamming"):
    win_coeffs = scipy.signal.get_window(window_fn, M*P)
    sinc       = scipy.signal.firwin(M * P, cutoff=1.0/P, window="rectangular")
    win_coeffs *= sinc
    return win_coeffs


def pfb_filterbank(x, M, P, window_fn="hamming"):
    win_coeffs = generate_win_coeffs(M, P, window_fn="hamming")
    x_fir = pfb_fir_frontend(x, win_coeffs, M, P)
    x_pfb = np.fft.fft(x_fir, axis=1)
    return x_pfb


def passband_fir(y,Nchan,fc_sig):
    #y input data
    sinc = scipy.signal.firwin(Nchan, cutoff=1.0/Nchan, window="rectangular")

    s_shift = np.exp(2.j*np.pi*(fc_sig/2)*np.arange(Nchan)*ts)
    fir = sinc * s_shift
    res = np.convolve(y,fir,mode='same')
    return res




def simulate(encode,Nchan,SK_m,fs,cc_sig,cc_tone,cc_fsk_space,sym_rate,bits_per_sym,num_SK,SKsigma,ms0,ms1,ask_bias,dc,dcper,wincut,newdata,newdc,Iplot,spect_lbl,sdb_lbl,ssplt_lbl,msplt_lbl,num):
    #do the thing
    print('hello?')

    #INPUTS:
    #Nchan=256
    #SK_m=512

    global y
    #global num
    #PFB taps
    M=24
    P=Nchan

    #save_dir = '/Users/evansmith/WVU/RFI_MIT/Simulations/pictures/spring2021/'
    #save_dir = '/Users/evansmith/WVU/RFI_MIT/Simulations/pics_for_pub/'
    save_dir = "C:/Users/etsmi/WVU/RFI_MIT/Simulations/pictures/gui/"

    #fs=50e6
    print('freq res: {} kHz'.format(1e-3*(fs/Nchan)))

    #carrier channels of signals
    #cc_sig = 120
    #cc_tone = 200

    #TODO: These should all be under a single try-except lmao what was I thinking
    if cc_sig == int(cc_sig):
        cc_sig = int(cc_sig)
    #cc_fsk_space = 130
    if cc_fsk_space == int(cc_fsk_space):
        cc_fsk_space = int(cc_fsk_space)
    #symbol rate in ksps
    #sym_rate = 100
    if sym_rate == int(sym_rate):
        sym_rate = int(sym_rate)

    #number of SK blocks
    #num_SK = 300

    #multiscale (ms0 = channels, ms1 = time)
    #ms0=4
    #ms1=2

    #ask_bias = 0.9

    #duty cycle
    #dc = 0.9#ratio in (0,1]
    if dc == int(dc):
        dc = int(dc)
    #dcper = 1#ms
    if dcper == int(dcper):
        dcper = int(dcper)

    #DERIVED VALUES:

    #carrier frequencies of signals
    fc_sig = cc_sig*(fs/Nchan)
    fc_tone = cc_tone*(fs/Nchan)

    fc_fsk_space = cc_fsk_space*(fs/Nchan)

    print('tone: {} MHz || signal: {} MHz'.format(fc_tone/1e6,fc_sig/1e6))
    print('FSK space frequency: {} MHz'.format(fc_fsk_space/1e6))

    ts=1/fs

    print('symbol rate {} kbps'.format(sym_rate))

    #derived number of samples per symbol (this should go inside each rfi generator)
    ns_per_bit = int( fs / (sym_rate*1e3) )

    print('number of time samples per symbol: {}'.format(ns_per_bit))

    #power levels
    noise_db = 0.0
    sig_db = 0.0
    noise_linear = 10**(noise_db/10)
    sig_linear = 10**(sig_db)

    #number of spectra
    num_spectra = num_SK * SK_m
    print(num_spectra)

    #number of symbols
    num_bits = math.ceil((num_spectra*Nchan)/ns_per_bit)

    new_data=1

#========================================

    SK_p = (1 - scipy.special.erf(SKsigma / math.sqrt(2))) / 2
    print('Probability of false alarm: {}'.format(SK_p))

    lt,ut=SK_thresholds(SK_m,p = SK_p)
    print('Thresholds: {} || {}'.format(lt,ut))



    if newdata:

        print('Making new data')
        if encode=='BPSK':
            print('Making BPSK signal...')
            y,sym,bits = bpsk(num_bits,sym_rate,fc_sig,Ebit=0.0,fs=fs)
        elif encode=='ASK':
            print('Making ASK signal...')
            y,sym_seq,bit_seq = ask(num_bits,ns_per_bit,bits_per_sym,wincut,fc_sig,Ebit=0.0,N0=None,fs=fs)
        elif encode=='BFSK':
            print('Making BFSK signal...')
            y,sym,f,p = bfsk(num_bits,sym_rate,fc_sig,fc_fsk_space,Ebit=0.0,N0=None,fs=fs)
        elif encode=='QPSK':
            print('Making QPSK signal...')
            y,sym,bits = qpsk(num_bits,sym_rate,fc_sig,Ebit=0.0,fs=fs)

    if newdc:
        #duty cycle
        ydc = duty_cycle(y,dc,dcper,fs=fs)
    else:
        ydc = y
#with fs=100e6,
#tbit=4096*10 is 2.4 kbps
#tbit = 5000 is 20 kbps
#tbit=512 is 192kbps
        #print(len(y))
    if newdc or newdata:
        ydc = ydc[:(Nchan*num_spectra)]
        #print(len(y))
        print('------------')

#generate noise
        n = np.random.normal(0,1,size=len(ydc)) + 1.j*np.random.normal(0,1,size=len(ydc))
        #n_var = np.var(n)
        #n_pow_factor = np.sqrt(noise_linear / n_var)
        #n *= n_pow_factor

        ramp = np.arange(len(ydc))/(len(ydc))
        #ramp = (0.4*ramp) + 0.6
        #ramp = 1
        ydc *= ramp

        #test = np.exp(2.j*np.pi*fc_tone*np.arange(len(y))*ts)
        fb_shape = (num_spectra,Nchan)



        fb = pfb_filterbank(n+ydc, M, P)
    #fb = pfb_filterbank(n, M, P)
    #fb_fir = pfb_filterbank(passband_fir(n+y,Nchan,fc_sig), M, P)

        s = np.zeros(fb_shape,dtype=np.complex64)
        s[:-M,:] = fb
        s[-M:,:] = fb[-1,:]

        s = np.abs(s.T)**2

        #f_db,s_db = power_mask(y,n,Nchan,SK_m,num_SK)
        f_db,s_db,sbig_db,fb_db = power_mask(ydc,n,Nchan,SK_m,num_SK,M,P)

        #clean up mem usage
        test = None


        print(fb.shape)
#hann=np.hanning(4096)
#hann=np.expand_dims(hann,axis=1)
#hann = np.repeat(hann,200,axis=1)

#hann_s = np.fft.fft(fb*hann,axis=1)
        #s = np.fft.fft(fb,axis=1)

        fb = None

#hann_s = np.abs(hann_s)**2
        #s = np.abs(s.T)**2

        np.save('prev_s.npy',s)
        np.save('prev_fdb.npy',f_db)

        #====================================#

    else:
        s = np.load('prev_s.npy')
        f_db = np.load('prev_fdb.npy')



    sk,s_ave = SK_master(s,SK_m)
    ms_sk,ms_f,ms_s1 = ms_SK_master(s,SK_m,ms0,ms1,lt,ut)

    pulse = np.ones((1,SK_m))

    #s = None

    #lt = 0.7650106929801714
    #ut = 1.3641737928198505

    f = np.zeros(sk.shape)
    f[sk>ut]=1
    f[sk<lt]=1

    #f_sing is SS flags
    f_sing = np.array(f)

    #apply OR mask to f
    #f is MS flags
    for ichan in range(ms0):
        for itime in range(ms1):
            f[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(num_SK-(ms1-1))][ms_f==1] = 1

    ms_f = None


#f = np.kron(f,pulse)

# s_flag_both = np.array(s_ave)
# s_nan_both = np.array(s_ave)
# s_flag_both[f==1]=1e-6
# s_nan_both[f==1]=np.nan
#
# s_flag_sing = np.array(s_ave)
# s_nan_sing = np.array(s_ave)
# s_flag_sing[f_sing==1]=1e-6
# s_nan_sing[f_sing==1]=np.nan
#
# s_flag_comp = np.array(s_ave)
# s_nan_comp = np.array(s_ave)
# #print(s_flag_comp.shape)
# #print(f_comp.shape)
# s_flag_comp[f_db==1]=1e-6
# s_nan_comp[f_db==1]=np.nan

    s_flag_both = np.array(s_ave)
    s_flag_both[f==1]=1e-3

    s_flag_sing = np.array(s_ave)
    s_flag_sing[f_sing==1]=1e-3



    s_flag_db = np.array(s_ave)
    print(s_flag_db.shape)
    print(f_db.shape)
    s_flag_db[f_db==1]=1e-3

    #f = None

    m = np.arange(Nchan)*800/Nchan
    ext = [0,100,50,0]

    #determine filename base
    if encode=='BPSK':
        fname_base = 'bpsk_{}ksps_chan{}_m{}_dc{}_{}ms'.format(sym_rate,cc_sig,SK_m,dc,dcper)
    elif encode=='ASK':
        fname_base = 'ask_{}ksps_chan{}_m{}_dc{}_{}ms_bias{}'.format(sym_rate,cc_sig,SK_m,dc,dcper,ask_bias)
    elif encode=='BFSK':
        fname_base = 'bfsk_{}ksps_chan{}_m{}_dc{}_{}ms_space{}'.format(sym_rate,cc_sig,SK_m,dc,dcper,cc_fsk_space)
    elif encode=='QPSK':
        fname_base = 'qpsk_{}ksps_chan{}_m{}_dc{}_{}ms'.format(sym_rate,cc_sig,SK_m,dc,dcper)

    print('Filename base: '+fname_base)

    fname = fname_base+'_spect.png'

    if newdata:

        fig = plt.figure()

        ax_lw=3
        ax = plt.axes([0.1, 0.1, 0.898, 0.898])

        ax.imshow(np.log10(s_ave),interpolation='nearest',aspect='auto',cmap='hot',vmin=-2.35,vmax=-1.35)
        ax.tick_params(axis='both',direction='in',width=2,length=8,top=True,right=True,pad=2)
        ax.spines['bottom'].set_linewidth(ax_lw)
        ax.spines['top'].set_linewidth(ax_lw)
        ax.spines['right'].set_linewidth(ax_lw)
        ax.spines['left'].set_linewidth(ax_lw)
        plt.ylim((140,100))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        ax.set_yticklabels(['',105,110,115,120,125,130,135,140])
        #ax.set_yticks(['',105,110,115,120,125,130,135,140])
        plt.ylabel('Channel',fontsize=13)
        plt.xlabel('Time',fontsize=13)
        plt.text(50,105,spect_lbl,fontsize=22,bbox=dict(facecolor='white', alpha=1))
        #plt.savefig(save_dir+fname,format='png')
        if Iplot:
            plt.show()
        plt.close()


    fname = fname_base+'_dB.png'
    if newdata:

        fig = plt.figure()

        ax_lw=3
        ax = plt.axes([0.1, 0.1, 0.898, 0.898])

        ax.imshow(np.log10(s_flag_db),interpolation='nearest',aspect='auto',cmap='hot',vmin=-2.35,vmax=-1.35)
        ax.tick_params(axis='both',direction='in',width=2,length=8,top=True,right=True,pad=2)
        ax.spines['bottom'].set_linewidth(ax_lw)
        ax.spines['top'].set_linewidth(ax_lw)
        ax.spines['right'].set_linewidth(ax_lw)
        ax.spines['left'].set_linewidth(ax_lw)
        plt.ylim((140,100))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        ax.set_yticklabels(['',105,110,115,120,125,130,135,140])
        #ax.set_yticks(['',105,110,115,120,125,130,135,140])
        plt.ylabel('Channel',fontsize=13)
        plt.xlabel('Time',fontsize=13)
        plt.text(50,105,sdb_lbl,fontsize=22,bbox=dict(facecolor='white', alpha=1))
        #plt.savefig(save_dir+fname,format='png')
        if Iplot:
            plt.show()
        plt.close()


    fname = str(num).zfill(3)+fname_base+'_SK_image.png'


    #noise = s_ave[20:100,:]
    #mean = np.mean(noise)
    #std = np.std(noise)

    noise = None

    #true_flags = np.zeros(s_ave.shape)
    #true_flags[s_ave > (mean + 4*std)] = 1

    rfi_flagged = np.copy(f_db[100:140,:])
    rfi_flagged[f_sing[100:140,:]==0] = 0

    flg_pct = (100.*np.count_nonzero(rfi_flagged))/np.count_nonzero(f_db)
    flg_pct = np.around(flg_pct,2)
    ss_flg = flg_pct

    rfi_overflagged = np.copy(-f_db[100:140,:]+1)
    rfi_overflagged[f_sing[100:140,:]==0]=0

    try:
        overflg_pct = (100.*np.count_nonzero(rfi_overflagged))/np.count_nonzero(f_db[100:140,:])
        overflg_pct = np.around(overflg_pct,2)
    except:
        overflg_pct=0.0


    flag_txt = "Flagged: {}".format(flg_pct)
    print(flag_txt)

    fig = plt.figure()

    ax_lw=3
    ax = plt.axes([0.1, 0.1, 0.898, 0.898])

    ax.imshow(np.log10(s_flag_sing),interpolation='nearest',aspect='auto',cmap='hot',vmin=-2.35,vmax=-1.35)
    ax.tick_params(axis='both',direction='in',width=2,length=8,top=True,right=True,pad=2)
    ax.spines['bottom'].set_linewidth(ax_lw)
    ax.spines['top'].set_linewidth(ax_lw)
    ax.spines['right'].set_linewidth(ax_lw)
    ax.spines['left'].set_linewidth(ax_lw)
    plt.ylim((140,100))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.set_yticklabels(['',105,110,115,120,125,130,135,140])
    plt.ylabel('Channel',fontsize=13)
    plt.xlabel('Time',fontsize=13)
    plt.text(50,105,ssplt_lbl,fontsize=22,bbox=dict(facecolor='white', alpha=1))
    plt.text(50,135,flag_txt,fontsize=16,bbox=dict(facecolor='white', alpha=1))
    plt.savefig(save_dir+fname,format='png')
    if Iplot:
        plt.show()
    plt.close()


    fname = fname_base+'_MS{}{}_MSSK.png'.format(ms0,ms1)
    noise = s_ave[20:100,:]

    mean = np.mean(noise)
    std = np.std(noise)

    noise = None

    #true_flags = np.zeros(s_ave.shape)
    #true_flags[s_ave > (mean + 4*std)] = 1

    rfi_flagged = np.copy(f_db[100:140,:])
    rfi_flagged[f[100:140,:]==0] = 0


    flg_pct = (100.*np.count_nonzero(rfi_flagged))/np.count_nonzero(f_db)
    flg_pct = np.around(flg_pct,2)

    rfi_overflagged = np.copy(-f_db[100:140,:]+1)
    rfi_overflagged[f[100:140,:]==0]=0

    try:
        overflg_pct = (100.*np.count_nonzero(rfi_overflagged))/np.count_nonzero(f_db[100:140,:])
        overflg_pct = np.around(overflg_pct,2)
    except:
        overflg_pct = 0.0

    #flag_txt = """MS flagged: {}%
#overflagged: {}%""".format(flg_pct,overflg_pct)

    flag_txt = "Flagged: {}".format(flg_pct)
    print(flag_txt)


    fig = plt.figure()

    ax_lw=3
    ax = plt.axes([0.1, 0.1, 0.898, 0.898])
    ax.imshow(np.log10(s_flag_both),interpolation='nearest',aspect='auto',cmap='hot',vmin=-2.35,vmax=-1.35)
    ax.tick_params(axis='both',direction='in',width=2,length=8,top=True,right=True,pad=2)
    ax.spines['bottom'].set_linewidth(ax_lw)
    ax.spines['top'].set_linewidth(ax_lw)
    ax.spines['right'].set_linewidth(ax_lw)
    ax.spines['left'].set_linewidth(ax_lw)
    plt.ylim((140,100))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.set_yticklabels(['',105,110,115,120,125,130,135,140])
    plt.ylabel('Channel',fontsize=13)
    plt.xlabel('Time',fontsize=13)
    plt.text(50,105,msplt_lbl,fontsize=22,bbox=dict(facecolor='white', alpha=1))
    plt.text(50,135,flag_txt,fontsize=16,bbox=dict(facecolor='white', alpha=1))
    #plt.savefig(save_dir+fname,format='png')
    if Iplot:
        plt.show()
    plt.close()




    #plot single scale SK results
    mid_chan = 120

    logs_ave = np.log10(s_ave)

    fname = fname_base+'_SSscatter.png'.format(ms0,ms1)

    fig = plt.figure()

    ax_lw=3
    ax = plt.axes([0.1, 0.1, 0.898, 0.898])

    ax.scatter(logs_ave[mid_chan,:],sk[mid_chan,:],marker='+',label='Center channel')
    for i in range(6):
        i+=1
        chans = np.r_[mid_chan+i,mid_chan-i]
        ax.scatter(logs_ave[chans,:],sk[chans,:],marker='+',label='+/- {}'.format(i))
    ax.scatter(logs_ave[50:53,:],sk[50:53,:],marker='+',label='Noise')

    ax.axhline(ut,linewidth=2,color='r',linestyle='--')
    ax.axhline(lt,linewidth=2,color='r',linestyle='--')
    ax.tick_params(axis='both',direction='in',width=2,length=8,top=True,right=True,pad=2)
    ax.spines['bottom'].set_linewidth(ax_lw)
    ax.spines['top'].set_linewidth(ax_lw)
    ax.spines['right'].set_linewidth(ax_lw)
    ax.spines['left'].set_linewidth(ax_lw)
    ax.text(0.15,0.85,ssplt_lbl,fontsize=22,transform=ax.transAxes)
    plt.legend()
    plt.ylabel('SK')
    plt.xlabel('Log Power')
    #plt.savefig(save_dir+fname,format='png')
    if Iplot:
        plt.show()
    plt.close()


    #plot multi-scale SK results

    # ms0 = 4 needs np.r_[mid_chan-(ms0-1) : mid_chan+1]
    mid_chans = np.r_[mid_chan-(ms0-1):mid_chan+1]
    #mid_chans1 = np.r_[117,119]
    #mid_chans2 = np.r_[118,120]

    logs_ave = np.log10(ms_s1)


    #plt.scatter(logs_ave[mid_chans,::2],ms_sk[mid_chans,::2],marker='+',label='Middle channels')
    #plt.scatter(logs_ave[mid_chans,1::2],ms_sk[mid_chans,1::2],marker='+',label='Middle channels')

    fname = fname_base+'_MS{}{}_MSscatter.png'.format(ms0,ms1)

    fig = plt.figure()

    ax_lw=3
    ax = plt.axes([0.1, 0.1, 0.8, 0.8])

    ax.scatter(logs_ave[mid_chans,:],ms_sk[mid_chans,:],marker='+',label='Middle channels')
    for i in range(6):
        i+=1
        chans = np.r_[mid_chan+i,mid_chan-(ms0-1)-i]
        ax.scatter(logs_ave[chans,:],ms_sk[chans,:],marker='+',label='+/- {}'.format(i))
    ax.scatter(logs_ave[50:53,:],ms_sk[50:53,:],marker='+',label='Noise')

    ax.axhline(ut,linewidth=2,color='r',linestyle='--')
    ax.axhline(lt,linewidth=2,color='r',linestyle='--')
    ax.tick_params(axis='both',direction='in',width=2,length=8,top=True,right=True,pad=2)
    ax.spines['bottom'].set_linewidth(ax_lw)
    ax.spines['top'].set_linewidth(ax_lw)
    ax.spines['right'].set_linewidth(ax_lw)
    ax.spines['left'].set_linewidth(ax_lw)
    ax.text(0.15,0.85,msplt_lbl,fontsize=22,transform=ax.transAxes)
    plt.legend()
    plt.ylabel('SK')
    plt.xlabel('Log Power')
    #plt.savefig(save_dir+fname,format='png')
    if Iplot:
        plt.show()
    plt.close()

    #os.system('say -volm 0.5 "doof"')
    #print('\a')
    #if np.random.randint(1,high=201) == 1:
    #    os.system('afplay ~/Desktop/memes/aiffs/legoyoda.aiff')
    #else:
    #    os.system('afplay /System/Library/Sounds/Sosumi.aiff')
    print("===========================================")
    print('Simulation done')
    print("===========================================")
