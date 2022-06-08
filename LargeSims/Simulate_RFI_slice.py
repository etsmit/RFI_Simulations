
#simulate a slice of RFI space
#uses multicore processing

#use $psrenv python

#first section of code is just the simulation function
#change sim parameters starting at line 420
#multicore processing starts at line 530

#flagging code can be replaced in lines 235 - 255

#==================================================
#imports

import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special
from scipy.signal import firwin, freqz, lfilter

import math
import os
import sys
import psutil

import time
import resource

from multiprocessing import Process,Queue

from Simulate_RFI_funcs import *
#==================================================
#functions

#put simulation thread in collection of multiprocessing threads
def put_q(q,a):
	q.put(do_sims())

#do each_core number of individual simulations
def do_sims():
	#s: random seed given by parent process
	#I know, global variables, but these will not change while code runs
	global s0_par
	global s1_par
	global slice_0
	global slice_1
	global sig_type
	global fft_type

	global fs
	global num_SK
	global fc_sig
	global fc_fsk_space
	global Nchan
	global num_spectra

	global each_core
	global out_flag_fname
	global cpu_list
	global num_cores

	global const_ms0
	global const_ms1
	global const_dc
	global const_dcper
	global const_SKm
	global const_ksps
	global const_wincut
	global const_SKsig
	global const_amp
	global const_adc_amp

	this_core_start_time = time.time()

	num_iters = each_core
	
	cpu_num = psutil.Process().cpu_num()
	print(cpu_num)
	cpu_list.append(cpu_num)
	this_fname_f = out_flag_fname[:-4]+'_cpu{}.npy'.format(cpu_num)
	this_fname_fpos = out_flag_fname[:-4]+'_fpos_cpu{}.npy'.format(cpu_num)
	this_fname_sk = out_flag_fname[:-4]+'_sktest_cpu{}.npy'.format(cpu_num)

	#max_m = slice_1[-1]

	const_bias = 0.71
	#output array of flagging and false pos for each set of sims
	out_flag = np.empty((len(slice_0),len(slice_1),num_iters),dtype=np.float32)
	out_fpos = np.empty((len(slice_0),len(slice_1),num_iters),dtype=np.float32)

	outsk = np.empty((5,len(slice_0),len(slice_1),num_iters,num_SK),dtype=np.float64)

	#for checking to see if we need to re-generate data
	prev_inputs = ['','','','','','','']


	#run through sims
	#kk: new independent sim
	for kk in range(num_iters):
		new_data = 1

		#move through 2-D experiment parameter space
		for ii in range(out_flag.shape[0]):
			for jj in range(out_flag.shape[1]):
			
				print('==========================')
				print('{}/{} || {}/{} || {}/{}'.format(ii+1,out_flag.shape[0],jj+1,out_flag.shape[1],kk+1,num_iters))

				# Deriving signal/flagging characteristics from par space
				#==============#
				if s1_par == 'SKm':
					SK_m = slice_1[jj]
				else:
					SK_m = const_SKm
				#==============#

				#==============#
				#symbol rate in ksps
				if s0_par == 'ksps':
					sym_rate = slice_0[ii]
				else:
					sym_rate = const_ksps
				#print('symbol rate {} kbps'.format(sym_rate))
				#==============#

				
				#derived number of samples per symbol
				ns_per_bit = int( fs / (sym_rate*1e3) )
				print('number of time samples per symbol: {}'.format(ns_per_bit))
				

				#==============#
				#multiscale (ms0 = channels, ms1 = time)
				if s1_par == 'mssk':
					ms0=int(slice_1[jj][0])
					ms1=int(slice_1[jj][1])
				else:
					ms0 = const_ms0
					ms1 = const_ms1
				#==============#

				#==============#
				if s0_par == 'askb':
					ask_bias = slice_0[ii]
				else:
					ask_bias = 0.71
				#==============#

				#==============#
				#duty cycle #ratio in (0,1]
				if s1_par == 'dc':
					dc = slice_1[jj]
				else:
					dc = const_dc
				dcper = const_dcper#ms
				#==============#

				#==============#
				#sK sigma threshold
				if s1_par == 'sk_sig':
					sksig = slice_1[jj]
				else:
					sksig = const_SKsig
				#==============#

				#==============#
				#bias test
				if s0_par == 'bias':
					bias = slice_0[ii]
				else:
					bias = const_bias
				#==============#


				#==============#
				#amplitude
				if s1_par == 'amp':
					amp = slice_1[jj]
				else:
					amp = const_amp
				#==============#

				adc_amp = const_adc_amp

				#power levels
				noise_db = 0.0
				sig_db = 0.0
				noise_linear = 10**(noise_db/10)
				sig_linear = 10**(sig_db)

				#number of spectra
				#max_m = np.max(slice_1)
				#print(f'max m: {max_m}')
				num_spectra = num_SK * SK_m
				#max_spectra = num_SK * max_m
				#print(num_spectra)

				#number of symbols
				num_bits = math.ceil((1.*num_spectra*Nchan)/ns_per_bit)
				#print(num_bits)

				#optimization: load/create x index array/carrier signal for re-use
				if (kk==0) and (ii==0) and (jj==0):
					print('creating first large array...')
					#x = np.arange(num_spectra*Nchan,dtype=np.uint32)
					#x = np.tile(np.arange(ns_per_bit),num_bits)
					#e_vec = np.exp(2.j*np.pi*fc_sig*x*ts)
					x = np.load('/home/scratch/esmith/big_x60.npy')
					e_vec = np.load('/home/scratch/esmith/big_evec60.npy')
					#fir_sz = int(0.2*ns_per_bit)
					#sinc = scipy.signal.firwin(fir_sz, cutoff=1.0/fir_sz, window="rectangular")
					#np.save('big_x60.npy',x)
					#np.save('big_evec60.npy',e_vec)

				


				#if running with variable sized x/e_vec
				this_x = x[:num_bits*ns_per_bit]
				this_evec = e_vec[:num_bits*ns_per_bit]


				#determine SK thresholds
				SK_p = (1-scipy.special.erf(sksig/math.sqrt(2))) / 2
				print(f'SK sigma: {sksig} || theory FPR: {SK_p}')
				lt,ut=SK_thresholds(int(SK_m),p=SK_p)
				#print(lt)
				#print(ut)

				#don't need to remake dataset if all we change is MS shape
				this_inputs = [SK_m,sym_rate,bias,dc,amp,ms0,ms1]
				print(prev_inputs)	
				print(this_inputs)
				if this_inputs[:-4] == prev_inputs[:-4]:
					new_data=0
				else:
					new_data=1
				
				if this_inputs[-4] == prev_inputs[-4]:
					new_dc=0
				else:
					new_dc=1
				if this_inputs[-3] == prev_inputs[-3]:
					new_amp=0
				else:
					new_amp=1
				prev_inputs = this_inputs
				print(new_data)
				print(new_dc)

				#===============================
				#Creating and flagging data
				if new_data:
					if sig_type == 'bfsk':
						y,sym,f,p = bfsk_fast(num_bits,sym_rate,fc_sig,fc_fsk_space,const_wincut,Ebit=0.0,N0=None,fs=fs)
					elif sig_type == 'ask':
						#y,sym,bits = ask_mod(x,e_vec,num_bits,ns_per_bit,const_wincut,fc_sig,bias,Ebit=0.0,N0=None,fs=fs)
						y,sym,bits = ask_fast(this_x,this_evec,num_bits,ns_per_bit,2,const_wincut,fc_sig,Ebit=0.0,N0=None,fs=fs)
					elif sig_type == 'bpsk':
						print('making bpsk signal...')
						y,sym,bits = bpsk_alt(this_x,this_evec,num_bits,sym_rate,const_wincut,fc_sig,Ebit=0.0,fs=fs)
					elif sig_type == 'qpsk':
						y,sym,bits = qpsk(x,num_bits,sym_rate,const_wincut,fc_sig,Ebit=0.0,fs=fs)

					sym=None
					bits=None

				#print(len(y))
				if new_dc:
					#print('new dc')
					ydc = duty_cycle(y,dc,dcper,fs=fs)
				else:
				#	#print('same dc')
					ydc = y
				#y = None


				if new_data or new_dc or new_amp:
					#truncate extra data
					ydc = ydc[:(Nchan*num_spectra)]
					#print(len(y))
					#make complex noise, mean 0 std 1
					n = np.random.RandomState().normal(0,adc_amp,size=len(ydc)).astype(np.int8) + 1.j*np.random.RandomState().normal(0,adc_amp,size=len(ydc)).astype(np.int8)

					#ramp up signal strength from 0 to 1
					ramp = np.arange(len(ydc))/(len(ydc))
					fb_shape = (num_spectra,Nchan)
					#print(num_spectra)
					ydc *= ramp
					#ydc *= amp
					ramp = None
					ydc.real = (adc_amp*ydc.real).astype(np.int8)
					ydc.imag = (adc_amp*ydc.imag).astype(np.int8)

					#channelize/square the data
					#make -10dB mask (channelizes y,n separately)
					if fft_type == 'pfb':
						M=24
						P=Nchan
						#print(len(y))
						fb = pfb_filterbank(n+ydc, M, P)
						#print(fb.shape)
						s = np.zeros(fb_shape,dtype=np.complex64)
						#account for M less spectra in pfb
						#copy last spectrum M times
						s[:-M,:] = fb
						s[-M:,:] = fb[-1,:]
						s = np.abs(s.T)**2
						fb = None
						f_db,s_db = power_mask_pfb(ydc,n,Nchan,SK_m,num_SK,M,P)

					elif fft_type == 'fft':
						fb = np.fft.fft(np.reshape(n+y,fb_shape),axis=1)
						s = np.abs(fb.T)**2
						f_db,s_db = power_mask_fft(y,n,Nchan,SK_m,num_SK)

					#clean up mem usage
					s_db=None

					#y = None
					n = None
					#fb = None


				#==========================#
				# RFI flagging

				#single-scale SK
				sk,s_ave = SK_master(s,SK_m)

				if s1_par=='amp':
					outsk[:,ii,jj,kk,:] = sk[115:120,:]


				#multi-scale SK
				ms_sk,ms_f,ms_s1 = ms_SK_master(s,SK_m,ms0,ms1,lt,ut)
				#s = None

				#single-scale SK flags
				f = np.zeros(f_db.shape)
				f[sk>ut]=1
				f[sk<lt]=1

				#add union of multiscale SK flags
				for ichan in range(ms0):
					for itime in range(ms1):
						f[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(num_SK-(ms1-1))][ms_f==1] = 1
				ms_f = None
				sk = None
				ms_sk = None
				#should leave this section with bool flagging array 'f'
				#==========================#



				#count up %flagged and false positive rate
				#rfi_flagged = np.copy(f_db)
				#rfi_flagged[f==0] = 0
				#flg_pct = (100.*np.count_nonzero(rfi_flagged))/np.count_nonzero(f_db)

				#flg_pct = np.around(flg_pct,2)

				# rfi_fpos = np.copy(-f_db+1)
				# rfi_fpos[f==0]=0

				# fpos_pct = (100.*np.count_nonzero(rfi_fpos[100:140]))/f_db[100:140].size
				# fpos_pct = np.around(fpos_pct,2)

				#from ryan
				tp = np.sum((f == True)&(f_db == True))
				tn = np.sum((f == False)&(f_db == False))
				fp = np.sum((f == True)&(f_db == False))
				fn = np.sum((f == False)&(f_db == True))
				flg_pct = (100.*tp/(tp+fn))
				flg_pct = np.around(flg_pct,2)
				fpos_pct = (100.*fp/(fp+tn))
				fpos_pct = np.around(fpos_pct,2)



				#send flagging results to output arrays, check mem usage
				print('Flagged: {} || False pos: {}'.format(flg_pct,fpos_pct))
				out_flag[ii,jj,kk] = flg_pct
				out_fpos[ii,jj,kk] = fpos_pct
				mem_gb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1e6
				print('mem usage: {} GB'.format(np.around(mem_gb,4)))


		#if (kk==0) and os.path.exists(this_fname_f):
		#make sure of unique intermediate files despite same cpu_num across threads
		check=0
		if (kk==0):
			while cpu_num in cpu_list:
				cpu_num = np.random.RandomState().randint(1,99)
				check +=1
				if check > 100:
					print('cant find good output filename...')
					break
			cpu_list.append(cpu_num)
			this_fname_f = this_fname_f[:-4]+f'{cpu_num}.npy'
			this_fname_fpos = this_fname_fpos[:-4]+f'{cpu_num}.npy'
			this_fname_sk = this_fname_sk[:-4]+f'{cpu_num}.npy'
		print('{} saving to {}'.format(cpu_num,this_fname_f))

		np.save(this_fname_f,out_flag)
		np.save(this_fname_fpos,out_fpos)
		np.save(this_fname_sk,outsk)

	print(f'{cpu_num} done!')
	this_core_end_time = time.time()
	print('core time: {} min'.format((this_core_end_time - this_core_start_time)/60))
	#send output arrays to get stitched with the other parallel processes
	return out_flag, out_fpos






#==================================================
#inits

out_flag_fname = sys.argv[1]
out_fpos_fname = out_flag_fname[:-4]+'_fpos.npy'

cpu_list = []
my_clrs = ["#0700c7","#08c40e","#de0000","#00e6da","#b57026","#f56ce0","#666666"]

#Number of SK realizations for uncertainties (in each single RFI configuration)
N_reals = 100

Nchan=256
#Nchan 512 - 200kHz freq res
#Nchan=4096
#4096 - 25kHz

#number of SK blocks
num_SK = 60

#sampling rate
fs=50e6
#fs=200e6
print('freq res: {} kHz'.format(1e-3*(fs/Nchan)))

#carrier channels of signals
cc_sig = 120
cc_tone = 200

cc_fsk_space = 130

#carrier frequencies of signals
fc_sig = cc_sig*(fs/Nchan)
fc_tone = cc_tone*(fs/Nchan)

fc_fsk_space = cc_fsk_space*(fs/Nchan)


print('tone: {} MHz || signal: {} MHz'.format(fc_tone/1e6,fc_sig/1e6))
print('FSK space frequency: {} MHz'.format(fc_fsk_space/1e6))


ts=1/fs

const_ms0 = 4
const_ms1 = 2
const_amp = 1
const_adc_amp=16

const_dc = 1
const_dcper = 1
const_SKm = 512

const_ksps = 20
const_wincut = 1.0

const_SKsig = 3.0

#======================================
#set up parameter space configuration

#slice parameters can be 'ksps','askb','dc','SKm','mssk'
#if using mssk or dc, put in slice 1 for optimization

#slice 0 parameter
s0_par = 'ksps'

#slice 1 parameter
s1_par = 'SKm'

#0:
slice_0 = np.array([1,4,20,100,200])# data rate ksps
#slice_0 = np.array([40,45,50,55,60,65,70,75,80,85,90,95,100])# data rate ksps
#slice_0 = np.around(np.arange(0.05,1.01,0.05),2)# duty cycles (5%)
#slice_0 = np.array([128,256,512,1024,2048,4096])# SK_m
#slice_0 = np.around(np.arange(0,1.01,0.1),2)

#1:
#slice_1 = np.around(np.arange(0.02,1.01,0.02),2)# duty cycles (2%)
#slice_1 = np.around(np.arange(0.05,1.01,0.05),2)# duty cycles (5%)
#slice_1 = np.around(np.arange(0.05,1.01,0.05),2)# amplitudes (5%)
#slice_1 = np.array([128,256,512,1024,2048,4096])# SKm
slice_1 = np.array([128,256,512,1024,2048])# SKm
#slice_1 = np.array(['11','12','21','22','24','42','44'])# mssk shape
#slice_1 = np.arange(0.2,6.1,0.4)# sk sigma threshold

#max_m = slice_1[-1]

#setup final output flag arrays
res_flag = np.empty((len(slice_0),len(slice_1),N_reals))
res_fpos = np.empty((len(slice_0),len(slice_1),N_reals))


#how is data channelized
#'pfb': 24-tap hanning pfb
#'fft': flat fft
fft_type = 'pfb'


#what signal modulation type? 'bpsk','bfsk','ask','qpsk'
sig_type = 'bpsk'



#number of multiprocessing cores to use (warning: no mem shared between them)
num_cores = 10





#======================================
#start actually doing things
start_time = time.time()

my_q = Queue()

mp_list = []

#start each multiprocess with different random seed

#each core does x iterations
each_core = int(N_reals/num_cores)


#check inputs/constants against what filename was input

print('=================================\n ---- Sanity Check ----')
print('Output filenames:')
print(out_flag_fname)
print(out_fpos_fname)
print('---------------------')
print('Modulation:     '+sig_type)
print('Slice 0:        '+s0_par)
print(f'  Range:        {slice_0}')
print('Slice 1:        '+s1_par)
print(f'  Range:        {slice_1}')
print('---------------------')
print(f'SigChan/Nchan:  {cc_sig}||{cc_fsk_space}||{Nchan}')
if (s0_par != 'SKm') and (s1_par != 'SKm'):
	print(f'M:              {const_SKm}')
if (s0_par != 'mssk') and (s1_par != 'mssk'):
	print(f'MSSK:           {const_ms0}{const_ms1}')
if (s0_par != 'dc') and (s1_par != 'dc'):
	print(f'dc:             {const_dc}')
if (s0_par != 'ksps') and (s1_par != 'ksps'):
	print(f'sym rate:       {const_ksps}')

print(f'wincut:         {const_wincut}')
print(f'dcper:          {const_dcper}')
print(f'samp rate:      {fs/1e6} MHz')
print('---------------------')
print(f'Using {num_cores} cores')
input('Everything look OK? Press enter...')


for p in range(num_cores):
	this_p = Process(target=put_q,args=(my_q,1))
	mp_list.append(this_p)


print('starting processes...')
for mp in mp_list:
	mp.start()

print('joining processes...')
for mp in mp_list:
	mp.join()

print('stitching together results...')
for pp in range(num_cores):
	#this_mp = mp_list[pp]
	this_f,this_of = my_q.get()
	res_flag[:,:,each_core*pp:each_core*(pp+1)] = this_f
	res_fpos[:,:,each_core*pp:each_core*(pp+1)] = this_of



print('=================================\n ---- Sanity Check ----')
print('Output filenames:')
print(out_flag_fname)
print(out_fpos_fname)
print('---------------------')
print('Modulation:     '+sig_type)
print('Slice 0:        '+s0_par)
print(f'  Range:        {slice_0}')
print('Slice 1:        '+s1_par)
print(f'  Range:        {slice_1}')
print('---------------------')
print(f'SigChan/Nchan:  {cc_sig}/{Nchan}')
if (s0_par != 'SKm') or (s1_par != 'SKm'):
	print(f'dc:             {const_SKm}')
if (s0_par != 'mssk') or (s1_par != 'mssk'):
	print(f'MSSK:           {const_ms0}{const_ms1}')
if (s0_par != 'dc') or (s1_par != 'dc'):
	print(f'dc:             {const_dc}')
if (s0_par != 'ksps') or (s1_par != 'ksps'):
	print(f'sym rate:       {const_ksps}')

print(f'wincut:         {const_wincut}')
print('---------------------')
print(f'Using {num_cores} cores')


np.save(out_flag_fname,res_flag)
print('flagging results saved to '+out_flag_fname)
np.save(out_fpos_fname,res_fpos)
print('fpos results saved to '+out_fpos_fname)

end_time = time.time()

print('total time: {} min'.format((end_time - start_time)/60))









