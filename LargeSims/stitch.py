#stitch together results
#in the case that Simulate_RFI_slice.py hangs after each thread is done
#happens when too much data enters each multithread? not sure


#put filename pattern with cpu* as first argument in quotes,
#and the intended final results filename
#like so:
#$ python stitch.py "f_ksps_dc__bpsk_m512_chan120_dcper1_cut1_SS_8bit_fpos_cpu*.npy" f_ksps_dc__bpsk_m512_chan120_dcper1_cut1_SS_8bit_fpos.npy

#need to make sure each cpu result file has the right first 2 shape dimensions of out_shape
#and each_core matches what you put in Simulate_RFI_slice


import numpy as np
import os, sys
import glob

in_patt = sys.argv[1]

out_fname = sys.argv[2]

print(in_patt)


files = glob.glob(in_patt)

out_shape = (5,20,100)
#out_shape = (5,5,20,100,60)
each_core = 10

res_flag = np.zeros(out_shape)

files = files[:10]

print(f'{len(files)} files found')


print('stitching together results...')
for pp in range(len(files)):
	this_data = np.load(files[pp])
	print(f'{this_data.shape} || {files[pp]}')
	res_flag[:,:,each_core*pp:each_core*(pp+1)] = this_data
	#res_flag[:,:,:,each_core*pp:each_core*(pp+1),:] = this_data

print('saving to {}'.format(out_fname))
np.save(out_fname,res_flag)

