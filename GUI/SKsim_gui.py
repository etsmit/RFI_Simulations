#This program does Master_notebook.ipynb but in gui form

import numpy as np
import tkinter as tk

import matplotlib.pyplot as plt

import os,sys
import time

#import glob


import scipy as sp
import scipy.optimize
import scipy.special

import math

from functools import partial


from SKsim_funcs import *

#exec(open("./current_split_labels.py").read())
#==============================
#inits


#colors
dark_bg = '#0050a1'
darker_bg = '#003c78'
liter_bg = '#004ddb'
white = '#FFFFFF'
dk_grayblue = '#23242C'


save_dir = '/Users/evansmith/WVU/RFI_MIT/Simulations/pics_for_pub/'
save_dir = ''

prev_inputs=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#==============================








#==============================
#functions

def sim_start():
    #starts simulation with parameters
    #collect inputs and put correct type
    #there's gotta be a better way to do this but haven't researched enough
    global prev_inputs
    global current_encoding
    global Nchan
    global SK_m
    global fs
    global cc_sig
    global cc_tone
    global cc_space
    global sym_rate
    global bits_per_sym
    global num_SK
    global SKsigma
    global ms0
    global ms1
    global ask_bias
    global dc
    global dcper
    global wincut
    global Iplot
    global ndata
    global spect_lbl
    global sdb_lbl
    global ssplt_lbl
    global msplt_lbl
    this_encode = current_encoding.get()
    this_Nchan = int(Nchan.get())
    this_SK_m = int(SK_m.get())
    this_fs = float(fs.get())
    this_cc_sig = float(cc_sig.get())
    this_cc_tone = float(cc_tone.get())
    this_cc_space = float(cc_space.get())
    this_sym_rate = float(sym_rate.get())
    this_bits = int(bits_per_sym.get())
    this_num_SK = int(num_SK.get())
    this_SKsigma = float(SKsigma.get())
    this_ms0 = int(ms0.get())
    this_ms1 = int(ms1.get())
    this_ask_bias = float(ask_bias.get())
    this_dc = float(dc.get())
    this_dcper = float(dcper.get())
    this_wincut = float(wincut.get())
    this_Iplot = bool(int(Iplot.get()))
    this_ndata = bool(int(ndata.get()))
    #this_spect_lbl = '('+spect_lbl.get()+')'
    #this_ssplt_lbl = '('+ssplt_lbl.get()+')'
    #this_msplt_lbl = '('+msplt_lbl.get()+')'
    this_spect_lbl = spect_lbl.get()
    this_sdb_lbl = sdb_lbl.get()
    this_ssplt_lbl = ssplt_lbl.get()
    this_msplt_lbl = msplt_lbl.get()
    #determine new_data, if we need to remake the signal
    this_inputs = [this_encode,this_Nchan,this_SK_m,this_fs,this_cc_sig,this_cc_tone,this_cc_space,this_sym_rate,this_bits,this_num_SK,this_ask_bias,this_dc,this_dcper,this_wincut]
    print('Old inputs:{}'.format(prev_inputs))
    print('New inputs:{}'.format(this_inputs))
    this_newdata = 0
    for i in range(len(this_inputs-3)):
        if this_inputs[i] != prev_inputs[i]:
            this_newdata = 1

    if this_ndata == 1:
        this_newdata = 1

    this_newdc = (this_inputs[-3] != prev_inputs[-3])
    prev_inputs = this_inputs
    print(this_newdata)
    simulate(this_encode,this_Nchan,this_SK_m,this_fs,this_cc_sig,this_cc_tone,this_cc_space,this_sym_rate,this_bits,this_num_SK,this_SKsigma,this_ms0,this_ms1,this_ask_bias,this_dc,this_dcper,this_wincut,this_newdata,this_newdc,this_Iplot,this_spect_lbl,this_sdb_lbl,this_ssplt_lbl,this_msplt_lbl)




#==============================










#================================
#Make GUI




root = tk.Tk()
root.title('SK Simulations')
root.configure(bg=dark_bg)


#RFI characteristics frame
RFI_frame = tk.Frame(root,bd=3,relief='sunken',bg=dk_grayblue)
RFI_frame.grid(column=1,row=1)

#SK stuff frame
SK_frame = tk.Frame(root,bd=3,relief='sunken',bg=dk_grayblue)
SK_frame.grid(column=1,row=2)

#Sim stuff frame
Sim_frame = tk.Frame(root,bd=3,relief='sunken',bg=dk_grayblue)
Sim_frame.grid(column=2,row=1)




#RFI stuff
encode_options_label = tk.Label(RFI_frame,text = 'Encoding: ',fg=white,bg=dk_grayblue)
encode_options_label.grid(column=1,row=1)

encode_options_list = ['BASK','BPSK','QPSK','BFSK']
current_encoding = tk.StringVar()
current_encoding.set('BPSK')

encode_options_dropdown = tk.OptionMenu(RFI_frame,current_encoding,*encode_options_list)
encode_options_dropdown.grid(column=2,row=1)

#signal channel
cc_sig = tk.StringVar()
cc_sig_label = tk.Label(RFI_frame,text='cc_sig (chan)',fg=white,bg=dk_grayblue)
cc_sig_label.grid(column=1,row=2)
cc_sig_entry = tk.Entry(RFI_frame,textvariable=cc_sig)
cc_sig_entry.grid(column=2,row=2)
cc_sig.set('120')

#test tone channel
cc_tone = tk.StringVar()
cc_tone_label = tk.Label(RFI_frame,text='cc_tone (chan)',fg=white,bg=dk_grayblue)
cc_tone_label.grid(column=1,row=3)
cc_tone_entry = tk.Entry(RFI_frame,textvariable=cc_tone)
cc_tone_entry.grid(column=2,row=3)
cc_tone.set('200')

#FSK space channel
cc_space = tk.StringVar()
cc_space_label = tk.Label(RFI_frame,text='BFSK cc_space (chan)',fg=white,bg=dk_grayblue)
cc_space_label.grid(column=1,row=4)
cc_space_entry = tk.Entry(RFI_frame,textvariable=cc_space)
cc_space_entry.grid(column=2,row=4)
cc_space.set('130')

#symbol rate
sym_rate = tk.StringVar()
sym_rate_label = tk.Label(RFI_frame,text='Symbol rate (kbps)',fg=white,bg=dk_grayblue)
sym_rate_label.grid(column=1,row=5)
sym_rate_entry = tk.Entry(RFI_frame,textvariable=sym_rate)
sym_rate_entry.grid(column=2,row=5)
sym_rate.set('1')

#duty cycle
dc = tk.StringVar()
dc_label = tk.Label(RFI_frame,text='Duty cycle (0-1)',fg=white,bg=dk_grayblue)
dc_label.grid(column=1,row=6)
dc_entry = tk.Entry(RFI_frame,textvariable=dc)
dc_entry.grid(column=2,row=6)
dc.set('1')

#duty cycle period
dcper = tk.StringVar()
dcper_label = tk.Label(RFI_frame,text='Duty cycle period (ms)',fg=white,bg=dk_grayblue)
dcper_label.grid(column=1,row=7)
dcper_entry = tk.Entry(RFI_frame,textvariable=dcper)
dcper_entry.grid(column=2,row=7)
dcper.set('1')

#ASK power level difference (ask_bias = power for '0' bit)
ask_bias = tk.StringVar()
ask_bias_label = tk.Label(RFI_frame,text='BASK bias',fg=white,bg=dk_grayblue)
ask_bias_label.grid(column=1,row=8)
ask_bias_entry = tk.Entry(RFI_frame,textvariable=ask_bias)
ask_bias_entry.grid(column=2,row=8)
ask_bias.set('0.71')

#smoothing window cutoff frequency
wincut = tk.StringVar()
wincut_label = tk.Label(RFI_frame,text='Window Cutoff',fg=white,bg=dk_grayblue)
wincut_label.grid(column=1,row=9)
wincut_entry = tk.Entry(RFI_frame,textvariable=wincut)
wincut_entry.grid(column=2,row=9)
wincut.set('1.0')

#number of bits per symbol (1 is binary 010101010001011...., 2 is quadrature 11-01-10-00-01-11-01-...)
bits_per_sym = tk.StringVar()
bits_per_sym_label = tk.Label(RFI_frame,text='Bit level',fg=white,bg=dk_grayblue)
bits_per_sym_label.grid(column=1,row=10)
bits_per_sym_entry = tk.Entry(RFI_frame,textvariable=bits_per_sym)
bits_per_sym_entry.grid(column=2,row=10)
bits_per_sym.set('1')



#SK stuff

#SK M
SK_m = tk.StringVar()
SK_m_label = tk.Label(SK_frame,text='SK_m',fg=white,bg=dk_grayblue)
SK_m_label.grid(column=1,row=1)
SK_m_entry = tk.Entry(SK_frame,textvariable=SK_m)
SK_m_entry.grid(column=2,row=1)
SK_m.set('512')

#multiscale m
ms0 = tk.StringVar()
ms0_label = tk.Label(SK_frame,text='ms0',fg=white,bg=dk_grayblue)
ms0_label.grid(column=1,row=2)
ms0_entry = tk.Entry(SK_frame,textvariable=ms0)
ms0_entry.grid(column=2,row=2)
ms0.set('1')

#multiscale n
ms1 = tk.StringVar()
ms1_label = tk.Label(SK_frame,text='ms1',fg=white,bg=dk_grayblue)
ms1_label.grid(column=1,row=3)
ms1_entry = tk.Entry(SK_frame,textvariable=ms1)
ms1_entry.grid(column=2,row=3)
ms1.set('1')

#number of SK bins
num_SK = tk.StringVar()
num_SK_label = tk.Label(SK_frame,text='num_SK',fg=white,bg=dk_grayblue)
num_SK_label.grid(column=1,row=4)
num_SK_entry = tk.Entry(SK_frame,textvariable=num_SK)
num_SK_entry.grid(column=2,row=4)
num_SK.set('300')

#SK sigma flagging
SKsigma = tk.StringVar()
SKsigma_label = tk.Label(SK_frame,text='SKsigma',fg=white,bg=dk_grayblue)
SKsigma_label.grid(column=1,row=5)
SKsigma_entry = tk.Entry(SK_frame,textvariable=SKsigma)
SKsigma_entry.grid(column=2,row=5)
SKsigma.set('3')



#Sim_frame stuff

#number of channels
Nchan = tk.StringVar()
Nchan_label = tk.Label(Sim_frame,text='Nchan',fg=white,bg=dk_grayblue)
Nchan_label.grid(column=1,row=1)
Nchan_entry = tk.Entry(Sim_frame,textvariable=Nchan)
Nchan_entry.grid(column=2,row=1)
Nchan.set('256')

#sampling rate
fs = tk.StringVar()
fs_label = tk.Label(Sim_frame,text='fs',fg=white,bg=dk_grayblue)
fs_label.grid(column=1,row=2)
fs_entry = tk.Entry(Sim_frame,textvariable=fs)
fs_entry.grid(column=2,row=2)
fs.set('50e6')

#show interactive plots while simulating?
Iplot = tk.StringVar(value='0')
Iplot_label = tk.Label(Sim_frame,text='Interactive plots',fg=white,bg=dk_grayblue)
Iplot_label.grid(column=1,row=3)
Iplot_check = tk.Checkbutton(Sim_frame,variable=Iplot,onvalue='1',offvalue='0')

Iplot_check.grid(column=2,row=3)


#remake data set?
ndata = tk.StringVar(value='0')
ndata_label = tk.Label(Sim_frame,text='New Data',fg=white,bg=dk_grayblue)
ndata_label.grid(column=1,row=4)
ndata_check = tk.Checkbutton(Sim_frame,variable=ndata,onvalue='1',offvalue='0')

ndata_check.grid(column=2,row=4)


#label for unflagged spectra
spect_lbl = tk.StringVar(value='(a)')
spect_lbl_label = tk.Label(Sim_frame,text='spect_lbl',fg=white,bg=dk_grayblue)
spect_lbl_label.grid(column=1,row=5)
spect_lbl_entry = tk.Entry(Sim_frame,textvariable=spect_lbl)
spect_lbl_entry.grid(column=2,row=5)

#label for comparison mask spectra
sdb_lbl = tk.StringVar(value='(b)')
sdb_lbl_label = tk.Label(Sim_frame,text='sdb_lbl',fg=white,bg=dk_grayblue)
sdb_lbl_label.grid(column=1,row=6)
sdb_lbl_entry = tk.Entry(Sim_frame,textvariable=sdb_lbl)
sdb_lbl_entry.grid(column=2,row=6)

#label for ss plot
ssplt_lbl = tk.StringVar(value='(c)')
ssplt_lbl_label = tk.Label(Sim_frame,text='ssplt_lbl',fg=white,bg=dk_grayblue)
ssplt_lbl_label.grid(column=1,row=7)
ssplt_lbl_entry = tk.Entry(Sim_frame,textvariable=ssplt_lbl)
ssplt_lbl_entry.grid(column=2,row=7)

#label for ms plot
msplt_lbl = tk.StringVar(value='(d)')
msplt_lbl_label = tk.Label(Sim_frame,text='msplt_lbl',fg=white,bg=dk_grayblue)
msplt_lbl_label.grid(column=1,row=8)
msplt_lbl_entry = tk.Entry(Sim_frame,textvariable=msplt_lbl)
msplt_lbl_entry.grid(column=2,row=8)




#Extra stuff


tk.Button(root,text='Simulate!',command=sim_start,fg=white,bd=3,relief='sunken',bg=dk_grayblue).grid(column=2,row=2)


#tk.Button(root,text='Exit',command=,fg=white,bg=darker_bg).grid(column=2,row=2))





#bindings
root.bind('<Return>',sim_start)



















root.mainloop()
