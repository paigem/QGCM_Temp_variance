import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
    
def confidence_envelope(z,w,spec,spec_std,n_std,avg_nums=np.array((3,5,9,17,33,51)),avg_percents=np.array((3,4,8,15,20,50))):
    
    # z is the % of confidence interval I want
    if z == 90:
        z = 1.645
    elif z == 95:
        z = 1.960
    elif z == 99:
        z = 2.576
    elif z == 80:
        z = 1.282
    elif z == 0: # i.e. just the standard error as I had for thesis
        z = 1
    
    one_percent = np.round(len(spec)/100)
    print('one_percent = ',one_percent)
    
    if np.sum(avg_percents) != 100:
        print('Error: avg_percents does not sum to 100%')
        
    bin_length = np.zeros(len(avg_nums))
        
    for ii in np.arange(len(avg_nums)):
        #if ii == 0:
        #    startk = 0
        #else:
        #    startk = len(avgd_spec)
        
        bin_length[ii] = np.round(avg_percents[ii]*one_percent)
        a = np.mod(bin_length[ii],avg_nums[ii])
        if a > 0:
            bin_length[ii] = bin_length[ii] - a
            
        bin_length = bin_length.astype(int)
            
        if ii == 0:
            temp = spec[:bin_length[ii]]
            temp_std = spec_std[:bin_length[ii]]
            temp_w = w[:bin_length[ii]]
        elif ii == len(avg_nums)-1:
            temp = spec[np.sum(bin_length[:ii]):]
            temp_std = spec_std[np.sum(bin_length[:ii]):]
            temp_w = w[np.sum(bin_length[:ii]):]
        else:
            temp = spec[np.sum(bin_length[:ii]):np.sum(bin_length[:ii])+bin_length[ii]]
            temp_std = spec_std[np.sum(bin_length[:ii]):np.sum(bin_length[:ii])+bin_length[ii]]
            temp_w = w[np.sum(bin_length[:ii]):np.sum(bin_length[:ii])+bin_length[ii]]
        
        l = len(temp_w[int(np.ceil(avg_nums[ii]/2))-1::avg_nums[ii]])
        
        # Declare this round of averaged w and spec
        avgd_w = np.zeros(l)
        avgd_spec = np.zeros(l)
        avgd_spec_std = np.zeros(l)
        min_line = np.zeros(l)
        max_line = np.zeros(l)
        
        avgd_w[:l] = temp_w[int(np.floor(avg_nums[ii]/2))::avg_nums[ii]]
        
        n_std_avg = np.sqrt((n_std**2)*avg_nums[ii])
        
        for k in np.arange(1,l+1):
            if k == l:
                avgd_spec[k-1] = np.mean(temp[(k-1)*avg_nums[ii]:])
                avgd_spec_std[k-1] = np.mean(temp_std[(k-1)*avg_nums[ii]:])
            else:
                avgd_spec[k-1] = np.mean(temp[(k-1)*avg_nums[ii]:k*avg_nums[ii]])
                avgd_spec_std[k-1] = np.mean(temp_std[(k-1)*avg_nums[ii]:k*avg_nums[ii]])
            
            min_line[k-1] = avgd_spec[k-1]-z*(avgd_spec_std[k-1]/n_std_avg)
            max_line[k-1] = avgd_spec[k-1]+z*(avgd_spec_std[k-1]/n_std_avg)
            
        
            
        if ii == 0:
            # Save avgd_spec and avgd_w in new arrays for concatenating
            avgd_w_final = np.copy(avgd_w)
            avgd_spec_final = np.copy(avgd_spec)
            min_line_final = np.copy(min_line)
            max_line_final = np.copy(max_line)
        else:
            avgd_w_final = np.concatenate((avgd_w_final,avgd_w),axis=0)
            avgd_spec_final = np.concatenate((avgd_spec_final,avgd_spec),axis=0)
            min_line_final = np.concatenate((min_line_final,min_line),axis=0)
            max_line_final = np.concatenate((max_line_final,max_line),axis=0)
                        
    return avgd_spec_final,avgd_w_final,min_line_final,max_line_final,bin_length