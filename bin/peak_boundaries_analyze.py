import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import re
import os
from collections import namedtuple
import utilities_pmr as pmr

Row = namedtuple('Row', 
[
    'Compound', 
    'Times',
    'Intensities', 
    'RT', 
    'StartRT',
    'StopRT',
    'PrecursorMz', 
    'PrecursorZ', 
    'FragmentIon', 
    'ProductMz', 
    'ProductZ', 
    'FWHM',
    'Concentration',
    'Replicate',
    'FileName',
    'Background',
])

MIN_RT_WIDTH = 0.2
RT_DIFF_THRESH = 0.2
TRANS_AREA_THRESH = 0.3
EXTRA_WIDTH_MULTIPLIER = 0.75

HighConcInfo = namedtuple('HighConcInfo', 
                          ['adjusted_rts', 
                           'med_half_width', 
                           'normalized_trans_area', 
                           'med_trans_areas',
                           'made_rt_adjustment',
                           'large_trans_error',
                          ])

Boundary = namedtuple('Boundary', ['FileName', 'Compound', 'Start', 'Stop'])

def compute_median_time_trace(matrix):
    num_rows,num_cols = matrix.shape
    # minimum intensity is 1 ion/s
    ones = np.ones(num_rows)
    # maximum for each row
    max_for_rows = np.maximum(ones, np.amax(matrix, axis=1))
    max_mat = np.empty_like(matrix)
    for cidx in range(num_cols):
        max_mat[:, cidx] = max_for_rows
    # finish the normalized matrix
    normalized_mat = matrix / max_mat
    # median trace
    medians = np.empty(num_cols)
    for cidx in range(num_cols):
        medians[cidx] = np.median(normalized_mat[:, cidx])
    return medians,normalized_mat
    
def compute_matrix_stats(matrix):
    '''
        Given a matrix of mz rows x time step cols,
        compute the median normalized time trace,
        and then compute the correlation of all the
        transitions to that trace
    '''
    median_time_trace,normalized_mat = compute_median_time_trace(matrix)
    # protect against issues when the signal was terrible and the median trace was all 0's (or all 1's due to protection)
    # in this case make a random vector which will have bad correlations with the other tranes
    med_std = np.std(median_time_trace)
    if med_std <= 0.0:
        median_time_trace = np.random.uniform(size=len(median_time_trace))
    mat_std = np.std(normalized_mat, axis=1)
    for ridx,std in enumerate(mat_std):
        if std <= 0.0:
            normalized_mat[ridx, :] = np.random.uniform(size=normalized_mat.shape[1])
        
    # make one matrix with the median trace as the first row
    mat_with_median = np.empty((normalized_mat.shape[0]+1, normalized_mat.shape[1]))
    mat_with_median[0, :] = median_time_trace
    mat_with_median[1:, :] = normalized_mat
    
    # correlation coefficients, remove the first one for the median to itself
    # the other values are the correlations of the other rows to the median
    corrcoeffs = np.corrcoef(mat_with_median)[0, 1:]
    num_rows = matrix.shape[0]
    areas = np.empty(num_rows)
    for ridx in range(num_rows):
        row = matrix[ridx, :]
        areas[ridx] = max(1, np.sum(row))
    
    max_area = max(1.0, np.max(areas))
    rel_areas = [x/max_area for x in areas]
    return corrcoeffs, areas, rel_areas

def create_transition_matrix(transitions, half_window_min=0.25):
    '''
        Input a list of rows from a dataframe for a particular precursor.
        Put the transitions into a matrix of number of transitions rows by 
        number of time steps columns
    '''
    data_by_conc = {}
    for trans in transitions:
        intensities = [float(x) for x in trans['Raw Intensities'].strip().split(',')]
        retention_times = [float(x) for x in trans['Raw Times'].strip().split(',')]
        compound = trans['Peptide Modified Sequence']
        rt = trans['Retention Time']
        start_rt = trans['Start Time']
        stop_rt = trans['End Time']
        precursor_mz = trans['Precursor Mz']
        precursor_z = trans['Precursor Charge']
        fragment_ion = trans['Fragment Ion']
        fragment_mz = trans['Product Mz']
        fragment_z = trans['Product Charge']
        fragment_fwhm = trans['Fwhm']
        conc = trans['Analyte Concentration']
        rep = trans['Replicate Name']
        file_name = trans['File Name']
        background = trans['Background']
        
        if not conc in data_by_conc:
            data_by_conc[conc] = {}
        if not rep in data_by_conc[conc]:
            data_by_conc[conc][rep] = []
            
        data_by_conc[conc][rep].append(Row(
            compound, retention_times, intensities, 
            rt, start_rt, stop_rt, 
            precursor_mz, precursor_z, 
            fragment_ion, fragment_mz, fragment_z, fragment_fwhm, 
            conc, rep, file_name, background))
    
    results_by_conc_rep = {}
    for conc,rep_data in data_by_conc.items():
        results_by_conc_rep[conc] = {}
        for rep,raw_data, in rep_data.items():
            # sort by product mz
            raw_data.sort(key=lambda x:x.ProductMz)
            num_columns = len(raw_data[0].Times)
            num_rows = len(raw_data)

            if num_rows == 0 or num_columns == 0:
                print('Zero size matrix for {}'.format(first_trans['Peptide Modified Sequence']))
                return False

            # create the transition matrix
            matrix = np.empty((num_rows, num_columns)) 
            for idx,data in enumerate(raw_data):
                matrix[idx, :] = data.Intensities[:num_columns]
            
            results_by_conc_rep[conc][rep] = {
                'matrix': matrix, 'raw_data': raw_data
            }
    return results_by_conc_rep

def load_file(file_name):
    '''
        Separate into data by sequence/charge, where each one will have
        a matrix of intensities for its transitions.
    '''
    df = pd.read_csv(file_name, delimiter='\t')
    data_by_sequence_charge = {}
    for _,row in df.iterrows():
        seq_charge = row['Peptide Modified Sequence'] + '/' + str(row['Precursor Charge'])     
        if not seq_charge in data_by_sequence_charge:
            # there could be multiple files with the same peptide I guess
            data_by_sequence_charge[seq_charge] = []
        data_by_sequence_charge[seq_charge].append(row)
    
    # now the data for each precursor are aggregated.  Create the transition matrices
    mats_by_seq_charge = {}
    for seqcharge,data in data_by_sequence_charge.items():
        mats = create_transition_matrix(data)
        if mats:
            mats_by_seq_charge[seqcharge] = mats
    return mats_by_seq_charge

def check_index_width(start_idx, stop_idx, size):
    width = stop_idx - start_idx
    if width < 1:
        start_idx = max(0, start_idx-1)
    width = stop_idx - start_idx
    if width < 1:
        stop_idx = min(size, stop_idx + 1)
    return start_idx, stop_idx

def validate_rep_rt(rts, rt_diff, nominal_rts, half_width, data_for_reps, rt_diff_thresh):
    '''
        Check that the replicate rt's are not too far from an expected time
        rts: the retention time list
        rt_diff: the difference between the rt and some good rt
        nominal_rt: the location where we think the peak should be
        half_width: the search space for a maximum if we adjust the rt
        rt_diff_thresh: if the rt diffs > this, the peak is replaced
        return the adjusted rts, and a list that has the replicate name of any
            replaced peaks
    '''
    # get the prospective rt's for each peak
    adjusted_rts = []
    made_rt_adjustment = []
    for rt,diff,nominal_rt,(rep,rep_data) in zip(rts, rt_diff, nominal_rts, data_for_reps.items()):
        if np.isnan(rt):
            adjusted_rts.append(nominal_rt)
        # this peak is different from the median rt,
        # set it to the most likely peak
        elif diff > rt_diff_thresh and np.abs(rt - nominal_rt) > rt_diff_thresh:
            # use the time with the maximum point inside the integration window
            raw_data = rep_data['raw_data']
            tic = np.sum(rep_data['matrix'], axis=0)
            times = raw_data[0].Times
            start_idx,stop_idx = np.searchsorted(times, [nominal_rt - half_width, nominal_rt + half_width])
            start_idx,stop_idx = check_index_width(start_idx, stop_idx, len(times)) 
            max_idx = np.argmax(tic[start_idx:stop_idx]) 
            new_rt = times[start_idx:stop_idx][max_idx]
            adjusted_rts.append(new_rt)            
            made_rt_adjustment.append(f'{rep} : RT {rt} to {new_rt}')
        else:
            adjusted_rts.append(rt)
            
    return adjusted_rts, made_rt_adjustment

def calculate_trans_areas(data_for_reps, adjusted_rts, half_width):
    '''
        Compute a matrix of normalized transitions areas.  The area
        is summed over a region around a retention time.  The areas
        are normalized to the summed area for all transitions
        adjusted_rts: peak retention times
        half_width: integration window
        return a matrix of areas where rows are transitions and columns are replicates
    '''
    first_key = next(iter(data_for_reps))
    num_rows,num_cols = len(data_for_reps[first_key]['raw_data']),len(data_for_reps)
    
    # get the transition ratios for each window
    normalized_trans_area = np.empty((num_rows, num_cols))
    for ridx,(rep,rep_data) in enumerate(data_for_reps.items()):
        # using the adjusted rt for the rt location
        adjusted_rt = adjusted_rts[ridx]
        matrix = rep_data['matrix']
        total_area = 1e-6
        trans_areas = np.empty(len(rep_data['raw_data']))
        for tidx,trans in enumerate(rep_data['raw_data']):            
            start_idx,stop_idx = np.searchsorted(trans.Times, [adjusted_rt - half_width, adjusted_rt + half_width])
            start_idx,stop_idx = check_index_width(start_idx, stop_idx, len(trans.Times))
            times,intensities = trans.Times[start_idx:stop_idx], trans.Intensities[start_idx:stop_idx]
            area = np.trapz(intensities,times)
            total_area += area
            trans_areas[tidx] = area
        # normalized transition areas
        trans_areas /= total_area
        normalized_trans_area[:, ridx] = trans_areas
    return normalized_trans_area

def compare_trans_areas(normalized_trans_area, nominal_trans_areas, trans_area_thresh):
    '''
        Compare the a set of transition areas (a spectrum really) to some nominal 
        set of areas
        normalized_trans_area: matrix of transitions x reps of normalized areas
        nominal_trans_areas: vector of normalized areas
        trans_area_thresh: if the normalized error is > this, store the replicate name
        return a table that has the replicate key for any replicate that doesn't match well
    '''
    # compute how well each replicate matches the median
    norm_nom = max(1e-6, np.linalg.norm(nominal_trans_areas))
    trans_errors,large_trans_error = [],[]
    for cidx in range(normalized_trans_area.shape[1]):
        row = normalized_trans_area[:, cidx]
        diff = np.subtract(row, nominal_trans_areas)
        norm_diff = np.linalg.norm(diff)
        error = norm_diff / norm_nom
        trans_errors.append(error)
        if error > trans_area_thresh:
            large_trans_error.append(True)
        else:
            large_trans_error.append(False)
    return large_trans_error, trans_errors

def get_rts(data_for_reps):
    rts = []
    for rep,rep_data in data_for_reps.items():
        raw_data = rep_data['raw_data']
        rt = np.median([x.RT for x in raw_data])
        rts.append(rt)
    return np.array(rts)

def analyze_high_conc_data(data, rt_diff_thresh, trans_area_thresh, half_width_multiplier):
    '''
        Try to ensure that each of the high concentration peaks has a similar RT and
            spectrum, by looking at the median RT and median normalized transition area.
        If the RT is odd, then we'll adjust it, and if the the spectrum is odd, then
            at the moment we log this in the return object, where it can be reported
        data : dictionary of replicates.  Each replicate has values
            'matrix': matrix, 'raw_data': raw_data, 'tic': np.sum(matrix, axis=0)
        rt_diff_thresh : if the rt is different than the median rt by > this amount,
            we'll shift the rt to a region closer to the middle
        trans_area_thresh : if the error between transition areas is > this amount,
            the replicate likely has the wrong peak
        return a HighConcInfo tuple
    '''
    # get the median integration width and rt
    med_width,rts,mid_dist = [],[],[]
    for rep,rep_data in data.items():
        raw_data = rep_data['raw_data']
        # use the min and max transition bounds
        min_start = np.min([x.StartRT for x in raw_data])
        max_stop = np.max([x.StopRT for x in raw_data])
        med_width.append(max_stop - min_start)
        rt = np.median([x.RT for x in raw_data])
        rts.append(rt)
        # the distance of the rt from the middle
        # peaks in the middle are more probable
        times = raw_data[0].Times
        mid_time = times[len(times)//2]
        mid_dist.append(np.abs(rt - mid_time))
        
    med_half_width = max(MIN_RT_WIDTH, np.median(med_width)) * half_width_multiplier
    med_rt = np.median(rts)
    
    # the "best" peak is likely the one in the middle
    _, sorted_rt = pmr.sort_two_lists(mid_dist, rts)
    middle_rt = sorted_rt[0]
    
    # compute how well each replicate rt matches the median
    rt_diff = np.abs(rts - med_rt)   
    
    # get the prospective rt's for each peak
    adjusted_rts,made_rt_adjustment = validate_rep_rt(
        rts, rt_diff, [med_rt]*len(rt_diff), med_half_width, data, rt_diff_thresh)

    # get the transition ratios for each window
    normalized_trans_area = calculate_trans_areas(data, adjusted_rts, med_half_width)
    
    # compute the median normalized transition area
    med_trans_areas = [np.median(normalized_trans_area[ridx, :]) for ridx in range(normalized_trans_area.shape[0])]

    # compute how well each replicate matches the median        
    is_large_trans_error,_ = compare_trans_areas(
        normalized_trans_area, med_trans_areas, trans_area_thresh)
    
    large_trans_error = [rep for rep,is_error in zip(data, is_large_trans_error) if is_error]    
           
    return HighConcInfo(
        adjusted_rts, 
        med_half_width, 
        normalized_trans_area, 
        med_trans_areas, 
        made_rt_adjustment, 
        large_trans_error)
       
SMALL_TIME_BOUND = 0.01
def move_boundary_off_edges(times, left, right):
    left_limit,right_limit = (times[0] + SMALL_TIME_BOUND, times[-1] - SMALL_TIME_BOUND)
    left = max(left, left_limit)
    right = min(right, right_limit)
    return left, right

def apply_high_conc_to_lower_concs(
    data_by_conc, high_conc_info, highest_conc, rt_diff_thresh, trans_area_thresh, apply_high_directly_to_low):
    '''
        Use the higher concentration information to adjust the lower concentration retention times
        return a list of Boundary objects
    '''
    boundaries = []
    for conc, reps_for_conc in data_by_conc.items():
        if conc == highest_conc or apply_high_directly_to_low:
            for rt,rep_data in zip(high_conc_info.adjusted_rts, reps_for_conc.values()):
                raw_data = rep_data['raw_data']
                left,right = move_boundary_off_edges(raw_data[0].Times,
                    rt - high_conc_info.med_half_width,
                    rt + high_conc_info.med_half_width)
                
                boundaries.append(Boundary(
                    raw_data[0].FileName,
                    raw_data[0].Compound,
                    left, 
                    right
                ))               
        else:
            rts = get_rts(reps_for_conc)        
            # compute how well each replicate rt matches high concentration rt
            rt_diff = np.abs(rts - high_conc_info.adjusted_rts)   

            # get the new rt's for each peak
            adjusted_rts,_ = validate_rep_rt(
                rts, 
                rt_diff, 
                high_conc_info.adjusted_rts, 
                high_conc_info.med_half_width, 
                reps_for_conc, 
                rt_diff_thresh)  

            # calculate the trans areas
            normalized_trans_area = calculate_trans_areas(reps_for_conc, adjusted_rts, high_conc_info.med_half_width)
            # check how they compare to the high concentration medians
            is_large_trans_error,_ = compare_trans_areas(
                normalized_trans_area, high_conc_info.med_trans_areas, trans_area_thresh)
            # only accept the new rt if the data look similar to the high conc data
            for idx,(is_trans_error, high_conc_rt) in enumerate(zip(is_large_trans_error, high_conc_info.adjusted_rts)):
                if is_trans_error:
                    adjusted_rts[idx] = high_conc_rt            
            
            for rt,rep_data in zip(adjusted_rts, reps_for_conc.values()):
                raw_data = rep_data['raw_data']
                left,right = move_boundary_off_edges(raw_data[0].Times,
                    rt - high_conc_info.med_half_width,
                    rt + high_conc_info.med_half_width)
                
                boundaries.append(Boundary(
                    raw_data[0].FileName,
                    raw_data[0].Compound,
                    left,
                    right
                ))    
    return boundaries
        
def save_boundaries(file_name, boundaries):
    with open(file_name, 'w') as f:
        f.write('File Name,Peptide Modified Sequence,Min Start Time,Max End Time\n')
        for b in boundaries:
            f.write(f'{b.FileName},{b.Compound},{b.Start},{b.Stop}\n')
            
def remove_brackets(pep):
    pep = re.sub('\[[\d\a\.\-\+]+]', '', pep)
    pep = re.sub('/\d+','', pep)
    return pep
    
def get_expected_num_replicates(data_by_conc):
    '''
        Sometimes I had to throw out a high conc sample because
        of running out of sample in the vial.  So the rt should come
        from the next highest conc replicate.  Check what is the expected
        number of replicates
    '''
    num_replicates = []
    for data in data_by_conc.values():
        num_replicates.append(len(data))
    return np.median(num_replicates)

def get_high_conc_data(data_by_conc, concentrations):
    '''
        In some case the high concentration data may only have say
        3 replicates but all the other concentrations have 4.  We 
        want the set of data used for the high concentration rt estimate
        to include the next most concentrated data for the 3 replicate case.
    '''
    expected_num_reps = get_expected_num_replicates(data_by_conc)
    # the nominal set of data
    highest_conc = concentrations[-1]
    highest_conc_data = data_by_conc[highest_conc]
    
    size_high_conc = len(highest_conc_data)
    # check if the data don't include all the reps that we need
    if size_high_conc < expected_num_reps:
        next_conc = concentrations[-2]
        next_conc_data = data_by_conc[next_conc]
        key_base = next(iter(next_conc_data))
        key_base = re.match('([\w\d\_]+_)\d+', key_base).group(1)

        start_idx, stop_idx = int(size_high_conc), int(expected_num_reps)
        for idx in range(start_idx, stop_idx):
            key = f'{key_base}{idx}'
            highest_conc_data[key] = next_conc_data[key]
    return highest_conc_data        
    
def validate_peaks(data_by_pep, half_width_multiplier, okay_pep_set={}, apply_high_directly_to_low=False):
    '''
        Apply the analysis to all the peptides in the set
        okay_pep_set: any pep in this set doesn't have its
        name reported in the print below
    '''
    validated_boundaries = []
    bad_count = 0
    for _,(pep,data_by_conc) in enumerate(data_by_pep.items()):
        concentrations = sorted([x for x in data_by_conc.keys()])
        highest_conc = concentrations[-1]        
        highest_conc_data = get_high_conc_data(data_by_conc, concentrations)
        # first analyze the high concentration data and adjust those retention times
        high_conc_info = analyze_high_conc_data(
            highest_conc_data, RT_DIFF_THRESH, TRANS_AREA_THRESH, half_width_multiplier)
        bare_name = remove_brackets(pep)
        if high_conc_info.large_trans_error and not bare_name in okay_pep_set:
            print(f'{bad_count}: {remove_brackets(pep)} Large Transition Error: {high_conc_info.large_trans_error}')
            bad_count +=1
#         if high_conc_info.made_rt_adjustment:
#             print(f'    {pep} RT Adjustment made to Reps: {high_conc_info.made_rt_adjustment}')
            
        # apply the high conc data to the lower concentrations
        boundaries = apply_high_conc_to_lower_concs(
            data_by_conc, high_conc_info, highest_conc, RT_DIFF_THRESH, TRANS_AREA_THRESH, apply_high_directly_to_low)
        
        validated_boundaries.extend(boundaries)
        
    return validated_boundaries

def analyze_and_save_new_boundaries(
    input_file_name, 
    output_file_name, okay_pep_set={}, 
    half_width_multiplier=EXTRA_WIDTH_MULTIPLIER, 
    apply_high_directly_to_low=False):
    '''
        Compose the entire pipeline:
            load
            make new boundaries
            save
    '''
    print(f'Loading Raw Data for {input_file_name}')
    # load raw data
    chrom_data = load_file(input_file_name)
    print('Determine Peak Boundaries')
    # determine set of boundaries
    boundaries = validate_peaks(chrom_data, half_width_multiplier, okay_pep_set, apply_high_directly_to_low)
    # save output file
    print(f'Save {output_file_name}')
    save_boundaries(output_file_name, boundaries)