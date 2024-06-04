import os
import numpy as np

EXTENSION_TO_DELIMITER_MAP = {
    '.txt': '\t',
    '.csv': ',',
}

def get_file_delimiter(file_name):
    '''
        determine which delimiter to use based on the extension
        return the corresponding value from EXTENSION_TO_DELIMITER_MAP
    '''
    ext = os.path.splitext(file_name)[1]

    if not ext in EXTENSION_TO_DELIMITER_MAP:
        raise Exception("Couldn't find extention {0} in map {1}".format(ext, EXTENSION_TO_DELIMITER_MAP))

    return EXTENSION_TO_DELIMITER_MAP[ext]

def homogenize_column_name(string):
    return string.strip().replace(' ', '').replace('_','').upper()

def map_required_columns(headers, required_column_names, skip_require=False):
    '''
        Create a map of required header names to their actual indices
        headers : list of string headers that were in the first line
        required_column_names : list of headers that should be in the header list
        raise an exception if any required header is not found
        return a list of indices that correspond to each required header value
    '''
    # make a dict of the required names
    actual_headers = {homogenize_column_name(name): index for index,name in enumerate(headers)}
    required_indices,orig_indices = [],[]
    for idx,required_header in enumerate(required_column_names):
        stripped_req_header = homogenize_column_name(required_header)
        if stripped_req_header in actual_headers:
            required_indices.append(actual_headers[stripped_req_header])
            orig_indices.append(idx)
        if stripped_req_header not in actual_headers and not skip_require:
            raise Exception("Required header \'{0}\' was not found in actual headers {1}".format(required_header, headers))
    return required_indices, orig_indices

def load_file_by_headers(file_name, required_column_names_and_converters, delimiter=None, optional_column_names_and_converters=None):
    '''
        Load a file and return data for columns specified by their corresponding headers.
        The headers can be specified in any case and spaces or underscores are ignored,
        allowing to pass for example protein_name and match to Protein Name, or ProteinName, etc.
        file_name : path of file to open
        required_column_names_and_converters : 
            list of tuples, the first ones being the strings for columns that should be present in the header,
            the second member of the tuple being the functions for how to load each column
        ex [
            ('name', str), ('output_value', float)
        ]
        return a list of dict for each row in the file, where each key in the dict is the required column name,
        and the value is the value from that row.
    '''
    if not isinstance(required_column_names_and_converters, list):
        raise Exception('A list of column names must be given')
    
    # extract the individual column names and functions
    required_column_names = [x[0] for x in required_column_names_and_converters]
    required_column_converters = [x[1] for x in required_column_names_and_converters]
    if optional_column_names_and_converters:
        optional_column_names = [x[0] for x in optional_column_names_and_converters]
        optional_column_converters = [x[1] for x in optional_column_names_and_converters]

    with open(file_name, 'r', encoding='utf-8-sig') as f:
        # get the delimiter for this file type
        if not delimiter:
            delimiter = get_file_delimiter(file_name)
        # get the header line and split it
        headers = [homogenize_column_name(x) for x in f.readline().split(delimiter)]
        # get the map of columns of which we want to get their index
        req_header_indices,_ = map_required_columns(headers, required_column_names)
        if optional_column_names_and_converters:
            opt_header_indices,orig_indices = map_required_columns(headers, optional_column_names, skip_require=True)
            for orig_idx,head_idx in zip(orig_indices, opt_header_indices):
                required_column_converters.append(optional_column_converters[orig_idx])
                req_header_indices.append(head_idx)
                required_column_names.append(optional_column_names[orig_idx])
        
        # create the map of column index to converter function
        converters = {index: func for index,func in zip(req_header_indices, required_column_converters)}
        # load the file 
        data = np.genfromtxt(file_name, delimiter=delimiter, skip_header=1, dtype=None, encoding=None, converters=converters, usecols=req_header_indices)
        
        output_items = []
        num_cols = len(required_column_names)
        if num_cols == 1:
            # if there's only one column, then we have to parse the output a little differently
            key = required_column_names[0]
            for value in data:
                output_items.append({key: value})
        else:
            for row in data:
                item = {}
                for name,value in zip(required_column_names, row):
                    item[name] = value
                output_items.append(item)
    return output_items