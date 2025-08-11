import numpy as np

# Hein: this code is really overcommented and not layed out super nicely

def bin_epochs(data,edges,split_data=False,return_idx=False):
    """
    Bin data into epochs defined by edges. Edges can be a list of jagged arrays (i.e. each array has different length) or a 1D or 2D numpy array.
    2D numpy arrays and lists of jagged arrays are interpreted as multiple discontinuous windows, where 1D numpy arrays are interpreted as a single continuous window.

    Parameters
    ----------
    data : array_like
        Data to be binned.
    edges : array_like
        Bin edges.

    Returns
    -------
    bin_counts : ndarray(dtype=object) or ndarray(dtype=int)
        Array of counts in each bin. If jagged arrays, returns array of objects where each element is an array of counts in each bin.
    data_bin : ndarray(dtype=object) (if split_data=True)
        Array of data points split into each epoch.
    """
    # figure out the shape of edges for reshaping output
    if (type(edges) is list) or (type(edges) is np.ndarray and edges.dtype != float): # jagged arrays
        nw = len(edges) # number of windows
        nb = np.array([len(e) for e in edges]) # length of each jagged array
        wbreak = (np.cumsum(nb)-1)[:-1] # indices of window breaks, excluding last index which gets dropped in binning function
        nb = nb - 1 # subtract 1 to get number of bins in each window
        edges = np.hstack(edges) # flatten jagged array
    elif type(edges) is np.ndarray:
        if edges.ndim == 1: # single continuous window
            nb = edges.shape[0] - 1 # number of bins
            nw = 1 # number of windows
        elif edges.ndim == 2: # multiple discontinuous windows
            nw,nb = edges.shape
            nb -= 1  # subtract 1 to get number of bins in each window
            wbreak = np.ravel_multi_index((np.arange(nw),np.repeat(nb,nw)),edges.shape)[:-1] # indices of window breaks, excluding last index which gets dropped in binning function
        else:
            raise ValueError('edges ndarray must be 1D or 2D')
    else:
        raise ValueError('edges must be list or numpy array')
    
    # bin data
    if np.any(np.diff(edges.ravel()) < 0):
        raise NotImplementedError('Overlapping bin edges not yet supported')
    else:
        output = bin_monotonic(data,edges.ravel(),split_data=split_data,return_idx=return_idx)

    if split_data is False:
        output = [output] # put in list so it works with for loop below

    # reshape output if multiple windows
    if nw > 1:
        # cut window breaks
        b = np.ones_like(output[0],bool)
        b[wbreak] = False 
        output = [out[b] for out in output] #bin_counts[b]
        if type(nb) is np.ndarray: # jagged arrays
            wsplit = np.flatnonzero(np.diff(np.hstack([np.repeat(i,n) for i,n in enumerate(nb)])))+1
            output = [np.array(np.split(out,wsplit),dtype=object) for out in output]
        else: 
            output = [np.reshape(out,(nw,nb)).squeeze() for out in output]

    if split_data:
        return output[0],output[1] # bin counts in 0, binned data in 1
    else:
        return output[0]


def bin_monotonic(data,edges,split_data=False,return_idx=False):
    """
    Bin timing data based on monotonically increasing bins.

    Parameters
    ----------
    data : array_like
        Data to be binned.
    edges : array_like
        Monotonically increasing bin edges.
    split_data : bool, optional
        If True, return data split into bins. Default is False.
    return_idx : bool, optional
        If True, return indices of data instead of data. Default is False.

    Returns
    -------
    bin_counts : ndarray(dtype=int)
        Array of counts in each bin.
    data_bin : ndarray(dtype=object) (if split_data=True)
        Array of length len(edges)-1, where each element is array of time within each bin. If no time stamps in bin, element is empty.

    """
    binned = np.digitize(data,edges) - 1 # subtract 1 to make 0-based
    mask = (binned >= 0) & (binned < edges.size-1) # mask for data that falls within bins
    bin_counts = np.bincount(binned[mask],minlength=edges.size-1) # bin counts as number of data points in each bin

    if split_data:
        b = np.unique(binned[mask]) # non-empty bins
        data_bin = np.empty(edges.size-1,dtype=object) # initialize array of objects
        if len(b)>1: # if more than 1 bin
            if return_idx:
                data_idx = np.arange(len(data)) # indices of data
                data_bin[b] = np.split(data_idx[mask],np.flatnonzero(np.diff(binned[mask])) + 1) # split masked data on bin boundaries
            else:
                data_bin[b] = np.split(data[mask],np.flatnonzero(np.diff(binned[mask])) + 1) # split masked data on bin boundaries
        else: # if only 1 bin with data
            if return_idx: 
                data_idx = np.arange(len(data))
                data_bin[b] = [data_idx[mask]]
            else:
                data_bin[b] = [data[mask]]

        del binned,mask,b
        return bin_counts,data_bin
    else:
        del binned,mask
        return bin_counts


# Hein: why would bins overlap? Trying to think of the sorting that would make this necessary?
def bin_overlap(data,edges,split_data=True,return_idx=False):
    """
    Bin timing data with overlapping edges. Slower than bin_monotonic.
    Sorts edges to be monotonically increasing before binning, then reverts back to original order.
    Counts within bins between non-increasing edges are set to 0, and split data is None.

    Parameters
    ----------
    data : array_like
        Data to be binned.
    epoch : array_like
        Non-monotonically increasing epoch boundaries.
    return_idx : bool, optional
        If True, return indices of data instead of data. Default is False.

    Returns
    -------
    data_bin : ndarray(dtype=object)
        Array of length len(epoch), where each element is array of time within each bin. If no time stamps in bin, element is empty.
    bin_counts : ndarray(dtype=int)
        Array of counts in each bin.
    """
    sort_inds = edges.argsort() # sort inds for monotonic increasing order
    unsort_inds = np.argsort(sort_inds) # unsort inds for original order
    epoch_sort = edges[sort_inds] # sorted epochs
    binned = np.digitize(data,epoch_sort) - 1 # subtract 1 to make 0-based
    mask = (binned >= 0) & (binned < epoch_sort.size-1) # mask for data that falls within epoch boundaries
    bin_counts_sort = np.bincount(binned[mask],minlength=edges.size-1) # bin counts of sorted edges

    # transform monotonic increasing bins to original bins
    bin_counts = np.zeros_like(bin_counts_sort) # initialize output bin counts
    b_no = np.where(np.diff(unsort_inds) == 1) # bins that have no overlap / encomapass only one sorted bin
    bin_counts[b_no] = bin_counts_sort[unsort_inds[b_no]] # assign non-overlapping bins to original index

    # initial split of data if requested
    if split_data:
        b = np.unique(binned[mask]) # nonempty bins
        data_bin_sort = np.empty(edges.size-1,dtype=object) # initialize array of objects
        if return_idx:
            data_idx = np.arange(len(data)) # indices of data
            data_bin_sort[b] = np.array(np.split(data_idx[mask],np.flatnonzero(np.diff(binned[mask])) + 1),object).squeeze() # initial split  
        else:
            data_bin_sort[b] = np.array(np.split(data[mask],np.flatnonzero(np.diff(binned[mask])) + 1),object).squeeze() # initial split
        data_bin = np.empty(edges.size-1,dtype=object) # initialize array of objects
        data_bin[b_no] = data_bin_sort[unsort_inds[b_no]] # assign non-overlapping bins to original index

    # handle overlapping bins
    for b_ov in np.where(np.diff(unsort_inds) > 1)[0]: # bins that have overlap / encompass multiple sorted bins
        o = np.arange(unsort_inds[b_ov],unsort_inds[b_ov+1]) # encompassed bins 
        bin_counts[b_ov] = bin_counts_sort[o].sum() # sum encompassed bins into original index

        if split_data:
            d = np.hstack(data_bin_sort[o]) # stack data from monotonic bins
            data_bin[b_ov] = d[d != None] # remove empty elements
    
    if split_data:
        return bin_counts,data_bin
    else:
        return bin_counts
