import numpy as np

def ben_find_peaks(y,minsep):
    """ Finds peaks in data subject to minimum separation requirement.

    Args:
        y         : Data for which to find local maxima.
        minsep    : Minimum number of points between maxima.
    
    Returns:
        peakinds  : Indices of local maxima.
    """
    candinds = np.where(np.logical_and((y[1:-1]>y[:-2]),(y[1:-1]>y[2:])))[0]+1

    peakinds = np.sort(pull_off_top(candinds,y[candinds],minsep))
    
    return peakinds


def pull_off_top(inds,maxvals,minsep):
    """ Recursive extraction of maxima.
    """
    print("=====")
    topind = inds[np.argmax(maxvals)]
    print(topind)
    print(inds)
    fi = np.where(np.abs(inds-topind)>minsep)[0]
    
    print(fi)
    
    if len(fi)>0:
        farinds = inds[fi]
        farmaxvals = maxvals[fi]
        return np.append(pull_off_top(farinds,farmaxvals,minsep),topind)
    else:
        return np.array([topind])
