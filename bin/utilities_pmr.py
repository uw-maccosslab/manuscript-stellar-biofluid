import numpy as np
import os
from enum import Enum
import math
import re
from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
from cycler import cycler
from datetime import datetime
import random

def get_thresholds_from_max(vector, threshold_frac):
    '''
        Given a vector of data and a threshold, find the maximum
        index, and then find the indices of the values that
        are max_value * threshold on either side of the peak
    '''
    max_index = np.argmax(vector)
    max_value = vector[max_index]
    threshold_y = threshold_frac * max_value
    
    # find the lower bound
    size_vec = len(vector)
    low_idx = max_index
    while low_idx >= 0 and vector[low_idx] > threshold_y:
        low_idx -= 1
    low_idx = max(low_idx, 0)
    
    # find the upper bound
    high_idx = max_index
    while high_idx < size_vec and vector[high_idx] > threshold_y:
        high_idx += 1
    high_idx = min(high_idx, size_vec-1)
    
    return {'max_index': max_index, 'max_value': max_value, 'low_idx': low_idx, 'high_idx': high_idx}
    
class sample_col(Enum):
    Sample = 0
    Location = 1
    
def setup_environment(cycle_type=None):
    '''
        Call the normal things I call to get the matplotlib settings I like.
    '''
	
    if cycle_type=="monochrome":
        # set mono chrome cycler
        plt.rcParams['axes.prop_cycle'] = (cycler('color', ['k']) * cycler('marker', ['', '.']) * cycler('linestyle', ['-', '--', ':', '-.']))
    elif cycle_type=="grayscale":
        plt.rcParams['axes.prop_cycle'] = (cycler('color', ['k', '0.75', '0.50', '0.25']))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    set_matplotlib_formats('jpeg')
    SMALL_SIZE = 5
    MEDIUM_SIZE = 6
    BIGGER_SIZE = 7

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 5
    plt.rcParams['hatch.linewidth'] = 0.1
    params = {
        'figure.figsize': (3, 2),
        # controls default text sizes
        'font.size': SMALL_SIZE,
        # fontsize of the axes title
        'axes.titlesize': SMALL_SIZE,
        # fontsize of the x and y labels
        'axes.labelsize': SMALL_SIZE,
        'xtick.labelsize': SMALL_SIZE,
        'ytick.labelsize': SMALL_SIZE,
        'legend.fontsize': SMALL_SIZE,
        'figure.titlesize': BIGGER_SIZE
    }
    plt.rcParams.update(params)
    return colors

def create_graph(figsize=(12,10)):
    '''
        create a new matplotlib figure and axes
    '''
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
    
def create_graph_axes(figsize=(12,10)):
    fig, ax = create_graph(figsize)
    return ax
    
def binary_search(value_to_find, array):
    '''
        do a binary search in array for value_to_find
        The array must be sorted from low to high
        return the index of the found value.
    '''
    size = len(array)
    if array is None or size == 0:
        return None

    low_idx, mid_idx, hi_idx = 0, 0, size - 1
    while low_idx <= hi_idx:
        mid_idx = int(low_idx + (hi_idx - low_idx) // 2)
        if array[mid_idx] == value_to_find:
            return mid_idx
        elif array[mid_idx] < value_to_find:
            low_idx = mid_idx + 1
        else:
            hi_idx = mid_idx - 1

    # the value actually isn't in the array.  determine
    # which index is closest to the value
    mid_diff = abs(array[mid_idx] - value_to_find)
    next_diff = 1e30 if hi_idx < 0 else abs(array[hi_idx] - value_to_find)
    prev_diff = 1e30 if low_idx >= size else abs(array[low_idx] -
                                                 value_to_find)
    if next_diff < mid_diff:
        return hi_idx
    elif prev_diff < mid_diff:
        return low_idx

    return mid_idx
    
def get_data_from_file(fileName, delimiter='\t', num_header_lines=None):
    '''
        open up a file of numbers and add each one to a line
    '''
    data, headers = [], []
    with open(fileName, 'r') as file:
        if num_header_lines:
            for i in range(num_header_lines):
                headers.append(file.readline())
        for line in file:
            data.append( [float(x) for x in line.strip().split(delimiter)] )
    return np.array(data), headers
    
    
def get_files_in_directory(directory, file_extension=None, string_in_name=None):
    '''
        get a list of all the files in this directory
    '''
    files = []
    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)
        if os.path.isfile(full_path):
            ext = os.path.splitext(f)[1]
            found = True if not string_in_name else re.search(string_in_name, f)
            if (not file_extension or ext == file_extension) and found:
                files.append(full_path)
    return files
    
    
def get_a_color(idx):
    idx = idx % 10
    return 'C{}'.format(idx)
    

def get_line_through_two_points(pt1, pt2):
    '''
        compute the slope and intercept for two points
        having format [x, y]
        return slope, intercept
    '''
    delta_x = pt2[0] - pt1[0]
    if delta_x == 0.0:
        return None
    delta_y = pt2[1] - pt1[1]
    slope = delta_y / delta_x
    intercept = pt1[1] - slope * pt1[0]
    return slope, intercept
    
    
class PlaneSnapShot(Enum):
    PosX = 0
    PosY = 1
    PosZ = 2
    Radius = 3
    VelX = 4
    VelY= 5
    VelZ = 6
    Speed = 7
    Status = 8
    Mass = 9
    CrossSection = 10
    Charge = 11
    Time = 12

class SnapShot(Enum):
    PosX = 0
    PosY = 1
    PosZ = 2
    Radius = 3
    VelX = 4
    VelY= 5
    VelZ = 6
    Speed = 7
    Status = 8
    Mass = 9
    CrossSection = 10
    
class IonStatus(Enum):
    Alive = 0
    Splat = 1
    Outside = 2
    PastPlane = 3
    FinishedMatrix = 4
    
class Histogram():
    def __init__(self, data, num_bins=128, min_value=None, max_value=None):
        '''
            Initialize a histogram by creating the xaxis and setting the
            y_bins to have 0 counts.  if the min and max values aren't
            specified, use the data min and maximum values
        '''
        if not min_value:
            min_value = min(data)
        if not max_value:
            max_value = max(data)
        self.step_size = (max_value - min_value) / num_bins
        if self.step_size == 0.0:
            raise Exception('histogram step size is 0.0')
        if num_bins == 0:
            raise Exception('histogram num bins is 0')
        
        self.half_stepsize = self.step_size / 2.0
        self.x_bins = [i * self.step_size + self.half_stepsize + min_value for i in range(num_bins)]
        self.y_bins = [0 for _ in self.x_bins]
        self.min_value = min_value
        self.max_value = max_value
        self.add_data(data)
    
    def add_data(self, data):
        '''
            Find the bin index for each datum and increment the appropriate counter
            param data : a 1D list of data
        '''
        for value in data:
            bin_idx = int((value - self.min_value + self.half_stepsize) // self.step_size)
            if bin_idx >= 0 and bin_idx < len(self.y_bins):
                self.y_bins[bin_idx] += 1
    
    def get_center_of_mass(self):
        '''
            compute the weighted average of the data,
            where the y's are the probabilities of each
            x value
        '''
        if len(self.x_bins) == 0:
            raise Exception('Histogram has length 0')
        sum_y = sum(self.y_bins)
        if sum_y == 0:
            raise Exception('Histogram sum of counts is 0')
        
        com = 0.0
        for idx, xbin in enumerate(self.x_bins):
            ybin = self.y_bins[idx]
            com += ybin * xbin
        return com / sum_y
        
def geom_spaced_series(minimum, maximum, steps):
    '''
        Create a geometrically spaced series
    '''
    const = math.exp(math.log(maximum / minimum) / steps)
    return [minimum * const**i for i in range(steps)]
    
def lin_spaced_series(minimum, maximum, steps_or_stepsize, mode='steps'):
    '''
        make a linear spaced series of values
        param minimum : smallest value, inclusive
        param maximum : laragest value, inclusive
        param steps : number of steps for computing the step size.  the returned array
            has size steps + 1 to span the range including the maximum
        return an array of scalars linear spaced from minimum to maximum
    '''
    if mode == 'steps':
        steps = steps_or_stepsize
        step_size = (maximum - minimum) / (float(steps))
        # add 1 step so that the series goes all the way to the last value
        values = [
            float(i) * float(step_size) + float(minimum)
            for i in range(int(steps + 1))
        ]
    else:
        step_size = steps_or_stepsize
        steps = int((maximum - minimum) / step_size)
        values = [
            float(i) * float(step_size) + float(minimum)
            for i in range(int(steps + 1))
        ]
    return values
    
    
def sort_two_lists( list1, list2 ):
    '''
        sort by a first list, and sort the second list
        with reference to the first list
    '''
    list1, list2 = (list(x) for x in zip(*sorted(zip(list1, list2), key=lambda pair: pair[0])))
    return list1, list2
    
def sort_two_lists_rev( list1, list2 ):
    '''
        sort by a first list, and sort the second list
        with reference to the first list
    '''
    list1, list2 = (list(x) for x in zip(*sorted(zip(list1, list2), reverse=True, key=lambda pair: pair[0])))
    return list1, list2
    
def golden_section_search(x_minimum,
                          x_maximum,
                          eval_function,
                          tolerance=0.01,
                          maximum_iterations=100):
    '''
        The 1D line search is an important tool for optimization.  We use it to search
        along a direction for the minimum of an objective function.  In our signal processing
        application, the objective function is the sum of squared differences between data and
        model.  But this function allows to pass in any object that has the method Calc_F that
        returns a scalar.  So what is the Golden Ratio?  This is the special number that the
        Greeks found exists in many places in nature, and is pleasing to the eye when things
        are made according to it.
        The ratio holds for two numbers a,b, a < b, when a / b = b / (a+b).  It is useful in
        line searches for the following reason:

        Let's say we have two starting points a_0 and b_0 that bracket a minimum.  We can
        subdivide the line that connects a_0 and b_0 by choosing two new points a_1 and b_1.
        if f(a_1) < f(b_1), then we choose the new endpoints to be a_0 and b_1, knowing that the
        minimum lies between these two.  If rho is the golden ratio, and we originally chose a_1
        and b_1 to be a_1 = a_0 + rho*(b_0-a_0) and b_1 = a_0 + (1-rho)*(b_0-a_0), then a_1 will be
        perfectly placed in the new segment a_0 a_1 b_1, such that we only have to pick one more
        spot between a_1 and b_1 to evaluate.  Thus we only have to evaluate F once on each
        iteration after the first one.

        The other nice thing we do here is convert a multidimension problem into a 1D problem,
        by way of the objective function value.  Thus we are finding the minimizer of the
        function Calc_F(Z(alpha)), where Z = x_minimum + d_k*alpha, Z,x_minimum,d_k in R^n, alpha
        in R^1. x_minimum = a_0, and x_minimum + d_k = x_maximum = b_0. So alpha is the scalar
        minimizer we are finding, that gives us the final vector minimizer
        x_opt = x_minimum + alpha*d_k.

        param x_minimum : lower bound of search, a type accepted by eval_function
        param x_maximum : upper bound of search, a type accepted by eval_function
        param eval_function : function with signature f(x) -> scalar
            in order to converge, f(x_minimum) > f(optimum) < f(x_maximum), that is,
            the optimum should be somewhere between x_minimum and x_maximum
        param tolerance : optimization stops when the end points of the search
            which gets normalized to [0, 1] are closer than this distance
        param maximum_iterations : maximum number of steps before quitting with
            a failure
        return None if no convergence, otherwise returns the x value at the optimum
    '''
    # the golden ratio
    rho = (3.0 - math.sqrt(5.0)) / 2.0
    one_minus_rho = 1 - rho
    # the four scalar points that define our search
    a_0, a_1, b_0, b_1 = None, None, None, None
    # and the function value points
    f_a1, f_b1 = None, None
    # d_k is the direction vector for the search,
    # starting at x_minimum, it goes to x_maximum
    d_k = x_maximum - x_minimum
    # the initial bounds of alpha,
    # the normalized x value
    a_0, b_0 = 0.0, 1.0

    def Z(alpha):
        '''
            Converts normalized x value
            to the original x units
            param alpha normalized x value
        '''
        return x_minimum + d_k * alpha

    # take the first step
    a_1, b_1 = rho * b_0, one_minus_rho * b_0
    f_a1, f_b1 = eval_function(Z(a_1)), eval_function(Z(b_1))

    # step until the tolerance is reached or we exceed the max iterations
    iteration = 0
    while abs(b_0 - a_0) > tolerance and iteration < maximum_iterations:
        # the optimum is on the left, so shift the search region to the LEFT
        if f_a1 < f_b1:
            b_0, b_1, f_b1 = b_1, a_1, f_a1
            # evaluate the new point
            a_1 = a_0 + rho * (b_0 - a_0)
            f_a1 = eval_function(Z(a_1))
        else:
            # the optimum is on the right, shift the search region to the RIGHT
            a_0, a_1, f_a1 = a_1, b_1, f_b1
            # evaluate the new point
            b_1 = a_0 + one_minus_rho * (b_0 - a_0)
            f_b1 = eval_function(Z(b_1))
        iteration += 1

    x_opt = None
    # take the minimum to be the lower point
    if f_a1 < f_b1:
        x_opt = Z(a_1)
    else:
        x_opt = Z(b_1)
    return x_opt
    
class LongStat():
    '''
        Convenient means of keeping a running average and standard deviation
    '''
    def __init__(self):
        self._count = 0
        self._x = 0
        self._x2 = 0
        
    def GetNumValuesInStat(self):
        return self._count

    def GetMean(self):
        return self._x
    
    def GetVar(self):
        return self._x2 - self._x*self._x
    
    def GetStd(self):
        return math.sqrt(self.GetVar())

    def __ComputeLongCoefficients(self, count):
        '''        
            Get the coefficients needed for computing the long statistics.
            These are just a few values that depend on the iteration number.
            param count, the iteration
            param invN, the inverse of the count value
            param oldFactor, (N-1)/N, the value that scales the old statistic
        '''
        fCount = float(count)
        invN = 1.0 / fCount
        oldFactor = (fCount - 1)*invN;
        return invN, oldFactor
    
    def Add(self, newValue):
        '''
            Computing the running E[x] and E[x^2], which
            are later combined to give the variance as
            E[(x - mu)^2] = E[x^2] - E[x]^2
        '''
        self._count+=1
        if self._count == 1:
            self._x = newValue;
            self._x2 = newValue*newValue
        else:
            invN, oldFactor = self.__ComputeLongCoefficients(self._count)
            self._x = oldFactor * self._x + invN * newValue;
            self._x2 = oldFactor * self._x2 + invN * newValue*newValue
            
            
def _sg_smooth(y_array, kernel):
    '''
        The savitsky golay kernel is used to smooth a set of data.
        https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
        A kernel is convolved with the data to remove higher frequency
        oscillations.  S-G uses a least-squares formulation to get the job done.
        It uses the fact that to solve the a polynomial least squares equation,
        you'd have A * x = b, where b are the observed data, x are the coefficients,
        and A is the special polynomial basis.  The matrix A is set up as follows :
        each column is a basis of increasing order from 0 to k.  For each observed point's
        x value, the column would be x^i, where i is the column index.  So if we had
        X = {1, 2, 3, 4}, the first rows of A would be

        1, 1, 1, 1
        1, 2, 4, 8
        1, 4, 16, 64

        The least squares solution is x = (At*A)^-1 * At * b
        If you make the matrix (At*A)^-1 * At, it will have size
        k x N, where N is the number of points in b.  The savitsky-golay coefficients
        for smoothing are the zeroth order, or first row of this matrix.  Incidentally,
        the second row can be used to estimate the first derivative of a set of data,
        third row the second derivative, etc.  Below is the code to find the coefficients
        in C++ using the eigen library.

        // make the J matrix, which is the polynomial basis matrix
        auto polyOrder = params.filterOrder;
        MatrixXf J(N, polyOrder + 1);

        for (auto order = 0; order <= polyOrder; ++order) {
            for (int z = -halfSize, row = 0; z <= halfSize; ++z, ++row) {
                J(row, order) = powf(z, order);
            }
        }

        MatrixXf Jt = J.transpose();
        MatrixXf C = (Jt*J).inverse() * Jt;
        _savitskyGolayCoefficients = C.row(0).transpose();

        param y_array : array of data to be smoothed
        param kernel : SG array that gets convolved with the data
    '''
    size_kernel, size_data = len(kernel), len(y_array)
    halfsize_kernel = math.floor(size_kernel // 2)
    # we need enough data to at least get one pass on it
    if size_data < size_kernel:
        return None

    smooth_y = []
    for i in range(size_data):
        result = 0.0
        # the start and stop indices of the data.  this may be
        # outside of the bounds of the data, but we'll just use the
        # actual data for those points
        start, stop = i - halfsize_kernel, i + halfsize_kernel + 1
        if start < 0 or stop >= size_data:
            result = y_array[i]
        else:
            # convolve the data with the kernel
            k = 0
            for j in range(start, stop):
                if j >= 0 and j < size_data:
                    result = result + kernel[k] * y_array[j]
                k += 1
        smooth_y.append(result)
    return smooth_y


_SG_7_3 = [
    -0.0952381,
    0.142857,
    0.285714,
    0.333333,
    0.285714,
    0.142857,
    -0.0952381,
]

def savitsky_golay_7pt_3order(y_array):
    '''
        7pt smoothing with 3rd order polynomial
    '''
    return _sg_smooth(y_array, _SG_7_3)


_SG_5_3 = [
    -0.0857143,
    0.342857,
    0.485714,
    0.342857,
    -0.0857143,
]


def savitsky_golay_5pt_3order(y_array):
    '''
    5pt smoothing with 3rd order polynomial
    '''
    return _sg_smooth(y_array, _SG_5_3)
    
    
def to_string(x):
    return x

def to_float(x):
    return float(x)

ext_to_delimiter = {
    '.txt': '\t',
    '.csv': ','
}

def get_file_delimiter(file_name):
    bare_filename, file_extension = os.path.splitext(file_name)
    if file_extension not in ext_to_delimiter:
        raise Exception('Extension {} not recognized'.format(file_extension))
    delimiter = ext_to_delimiter[file_extension] 
    return delimiter

def load_file_with_converters(enum_class, converters, file_name):
    '''
        File loading utility
        enum_class : a class that has an enum for each column in the file
        converters : dictionary by num, having converter functions like to_string, etc
        return a list of lists indexed by the enum index
   '''
    data = np.genfromtxt(file_name, delimiter=get_file_delimiter(file_name), 
                         skip_header=1, dtype=None, encoding=None,
                         converters=converters)
    items = []
    for row in data:
        item = {}
        for col in enum_class:
            item[col] = row[col.value]
        items.append(item)
    return items

def find_lower_bound(xaxis, yaxis, low_idx, threshold):
    '''
        find where a set of data drop below a threshold,
        going from right to left
    '''
    while low_idx >= 0 and yaxis[low_idx] > threshold:
        low_idx -= 1
    hi_idx = low_idx + 1
    if hi_idx >= len(xaxis):
        return False
        
    if xaxis[low_idx] == xaxis[hi_idx]:
        raise Exception('dX = 0.0')
        
    # look at the local slope between these points
    slope,intercept = get_line_through_two_points(
        (xaxis[low_idx], yaxis[low_idx]), 
        (xaxis[hi_idx], yaxis[hi_idx])
    )
    # and interpolate to the target value
    return (threshold - intercept)/slope, low_idx

def find_upper_bound(xaxis, yaxis, hi_idx, threshold):
    '''
        find where a set of data drop below a threshold,
        going from left to right
    '''
    size = len(xaxis)
    while hi_idx < size-1 and yaxis[hi_idx] > threshold:
        hi_idx += 1
    low_idx = hi_idx - 1
    if low_idx < 0:
        return False
        
    if xaxis[low_idx] == xaxis[hi_idx]:
        raise Exception('dX = 0.0')
        
    # look at the local slope between these points
    slope,intercept = get_line_through_two_points(
        (xaxis[low_idx], yaxis[low_idx]), 
        (xaxis[hi_idx], yaxis[hi_idx])
    )
    # and interpolate to the target value
    return (threshold - intercept)/slope, hi_idx

def rotate_point(x, y, angle_rad):
    cos,sin = np.cos(angle_rad),np.sin(angle_rad)
    return cos*x-sin*y,sin*x+cos*y

def draw_brace(ax, span, position, text, text_pos, brace_scale=1.0, beta_scale=300., rotate=False, rotate_text=False):
    '''
        all positions and sizes are in axes units
        span: size of the curl
        position: placement of the tip of the curl
        text: label to place somewhere
        text_pos: position for the label
        beta_scale: scaling for the curl, higher makes a smaller radius
        rotate: true rotates to place the curl vertically
        rotate_text: true rotates the text vertically        
    '''
    # get the total width to help scale the figure
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    resolution = int(span/xax_span*100)*2+1 # guaranteed uneven
    beta = beta_scale/xax_span # the higher this is, the smaller the radius
    # center the shape at (0, 0)
    x = np.linspace(-span/2., span/2., resolution)
    # calculate the shape
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    # put the tip of the curl at (0, 0)
    max_y = np.max(y)    
    min_y = np.min(y)
    y /= (max_y-min_y)
    y *= brace_scale
    y -= max_y
    # rotate the trace before shifting
    if rotate:
        x,y = rotate_point(x, y, np.pi/2)
    # shift to the user's spot   
    x += position[0]        
    y += position[1]
    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1, clip_on=False)
    # put the text
    ax.text(text_pos[0], text_pos[1], text, ha='center', va='bottom', rotation=90 if rotate_text else 0)