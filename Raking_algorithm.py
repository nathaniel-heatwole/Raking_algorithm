# RAKING_ALGORITHM.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Uses a raking algorithm to adjust (benchmark) the output from a logistic regression equation
# Training data: synthetic data for 20 students on whether they passed an exam and the number of hours that they studied (benchmark value is user-specified)

import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from colorama import Fore, Style
from matplotlib.backends.backend_pdf import PdfPages

time0 = time.time()
ver = ''  # version (empty or integer)

topic = 'Raking algorithm'
topic_underscore = topic.replace(' ','_')

#--------------#
#  PARAMETERS  #
#--------------#

# NOTE: runtime increases considerably as the absolute difference between 'target' value (below) and the initial total (9.9) increases

target = 13             # target count (i.e., what we want the total count to be) (sum of predicted probabilities) (range 0-20, exclusive)
method = 'unlimited'    # 'fixed' (run for specified number of iterations), or 'unlimited' (run until convergence is achieved)
total_iterations = 100  # total iterations ('fixed' method only)
precision = 10          # numerical granularity to use

decimal_places = 2

#-------------#
#  FUNCTIONS  #
#-------------#

def raking_algorithm():
    # adjust columns (applies column-specific scalar multipliers)
    mult_col0 = round(non_target/np.sum(exam['prob 0 new']), precision)
    mult_col1 = round(target/np.sum(exam['prob 1 new']), precision)  # multiplier of one indicates the target value is hit exactly
    exam['prob 0 new'] *= mult_col0
    exam['prob 1 new'] *= mult_col1
    # adjust rows (applies row-specific scalar multipliers)
    exam['prob total'] = round(exam['prob 0 new'] + exam['prob 1 new'], precision)
    exam['mult row'] = round(1/exam['prob total'], precision)  # multiplier of one indicates the rowwise probabilities (p0, p1) sum to one (as they should)    
    exam['prob 0 new'] *= exam['mult row']
    exam['prob 1 new'] *= exam['mult row']
    return exam, mult_col0, mult_col1

def format_plot():
    plt.xlabel('Hours studied', fontsize=axis_labels_size)
    plt.ylabel('Pass probability', fontsize=axis_labels_size)
    plt.xlim(-x_margin, max_hours_ceil + x_margin)
    plt.ylim(-y_margin, y_margin + 1)
    plt.xticks(np.arange(0, max_hours_ceil + 1), fontsize=axis_ticks_size)
    plt.yticks(np.arange(0, p_ticks + 1, p_ticks), fontsize=axis_ticks_size)
    plt.grid(True, alpha=0.5, zorder=0)
    plt.show(True)

def console_print(title, df):
    print(Fore.GREEN + '\033[1m' + '\n' + title + Style.RESET_ALL)
    print(df)
    
def txt_export(title, df, f):
    print(title, file=f)
    print(df, file=f)

#-----------------#
#  TRAINING DATA  #
#-----------------#

# input data (from https://en.wikipedia.org/wiki/Logistic_regression#Example)
passed_exam = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
hours_studied = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5]

# aggregate into df
exam = pd.DataFrame()
exam['hours'] = hours_studied
exam['passed'] = passed_exam

# metrics
total_obs = len(exam)
non_target = total_obs - target
target_prob = round(target/total_obs, decimal_places)

# initial predicted probabilities (logistic regression output)
# exam['x'] = -4.1 + (1.5 * exam['hours'])  # logistic regression equation (determined elsewhere - source at top)
# exam['prob 1'] = np.exp(exam['x']) / (1 + np.exp(exam['x']))  # passed exam (logit)
# exam['prob 0'] = 1 - exam['prob 1']  # did not pass exam (1 - logit)
# exam['pred class'] = round(exam['prob 1'])  # predicted group

# run regression
x = pd.DataFrame(exam['hours'])
y = pd.DataFrame(exam['passed'])
logit_eqn = sm.Logit(y, sm.add_constant(x))  # 'add_constant' instructs the regression to fit an intercept
logit_fit = logit_eqn.fit()
slope = round(logit_fit.params.iloc[1], decimal_places)
intercept = round(logit_fit.params.iloc[0], decimal_places)

# make predictions (generates the initial probabilities)
exam['prob 1'] = logit_fit.predict()  # pass probability
exam['prob 0'] = 1 - exam['prob 1']  # non-pass probability (1 - logit)
exam['pred class'] = round(exam['prob 1'])  # predicted group membership

#-------------#
#  ALGORITHM  #
#-------------#

# initalize
exam['prob 0 new'] = exam['prob 0']
exam['prob 1 new'] = exam['prob 1']

# iteratively apply rowwise and columnwise adjustments (multipliers)
if method == 'unlimited':  # continue until convergence is achieved
    mult_col0 = 0
    mult_col1 = 0
    exam['mult row'] = 0
    while (mult_col0 != 1) or (mult_col1 != 1) or (np.sum(exam['mult row'] == 1) != total_obs):
        exam, mult_col0, mult_col1 = raking_algorithm()
elif method == 'fixed':  # continue for fixed number of iterations
    for r in range(total_iterations):
        exam, mult_col0, mult_col1 = raking_algorithm()

exam['pred class new'] = round(exam['prob 1 new'])  # predicted group

# change in target value
exam['prob 1 delta'] = exam['prob 1 new'] - exam['prob 1']
exam['prob 1 delta abs'] = abs(exam['prob 1 delta'])

#-------------------#
#  SUMMARY RESULTS  #
#-------------------#

fmt = '.' + str(decimal_places) + 'f'

# metrics
orig_total = round(np.sum(exam['prob 1']), decimal_places)
shifted_total = round(np.sum(exam['prob 1 new']), decimal_places)
orig_avg_prob = round(np.mean(exam['prob 1']), decimal_places)
shifted_avg_prob = round(np.mean(exam['prob 1 new']), decimal_places)

# dataframe
summary = pd.DataFrame()
summary.index = ['total observations', 'total predicted [sum(p)]', 'avg probability']
summary['original'] = [str(total_obs), format(orig_total, fmt), format(orig_avg_prob, fmt)]
summary['shifted'] = [str(total_obs), format(shifted_total, fmt), format(shifted_avg_prob, fmt)]
summary['target'] = [str(total_obs), format(target, fmt), format(target_prob, fmt)]

#---------#
#  PLOTS  #
#---------#

# plot parameters
title_size = 11
axis_labels_size = 9
axis_ticks_size = 7
legend_size = 8
point_size = 8
line_width = 1.25
x_plot = exam['hours']
max_hours_ceil = np.ceil(max(x_plot))
x_margin = 0.25
y_margin = 0.05
p_ticks = 0.2

# solid area plot parameters
if shifted_total > orig_total:
    direction = 'added'
    fill_color = 'blue'
    y_component = 1
    y_lower = exam['prob 1']
    y_upper = exam['prob 1 new']
else:
    direction = 'removed'
    fill_color = 'red'
    y_component = -1
    y_lower = exam['prob 1 new']
    y_upper = exam['prob 1']

# labels
orig_label = 'original logit (n = ' + str(format(orig_total, fmt)) + ')'
shifted_label = 'shifted curve (n = ' + str(format(shifted_total, fmt)) + ')'
training_label = 'training data (n = ' + str(total_obs) + ')'

# original and benchmarked curves
fig1 = plt.figure()
plt.title(topic + ' - exam performance', fontsize=title_size, fontweight='bold')
plt.plot(x_plot, exam['prob 1'], color='blue', label=orig_label, linewidth=line_width, zorder=5)
plt.plot(x_plot, exam['prob 1 new'], color='red', label=shifted_label, linewidth=line_width, zorder=10)
plt.scatter(x_plot, exam['passed'], color='black', marker='*', s=point_size, label=training_label, zorder=15)
plt.legend(loc='upper right', bbox_to_anchor=(0.97, 0.24), fontsize=legend_size, facecolor='white', framealpha=1)
format_plot()
del orig_label, shifted_label

exam.sort_values(by=['hours'], inplace=True)

# probability region (shaded area is the probability that was added or removed by the shift)
fig2, ax = plt.subplots()
plt.title(topic + ' - region of probability ' + direction, fontsize=title_size, fontweight='bold')
ax.fill_between(x_plot, y_upper, y_lower, color=fill_color, alpha=0.3, zorder=5)
# vectors (arrows) showing direction and magnitude of shift
for h in np.arange(1, 5, 1):
    df = exam.loc[exam['hours'] >= h]
    df.reset_index(inplace=True)
    arrow_x_start = df['hours'][0]  # zero keeps only first row (in the event of ties)
    arrow_y_start = df['prob 1'][0]
    arrow_length = df['prob 1 delta abs'][0]
    if (arrow_length > 0.05):  # include only if the vector is sufficiently long
        plt.quiver(arrow_x_start, arrow_y_start, 0, y_component, color='black', scale=1/arrow_length, scale_units='y', width=0.004, zorder=10)
format_plot()
del df

#----------#
#  EXPORT  #
#----------#

# export summary (console, txt)
with open(topic_underscore + '_summary' + ver + '.txt', 'w') as f:
    title = topic.upper() + ' SUMMARY'
    df = summary
    console_print(title, df)
    txt_export(title, df, f)
del f, title, df

# export plots (pdf)
pdf = PdfPages(topic_underscore + '_plots' + ver + '.pdf')
for f in [fig1, fig2]:
    pdf.savefig(f)
pdf.close()
del pdf, f

###

# runtime
runtime_sec = round(time.time() - time0, 2)
if runtime_sec < 60:
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec / 60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0

