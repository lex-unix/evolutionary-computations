import evocomp
from evocomp import visualization

offspring_sizes = [60, 100, 140]
study_results = visualization.study(
    param_values=offspring_sizes,
    objective=evocomp.Easom(),
    setup=lambda offspring_size: evocomp.EvoStrategy(lmda=offspring_size, epochs=20),
)

# Displaying results to console
visualization.display(study_results)
"""
Parameter Study Results:
Parameter     Fitness        Solution               Epochs     Time(s)
-------------------------------------------------------------------------
60            -0.999915      3.145021, 3.148316     21         0.22
100           -0.999965      3.143419, 3.137123     21         0.38
140           -0.999999      3.141046, 3.14229      21         0.52
"""

# Using custom display config
visualization.display(
    study_results,
    visualization.DisplayConfig(
        param_name='Lambda',
        algorithm='Evolutionary Strategy',
        objective='Easom',
        float_precision=3,
        time_precision=3,
    ),
)
"""
Algorithm: Evolutionary Strategy
Function:  Easom

Parameter Study Results:
Lambda     Fitness        Solution         Epochs     Time(s)
----------------------------------------------------------------
60         -1.000         3.145, 3.148     21         0.221
100        -1.000         3.143, 3.137     21         0.380
140        -1.000         3.141, 3.142     21         0.517
"""

# Plot results using matplotlib and save
visualization.plot_histories(study_results, 'lambda', 'Easom', 'Evolutionary Strategy', save=False)

# Export as csv
visualization.export_results(study_results, 'study_results', 'csv')

# Export as excel
visualization.export_results(study_results, 'study_results', 'xlsx')
