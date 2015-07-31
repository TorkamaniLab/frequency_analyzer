# Frequency Analyzer

    A tool for analyzing, grouping, and visualizing frequencies found in 
    raw accelerometer data (i.e. from a smartwatch).

For a full list of options see the help menu `--help`.

![A sample graph](/sample_analysis.png)

## How do I use it?

**Typical Usage:**

    python frequency_analyzer.py --graph -i raw__data.csv -o results.csv

This usage pattern will take `raw_data.csv` and do the analysis on it, save the results to `results.csv` and graph the output. It's pretty simple.

**Crazy Usage:**

    python frequency_analyzer.py -gver -i raw_data.csv -o results.csv -k 0.6 -a z:45,y:10,z:35 -l 3 -g

This usage, which I think uses every option, will take `raw_data.csv`, do the analysis, save the results to `results.csv` and graph the result (just like above). It will also save all intermediate data sets to the `cwd `, detail it's progress with verbose output, ignore gravity in the calculations, transform the resulting data by the angles `z:45, y:10, z:35`, compute 3 levels of the discrete wave transform, and consider all frequencies over 60% of the max as significant.

## What does it do?

Imagine the kind of accelerometer data captured by your phone or smartwatch as you move. This tool takes that raw data and presents it as a series of frequency profiles and pulls out what it deems as 'significant frequencies'. This is useful if you're looking to see what frequencies make up the signal you're trying to analyze. It also calculates some handy values for each frequency grouping to give you even more information about what makes up your signal.

## What kind of data does it use?

frequency_analyzer accepts 3d acclerometer input data in csv format. The data 
must have a time unit as the first column and then be in x, y, z order immediately 
following.

According to the `--format` option:

    Time,X,Y,Z
    0,213,-435,54
    20,332,-3245,4
    ...

Times may be either absolute (i.e. from the Unix Epoch) or local (beginning from zero).
Frequency analyzer expects the acceleration values to be in milliG's. See `--help` for
more information. The header may also be ommited if the `--no-header` option is provided.

Frequency Analyzer has the ability to save it's intermediate data via the `--save` option.
When this option is provided, Frequency Analyzer will write intermediate csv and graph 
images to the current working directory.

If you wish to see the results of Frequency Analyzer graphically and interactivly, supplying
the `--graph` option will display an interactive figure of the data. This figure includes
the graph of the frequency domain of the data as well as the position waves in all 3 dimensions.

If the acceleration data was collected when the acclerometer was not in line with the inertial 
frame of the earth, and the transformation angles are known, then supplying the `--angle` option,
and providing the appropriate transformation angles and sequence will allow Frequency Analyzer to
remove the earth's gravity from the calculations. If it is not known, then supply the `--no-gravity`
option and Frequency Analyzer will not attempt to remove the earth's gravity from the reference 
frame measurements or transform the data to the earth's inertial frame.
