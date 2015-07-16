# Frequency Analyzer

    A tool for analyzing, grouping, and visualizing frequencies found in 
    raw accelerometer data (i.e. from a smartwatch).

For a full list of options see the help menu `--help`.

## What does it do?

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
