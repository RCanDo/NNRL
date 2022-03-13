"""
title: Launching TensorBoard
link: https://www.tensorflow.org/guide/summaries_and_tensorboard#launching_tensorboard
date: 2018-08-25 Sat 11:38:13
author: kasprark
"""

"""
To run TensorBoard, use the following command (best in Anaconda Prompt)

> tensorboard --logdir ./ --host=127.0.0.1
or
> tensorboard --logdir [your logdir] --host=127.0.0.1

alternatively

> tensorboard --logdir=path/to/log-directory

> python -m tensorboard.main --logdir ....

where `logdir` points to the directory where the `FileWriter` _serialized_ its data.
If this `logdir` directory contains subdirectories which contain _serialized_ data from separate runs,
then TensorBoard will visualize the data from all of those runs.

Once TensorBoard is running, navigate your web browser to
    localhost:6006
or
    127.0.0.1:6006
to view the TensorBoard.

When looking at TensorBoard, you will see the navigation tabs in the top right corner.
Each tab represents a set of serialized data that can be visualized.
"""