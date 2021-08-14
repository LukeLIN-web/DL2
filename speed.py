import numpy as np
import ast
import scipy.interpolate
import matplotlib.pyplot as plt


# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import MaxNLocator

def fit() -> dict:
    # fit a speed function for each model
    speed_funcs = dict()
    records = []
    with open("config_speed.txt", "r") as f:
        for line in f:
            records.append(ast.literal_eval(line.replace('\n', '')))
    speed_maps = dict()
    for record in records:
        model, sync_mode, tot_batch_size, num_ps, num_worker, speeds, ps_cpu_usages, worker_cpu_usages = record
        # acutally ,we just use three column from this.
        if model not in speed_maps:
            speed_maps[model] = []
        speed_maps[model].append((num_ps, num_worker, sum(speeds)))  # speeds is sample
    for model in speed_maps.keys():
        x = [];y = [];z = []
        for _num_ps, _num_worker, _speed in speed_maps[model]:
            x.append(_num_ps)
            y.append(_num_worker)
            z.append(_speed)
        interp = scipy.interpolate.Rbf(np.array(x), np.array(y), np.array(z), function='linear')
        speed_funcs[model] = interp  # store function
        '''		you could plot function with following codes		'''
        # xnew, ynew = np.mgrid[-1:1:100j, -1:1:100j]  # 输入输出都是二维
        # znew = interp(xnew,ynew)
        # ax = plt.subplot(111, projection='3d')
        # ax.plot_surface(xnew, ynew, znew)
        # ax.scatter(x, y, z, c='r', marker='^')
        # plt.show()
    return speed_funcs


speed_funcs = fit()
