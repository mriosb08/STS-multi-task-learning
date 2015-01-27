import numpy as np
#import pylab as pb
import GPy
import sys
import re
import time
import pickle
def main(args):
    (train_file, output_file, num_tasks, rank) = args
    (X_train, y_train, id_train) = load_file(train_file)
    dim = len(X_train[0])
    start = time.time()
    num_tasks = int(num_tasks)
    rank = int(rank)
    kern_coreg = GPy.kern.rbf(input_dim=(dim-1))**GPy.kern.coregionalize(output_dim=num_tasks, rank=rank)
    m = GPy.models.GPRegression(X_train, y_train, kern_coreg) #all dataset
    m.optimize('bfgs')
    end = time.time()
    print 'training time', end - start
    pickle.dump(m, open(output_file, "wb")) 
    return

def load_file(feature_file, id_t=None):
    with open(feature_file) as f:
        X = []
        y = []
        id = []
        data = {}
        for line in f:
            line = line.strip()
            #print line
            p = re.compile(r'\s+')
            cols = p.split(line)
            #print cols
            (id_task, file_name, num_instance) = cols[0].split("|||")
            score = cols[1]
            #features = list(id_task) + cols[2:]
            if id_t:
                id_task = id_t
            else:
                id_task = int(id_task) -1
            features = cols[2:]
            features.append(id_task)
            y.append([float(score)])
            X.append([float(x) for x in features])
            id.append((int(id_task), file_name))
        y = np.asarray(y)
        X = np.asarray(X)
    return (X, y, id)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage:./gpy_mtl_sts_train.py <train-file> <model-file> <num-task> <rank>"
        sys.exit(0)
    else:
        main(sys.argv[1:])
