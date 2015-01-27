import numpy as np
#import pylab as pb
import GPy
import sys
import re
import time
def main(args):
    (train_file, test_file, output_file, num_tasks, test_task, n) = args
    (X_train, y_train, id_train) = load_file(train_file)
    (X_test, y_test, id_test) = load_file(test_file, test_task)
    dim = len(X_train[0])
    start = time.time()
    kern_coreg = GPy.kern.RBF(input_dim=(dim-1))**GPy.kern.Coregionalize(1, output_dim=int(num_tasks), rank=1)
	
    m = GPy.models.SparseGPRegression(X_train, y_train, kernel=kern_coreg, num_inducing=int(n)) #all dataset

    m.optimize('bfgs')
    end = time.time()
    print 'training time', end - start
    start = time.time()
    (Yp, Vp, up95, lo95) = m.predict(X_test)
    end = time.time()
    print 'test time', end - start
    print m
    print m.log_likelihood()
    np.savetxt(output_file, Yp, fmt='%.3f', newline='\n')
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
    if len(sys.argv) != 7:
        print "Usage:./gpy_sparse_mtl_sts.py <train-file> <test-file> <prediction-file> <num-tasks> <test-task> <induce-points>"
        sys.exit(0)
    else:
        main(sys.argv[1:])
