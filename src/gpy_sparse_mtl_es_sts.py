import numpy as np
#import pylab as pb
import GPy
import sys
import re
import time
def main(args):
    (train_file, test_file, output_file, num_tasks, rank, n) = args

    (X_train, y_train, id_train) = load_file(train_file)
    test_t = 8
    print 'test task:',test_t
    (X_test, y_test, id_test) = load_file(test_file, test_t) #change for es task
    
    
    dim = len(X_train[0])
    start = time.time()
    num_tasks = int(num_tasks)
    rank = int(rank)
    (W, kappa) = buildB(num_tasks, rank)
    #print W
    kern_coreg = GPy.kern.rbf(input_dim=(dim-1))**GPy.kern.coregionalize(output_dim=num_tasks, W=W, rank=rank)
    #print 'W', kern_coreg.parts[0].k2.W
    #print 'B',  kern_coreg.parts[0].k2.B
    #B_orig = kern_coreg.parts[0].k2.W
    #kappa_orig = kern_coreg.parts[0].k2.kappa

    
    m = GPy.models.SparseGPRegression(X_train, y_train, kern_coreg, num_inducing=int(n)) #all dataset

    m.optimize('bfgs')
    #print m.constrained_indices

    #print 'W-op', kern_coreg.parts[0].k2.W
    #print 'B-op', kern_coreg.parts[0].k2.B
    end = time.time()
    print 'training time', end - start
    start = time.time()
    (Yp, Vp, up95, lo95) = m.predict(X_test)
    end = time.time()
    print 'test time', end - start
    #print Yp
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



def buildB(num_tasks, rank):
    #B_ind = np.identity(num_tasks, rank)
    #B_pooled = np.ones((num_tasks, num_tasks))
    #B_combo = 1 + alfa * np.identity(num_tasks)
    W = np.ones((num_tasks, rank))
    kappa = []
    return (W, kappa)

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print "Usage:./gpy_mtl_es_sts.py <train-file> <test-file> <prediction-file> <num-tasks> <rank> <num-inducing>"
        sys.exit(0)
    else:
        main(sys.argv[1:])
