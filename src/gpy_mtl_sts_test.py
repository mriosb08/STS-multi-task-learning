import numpy as np
#import pylab as pb
import GPy
import sys
import re
import time
import pickle
def main(args):
    (model_file, test_file, output_file, id_task) = args
    (X_test, y_test, id_test) = load_file(test_file, id_task)
    m = pickle.load(open(model_file, "rb"))
    start = time.time()
    print 'model loaded...'
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


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage:./gpy_mtl_sts_test.py <model-file> <test-file> <prediction-file> <id-task>"
        sys.exit(0)
    else:
        main(sys.argv[1:])
