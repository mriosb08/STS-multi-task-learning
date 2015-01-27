#!/usr/bin/python

import sys
import GPy
import numpy as np
import time

def main(args):
    (train_file, test_file, output_file) = args
    (X_train, y_train) = load_file(train_file)
    (X_test, y_test) = load_file(test_file)
    #print X_test, y_test
    #print X_train, y_train
    dim = len(X_train[0])
    print dim
    #print X_train[0]
    #print y_train
    #print X_train.shape
    #print y_train.shape
    start = time.time()
    
    t_distribution = GPy.likelihoods.noise_model_constructors.student_t(deg_free=5, sigma2=2)
    stu_t_likelihood = GPy.likelihoods.Laplace(y_train.copy(), t_distribution)
    kern = GPy.kern.rbf(input_dim=dim) + GPy.kern.Matern32(input_dim=dim)

    m = GPy.models.GPRegression(X_train, y_train, kernel=kern, likelihood=stu_t_likelihood)
    m.constrain_positive('t_noise')
    m.optimize('bfgs')
    end = time.time()
    print 'training time', end - start
    start = time.time()
    (Yp, Vp, up95, lo95) = m.predict(X_test)
    end = time.time()
    print 'test time', end - start
    #print Yp
    print m
    print m.log_likelihood()
    #with open(output_file,'w') as out:
    #for yp, y in zip(Yp, y_test):
        #print yp,'\t',y
        #print >>out, yp.
        
    np.savetxt(output_file, Yp, fmt='%.3f', newline='\n')
    return

def load_file(feature_file):
    with open(feature_file) as f:
        X = []
        y = []
        for line in f:
            line = line.strip()
            #print line
            (y_temp, x_temp) = line.split("\t")
            y.append([float(y_temp)])
            X.append([float(x) for x in x_temp.split("|||")])
        y = np.asarray(y)
        X = np.asarray(X)
    return (X, y)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage:./gpy_sts <train-file> <test-file> <prediction-file>'
        print 'Note: this model uses a non-gaussian likelihood student-t'
    else:
        main(sys.argv[1:])

