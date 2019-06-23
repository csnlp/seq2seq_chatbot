#!/usr/local/bin/python3

import sys
import pickle

print(sys.version)
testpkl = pickle.loads(open("dataset_cornell.pkl", "rb").read())

pickle.dump(testpkl, open("dataset_cornell_p2.pkl","wb"), protocol=2)
