import pickle
import os
cwd = os.getcwd()
print(cwd)
print(os.listdir())
stats = pickle.load(open("static/lorl_state_stats.pkl", "rb"))
print(stats)
