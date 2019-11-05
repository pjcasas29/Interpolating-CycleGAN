from visdom import Visdom 
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--log_file', type=str, default='', help='log file')

opt = parser.parse_args()

viz = Visdom(port=8097)
viz.replay_log(opt.log_file)
