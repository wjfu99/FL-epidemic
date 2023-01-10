from simulator.simulator import Engine
import numpy as np
import argparse, datetime
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="infected parameter beta", default=0.405)
parser.add_argument("--mu", type=float, help="infected parameter mu", default=0.071)
parser.add_argument("--eps", type=float, help="infected parameter mu", default=1/14)
parser.add_argument("--time_granularity", type=float, help="time granularity(hours)", default=0.5)
parser.add_argument("--init_ratio", type=float, help="initialized infected case ratio", default=0.001)
parser.add_argument("--init_mode", type=str, help="initialized infected case ratio", default="global")
# parser.add_argument("--beta", type=float, help="infected parameter beta", default=0.1)

args = parser.parse_args()
args.time_granularity = datetime.timedelta(hours=args.time_granularity)

traj = np.load("./data/beijing/processed_data/traj_mat(filled).npy")
eng = Engine(**vars(args))
eng.load_traj(traj)
eng.init_epi(args.init_mode, args.init_ratio)
# eng.get_state_count
# eng.next()
eng.next(step_num=480*4)
print(eng.get_state_count)

joblib.dump(eng.get_usr_states, "label")
joblib.dump(eng, 'eng(large)')
