import numpy as np
from coordTransform_utils import gcj02_to_wgs84
poi_locs = np.load('./processed_data/poi_loc.npy', allow_pickle=True).item()
poi_locs_wgs84 = {}
for pid, coord in poi_locs.items():
    wgs_coord = gcj02_to_wgs84(*coord)
    poi_locs_wgs84[pid] = wgs_coord
np.save('./processed_data/poi_loc_wgs84.npy', poi_locs_wgs84)
