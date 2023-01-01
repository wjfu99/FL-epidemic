import numpy as np
import shapefile
from shapely import geometry, contains_xy
from tqdm import tqdm
from multiprocessing import Pool

poi_locs = np.load('./processed_data/poi_loc.npy', allow_pickle=True).item()
regions = shapefile.Reader('./raw_data/beijing-WGS/ST_R_CN_WGS.shp', encoding='latin1')
# shapeRecs = r.shapeRecords()
loc2region = {}


def region_map(poi_locs):
    pid = poi_locs[0]
    coord = poi_locs[1]
    if contains_xy(outline, *coord):
        return pid
    else:
        return None

for region in tqdm(regions.iterShapeRecords(), total=len(regions)):
    outline = geometry.shape(region.shape)
    record = region.record
    pool = Pool(processes=230)
    for pid, coord in poi_locs.items():
        pool.apply_async(region_map, args=((pid, coord)))
    pool.close()
    pool.join()


################ For test the unique id of the shp file ###########
# a = []
# b = []
# for region in tqdm(regions.iterShapeRecords(), total=len(regions)):
#     outline = geometry.shape(region.shape)
#     record = region.record
#     a.append(record['UserID'])
#     b.append(record['QH_CODE'])