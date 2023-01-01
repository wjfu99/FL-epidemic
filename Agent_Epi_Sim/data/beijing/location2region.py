import numpy as np
import shapefile
from shapely import geometry, contains_xy
from tqdm import tqdm
from multiprocessing import Pool
def region_map(region):
    poi_locs = np.load('./processed_data/poi_loc.npy', allow_pickle=True).item()
    outline = region[1]
    rid = region[0]
    locs = []
    for pid, coord in poi_locs.items():
        if contains_xy(outline, *coord):
            locs.append(pid)
    return locs
    return None

regions = shapefile.Reader('./raw_data/beijing-WGS/ST_R_CN_WGS.shp', encoding='latin1')
# shapeRecs = r.shapeRecords()
# regions = regions.iterShapeRecords()
new_regions = []
for region in tqdm(regions.iterShapeRecords(), total=len(regions)):
    record = region.record
    outline = geometry.shape(region.shape)
    new_regions.append([record['QH_CODE'], outline])
loc2region = {}
pool = Pool(processes=1)
# for region in tqdm(regions.iterShapeRecords(), total=len(regions)):
region2loc = tqdm(pool.imap(region_map, new_regions), total=6633)
# pool.map(region_map, new_regions)
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