import numpy as np
import shapefile
from shapely import geometry, contains_xy
from tqdm import tqdm
from multiprocessing import Pool
def region_map(region):
    poi_locs = np.load('./processed_data/poi_loc_wgs84.npy', allow_pickle=True).item()
    outline = region[1]
    rid = region[0]
    locs = []
    for pid, coord in poi_locs.items():
        if contains_xy(outline, *coord):
            locs.append(pid)
    return {rid: locs}

regions = shapefile.Reader('./raw_data/beijing-WGS/ST_R_CN_WGS.shp', encoding='latin1')
# shapeRecs = r.shapeRecords()
# regions = regions.iterShapeRecords()
new_regions = []
for region in tqdm(regions.iterShapeRecords(), total=len(regions)):
    record = region.record
    outline = geometry.shape(region.shape)
    new_regions.append([record['QH_CODE'], outline])
loc2region = {}
pool = Pool(processes=200)
# for region in tqdm(regions.iterShapeRecords(), total=len(regions)):
region2loc_l = list(tqdm(pool.imap(region_map, new_regions), total=len(new_regions))) # list here is very important.
# pool.map(region_map, new_regions)
pool.close()
pool.join()
region2loc = {}
loc2region = {}
for region in region2loc_l:
    region2loc.update(region)
    # for rid, locs in region:
    rid = list(region.keys())[0]
    locs = list(region.values())[0]
    for loc in locs:
        loc2region[loc] = rid
np.save("./processed_data/region2poi_loc(wgs84).npy", region2loc)
np.save("./processed_data/poi_loc2region(wgs84).npy", loc2region)




################ For test the unique id of the shp file ###########
# a = []
# b = []
# for region in tqdm(regions.iterShapeRecords(), total=len(regions)):
#     outline = geometry.shape(region.shape)
#     record = region.record
#     a.append(record['UserID'])
#     b.append(record['QH_CODE'])