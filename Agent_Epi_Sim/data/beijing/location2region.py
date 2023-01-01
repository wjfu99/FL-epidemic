import numpy as np
import shapefile
from shapely import geometry, contains_xy
from tqdm import tqdm

poi_locs = np.load('./processed_data/poi_loc.npy', allow_pickle=True).item()
regions = shapefile.Reader('./raw_data/beijing-WGS/ST_R_CN_WGS.shp', encoding='latin1')
# shapeRecs = r.shapeRecords()
loc2region = {}
for region in tqdm(regions.iterShapeRecords(), total=len(regions)):
    outline = geometry.shape(region.shape)
    record = region.record
    for pid, coord in poi_locs.items():
        if contains_xy(outline, *coord):
            loc2region[pid] = record['QH_CODE']


################ For test the unique id of the shp file ###########
# a = []
# b = []
# for region in tqdm(regions.iterShapeRecords(), total=len(regions)):
#     outline = geometry.shape(region.shape)
#     record = region.record
#     a.append(record['UserID'])
#     b.append(record['QH_CODE'])