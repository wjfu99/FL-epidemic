1.poi_process.py 处理poi数据，处理好的poi数据保存为字典形式。
2.pre_process.py 导入原始轨迹数据，将原始不规整时间点数据处理为规整的以半小时为时间粒度的数据。
3.start_and_end.py 导入2处理好的轨迹数据，提取每个user的轨迹开始时间和结束时间。
4.user_data.py 通过由3处理得到的用户开始时间和结束时间，计算出频率最高的开始时间和结束时间，
以此确定时间范围，在这个时间范围内进行轨迹筛选，删除轨迹点出现次数小于总时间点数10%的用户轨迹。
5.region_process.py 划定区域(限制区域的经纬度)，如果用户轨迹包含区域之外的poi则删除该条轨迹，保存筛选后的轨迹。
6.traj_process.py 对user_id和poi_id重新命名，user_id映射到0——用户数量，poi_id 映射为0-POI数量。并对缺失的轨迹点用-1填充。
7.data_generater.py 导入6生成的轨迹，运行疫情模型，目前疫情模型基于GLEaM,每10天进行一次核酸检测并隔离,检测好像是随机抽0。1的人进行检测。
程序运行顺序按照1-2-3-4-5-6-7即可。
final_result中是之前处理好的疫情数据。ori_data.npy是处理好的轨迹数据。

数据

poi_loc 对应poi的坐标信息

总人数应该是15279
区域数应该是11459

修改代码，保留区域的坐标，然后设置区域聚合代码


region_process.py 划定区域(限制区域的经纬度（需要修改）)，如果用户轨迹包含区域之外的poi则删除该条轨迹，保存筛选后的轨迹。 输出final user，用户id：地区id（都是未映射的id）
input: userd.npy, poi_loc.npy
output: final_user.npy, bj_loc.npy

poi_cluster.py 完成区域聚合（修改经纬度）
input: final_user.npy, poi_loc.npy
output: region_id.npy

traj_process.py 对user_id和poi_id重新命名，user_id映射到0——用户数量，poi_id 映射为0-POI数量。并对缺失的轨迹点用-1填充。
input: final_user.npy, region_id.npy
output: ori_data.npy这个应该是聚合之后的聚合前是ori_data_old.npy, user_id.npy

data_generater.py 导入6生成的轨迹，运行疫情模型，目前疫情模型基于GLEaM,每10天进行一次核酸检测并隔离,检测好像是随机抽0。1的人进行检测。
input: ori_data.npy
output: sample_result.npy per_data.npy

construct_h.py
construct_g.py
get_fts.py


