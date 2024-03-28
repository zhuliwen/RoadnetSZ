# -*- coding: utf-8 -*-

'''

1) interacting with SUMO, including
      retrive values, set lights
2) interacting with sumo_agent, including
      returning status, rewards, etc.

'''

import numpy as np
import math
import os
import json
import sys
import xml.etree.ElementTree as ET
from sys import platform
from cityflow_agent import Vehicles
import collections
import cityflow as engine


ccccc

'''
TRAFFIC_FILE = "anon_4_4_700_0.3_synthetic.json"
PATH_TO_CONF = "conf/4_4"
ROAD_NET_FILE = "./data/4_4/roadnet_4_4.json"
CITYFLOW_CONFIG_FILE = "./data/4_4/cityflow.config"
DIR = "data/4_4/"
ROADNETFILE = "roadnet_4_4.json"
'''


from agent import State
from keras import backend as K

###### Please Specify the location of your traci module


# --- my modification ---
NUM_OF_NEIGHBORS = 4

yeta = 0.15
tao = 2
constantC = 40.0
carWidth = 3.3
grid_width = 4
area_length, area_width = 600, 600
direction_lane_dict = {"NSG": [1, 0], "SNG": [1, 0], "EWG": [1, 0], "WEG": [1, 0],
                       "NWG": [0], "WSG": [0], "SEG": [0], "ENG": [0],
                       "NEG": [2], "WNG": [2], "SWG": [2], "ESG": [2]}

#min_phase_time = [30, 96, 74]
min_phase_time_7 = [10, 35]

# --- my modification ---
global_listLanes = collections.OrderedDict()
global_all_lanes_each_node = collections.OrderedDict()
entering_lanes = collections.OrderedDict()
coordinate_offset = collections.OrderedDict()

# --- my modification ---
def get_node_id_list():
    # sumo may not has started
    # return traci.trafficlight.getIDList()
    # it's a little slow, so we need to only calculate it once
    if not hasattr(get_node_id_list, 'node_id_list'):
        get_node_id_list.node_id_list = []
        file = ROAD_NET_FILE
        with open(file) as json_data:
            net = json.load(json_data)
            for node_id in net['intersections']:
                if not node_id['virtual']:
                    get_node_id_list.node_id_list.append(node_id['id'])
    return get_node_id_list.node_id_list


def reset(para_set):
    cityflow_config = {
        "interval": 1,
        "seed": 0,
        "laneChange": False,
        "dir": DIR,
        "roadnetFile": ROADNETFILE,
        "flowFile": TRAFFIC_FILE,
        "rlTrafficLight": True,
        "saveReplay": True,
        "roadnetLogFile": "frontend/web/roadnetLogFile.json",
        "replayLogFile": "frontend/web/replayLogFile.txt"
    }
    print("=========================")
    print(cityflow_config)

    print(os.getcwd())
    print(os.path.join(para_set.DIR, "cityflow.config"))
    # with open(os.path.join(para_set.DIR, "cityflow.config"), "w") as json_file:
    with open(CITYFLOW_CONFIG_FILE, "w") as json_file:
        json.dump(cityflow_config, json_file)  # json.dumps将一个Python数据结构转换为JSON
    eng = engine.Engine(CITYFLOW_CONFIG_FILE, thread_num=1)
    return eng


def find_neighbors():
    node_id_list = get_node_id_list()
    if not hasattr(find_neighbors, 'neighbers'):
        file = ROAD_NET_FILE
        with open(file) as json_data:
            net = json.load(json_data)

            find_neighbors.neighbors = collections.OrderedDict()
            edge_id_dict = collections.OrderedDict()
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['to'] = road['endIntersection']

            for node_id in node_id_list:  # eg. i = intersection_1_1
                find_neighbors.neighbors[node_id] = []
                find_neighbors.neighbors[node_id] = [node_id_list.index(node_id)]
                for j in range(4):
                    road_id = node_id.replace("intersection", "road") + "_" + str(j)
                    neighboring_node = edge_id_dict[road_id]['to']
                    # calculate the neighboring intersections
                    if neighboring_node in node_id_list:
                        find_neighbors.neighbors[node_id].append(node_id_list.index(neighboring_node))
                    else:
                        find_neighbors.neighbors[node_id].append(-1)
                if len(find_neighbors.neighbors[node_id]) < 1 + NUM_OF_NEIGHBORS:
                    find_neighbors.neighbors[node_id].extend(
                        [-1 for _ in range(1 + NUM_OF_NEIGHBORS - len(find_neighbors.neighbors[node_id]))])
                if len(find_neighbors.neighbors[node_id]) != 1 + NUM_OF_NEIGHBORS:
                    find_neighbors.neighbors[node_id] = find_neighbors.neighbors[node_id][0:1 + NUM_OF_NEIGHBORS]
    return find_neighbors.neighbors

def get_node_phases(node_id):
    # or: return traci.trafficlight.getCompleteRedYellowGreenDefinition()
    if not hasattr(get_node_phases, 'phase_dict'):
        get_node_phases.phase_dict = collections.OrderedDict()
    if node_id not in get_node_phases.phase_dict:
        tree = ET.parse(NET_FILE)
        root = tree.getroot()
        node = root.find("./tlLogic[@id='%s']" % node_id)
        phase = node.findall("./phase")
        get_node_phases.phase_dict[node_id] = [x.get('state') for x in node]
    return get_node_phases.phase_dict[node_id]
    # or traci.trafficlight.getCompleteRedYellowGreenDefinition(node_id)

def get_all_lanes_of_this_node(node_id):
    file = ROAD_NET_FILE
    with open(file) as json_data:
        net = json.load(json_data)
        for node in net['intersections']:
            if node['id'] == node_id:
                node_information = node
                break
        controlled_roads = node_information['roads']
        how_much_lanes = {}
        lanes = []
        for road in net['roads']:
            how_much_lanes[road['id']] = len(road['lanes'])
        for road in controlled_roads:
            for i in range(how_much_lanes[road]):
                lane_id = road + '_' + str(i)
                lanes.append(lane_id)
        return lanes

def get_controlled_lanes(node_id):
    # only used once
    a = int(node_id[-3])
    b = int(node_id[-1])
    list_approachs = ["W", "E", "N", "S"]
    lane_num = {'left': 1, 'right': 1, 'straight': 1}
    dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(a - 1, b)}
    dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(a + 1, b)})
    dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(a, b - 1)})
    dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(a, b + 1)})
    list_entering_lanes = []
    for approach in list_approachs:
        list_entering_lanes += [dic_entering_approach_to_edge[approach] + '_' + str(i) for i in
                                range(sum(list(lane_num.values())))]
    return list_entering_lanes
    # root = tree.getroot()
    # edge = root.findall("./edge[@to='%s']" % node_id)
    # return [y.get('id') for x in edge for y in x.findall("./lane")]

# now it is useless
def get_node_coordination(node_id):
    if not hasattr(get_node_coordination, 'coordination_dict'):
        get_node_coordination.coordination_dict = collections.OrderedDict()
    if node_id not in get_node_coordination.coordination_dict:
        tree = ET.parse(NET_FILE)
        root = tree.getroot()
        node = root.find("./junction[@id='%s'][@type='traffic_light']" % node_id)
        offset = root.find("./location")
        offset_x, offset_y = offset.get('netOffset').split(",")
        offset_x, offset_y = float(offset_x), float(offset_y)
        x, y = float(node.get('x')), float(node.get('y'))
        get_node_coordination.coordination_dict[node_id] = (x-offset_x, y-offset_y)
    return get_node_coordination.coordination_dict[node_id]

def get_incoming_node(node_id_list):
    # only used once
    incoming, outgoing, road_list = collections.OrderedDict(), collections.OrderedDict(), collections.OrderedDict()
    tree = ET.parse(NET_FILE)
    root = tree.getroot()
    edge = root.findall("./edge")
    for x in edge:
        src = x.get('from')
        dst = x.get('to')
        road = x.get('id')
        if '#' in road: # one segment
            road = road[:road.index('#')] # only record its main name
        if src != None and dst != None: # real edge
            if src not in outgoing:
                outgoing[src] = set()
            outgoing[src].add(dst)
            if dst not in incoming:
                incoming[dst] = set()
            incoming[dst].add(src)
            if road not in road_list:
                road_list[road] = []
                if road.startswith('-'):
                    if dst in node_id_list:
                        road_list[road].append(dst)
                else:
                    if src in node_id_list:
                        road_list[road].append(src)
            if road.startswith('-'):
                if src in node_id_list:
                    road_list[road].insert(0, src)
            else:
                if dst in node_id_list:
                    road_list[road].append(dst)
            # if it's a segment, the list will be like: src, dst, dst2, dst3, ...
            # segments will appear in order.
            # skip non-node
    for dst in node_id_list:
        if incoming. __contains__(dst) == False:
            incoming[dst] = set()
    for src in node_id_list:
        if outgoing. __contains__(src) == False:
            outgoing[src] = set()

    incoming_node = collections.OrderedDict()
    for dst in node_id_list:
        incoming_node[dst] = set()
        for src in incoming[dst]: # count those who are nodes and 1-hop incoming
            if src in node_id_list:
                incoming_node[dst].add(src)
    # special: if it's a multi-hop incoming, oon different segments of one road, via some non-nodes
    for road, route in road_list.items():
        for i in range(len(route)-1):
            if route[i] not in incoming_node[route[i+1]]:
                incoming_node[route[i+1]].add(route[i])
                print('%s and %s add connection via road: %s' % (route[i], route[i+1], road))
    # special: if it's a 2-hop incoming, via a non-node
    for dst in node_id_list:
        for src in node_id_list:
            if src != dst and src not in incoming_node[dst]: # no connection
                for via in outgoing[src].intersection(incoming[dst]):
                    if via not in node_id_list:
                        incoming_node[dst].add(src)
                        print('%s and %s add connection via non-node: %s' % (src, dst, via))
                        break

    return incoming_node

'''
input: phase "NSG_SNG" , four lane number, in the key of W,E,S,N
output: 
1.affected lane number: 4_0_0, 4_0_1, 3_0_0, 3_0_1
# 2.destination lane number, 0_3_0,0_3_1  

'''

# --- my modification ---
# record all the vehicles that have left the network
all_vehicles_enter_time_dict = collections.OrderedDict()
lane_vehicle_arrive_leave_time_dict = collections.OrderedDict()
for node_id_1 in get_node_id_list():
    lane_vehicle_arrive_leave_time_dict[node_id_1] = collections.OrderedDict()
# record all the vehicles that have left the each node area


def start_sumo(eng):
    # traci.start(sumo_cmd_str)
    # --- my modification ---
    for node_id in get_node_id_list():
        # TODO: use find_surrounding_lane_WESN & phase_affected_lane instead
        temp = get_controlled_lanes(node_id)
        temp1 = get_all_lanes_of_this_node(node_id)
        # x, y = get_node_coordination(node_id)
        global_listLanes[node_id] = temp
        global_all_lanes_each_node[node_id] = temp1
        entering_lanes[node_id] = temp
        # coordinate_offset[node_id] = [x, y]
        # --- my modification ---
    random_phase = []

    for node_id in get_node_id_list():
        random_num = 1
        random_phase.append(random_num)
        eng.set_tl_phase(node_id, random_num)
    eng.next_step()
    # for i in range(20):
    #    eng.next_step()
    return random_phase

def end_sumo(eng, episode, current_time, file_name_travel_time, episode_time):
    # --- my modification ---
    f_travel_time = open(file_name_travel_time, "a")
    f_average_travel_time = open("travel_time.txt", "a")
    print('%d vehicles have left the network.' % len(all_vehicles_enter_time_dict))
    if all_vehicles_enter_time_dict:
        average_travel_time = np.mean(list(all_vehicles_enter_time_dict.values()))
        print('Their average travel time: %f' % np.mean(list(all_vehicles_enter_time_dict.values())))
    else:
        average_travel_time = '999999999999999999999'
        print('Their average travel time: ', '999999999999999999999')
    # average_travel_time_from_cityflow = eng.get_average_travel_time()
    mmm = eng.get_average_travel_time()
    print('average travel time from cithflow api is %d' % mmm)
    memory_str = 'episode = %s\ttime = %d\t%d vehicles left\tepisode_time = %f' % (
        episode, current_time, len(all_vehicles_enter_time_dict), episode_time)

    travel_time_dict_from_lane = {}
    memory_str_cityflow = 'average_travel_time_cityflow = %f' % mmm

    f_travel_time.close()

    f_average_travel_time.write(memory_str + "\n")
    f_average_travel_time.write(memory_str_cityflow + "\n")
    f_average_travel_time.close()

    all_vehicles_enter_time_dict.clear()
    travel_time_dict_from_lane.clear()

def end_sumo_test(eng, episode, current_time, file_name_travel_time, episode_time):
    # --- my modification ---
    f_travel_time = open(file_name_travel_time, "a")
    f_average_travel_time = open("test_travel_time.txt", "a")
    print('%d vehicles have left the network.' % len(all_vehicles_enter_time_dict))
    if all_vehicles_enter_time_dict:
        average_travel_time = np.mean(list(all_vehicles_enter_time_dict.values()))
        print('Their average travel time: %f' % np.mean(list(all_vehicles_enter_time_dict.values())))

    else:
        average_travel_time = '999999999999999999999'
        print('Their average travel time: ', '999999999999999999999')
    mmm = eng.get_average_travel_time()
    memory_str = 'episode = %s\ttime = %d\t%d vehicles left\tepisode_time = %f' % (
        episode, current_time, len(all_vehicles_enter_time_dict),  episode_time)
    travel_time_dict_from_lane = {}
    memory_str_cityflow = 'average_travel_time_cityflow = %f' % mmm

    f_travel_time.close()

    f_average_travel_time.write(memory_str + "\n")
    f_average_travel_time.write(memory_str_cityflow + "\n")
    f_average_travel_time.close()

    all_vehicles_enter_time_dict.clear()
    travel_time_dict_from_lane.clear()

def get_current_time(eng):
    return eng.get_current_time()


def clear_local_travel_time():
    all_vehicles_location_enter_time_dict = collections.OrderedDict()
    all_vehicles_this_node_enter_time_dict = collections.OrderedDict()
    for node_id_1 in get_node_id_list():
        all_vehicles_location_enter_time_dict[node_id_1] = collections.OrderedDict()
        all_vehicles_this_node_enter_time_dict[node_id_1] = collections.OrderedDict()


# it looks useless
def phase_affected_lane(phase="NSG_SNG", 
                    four_lane_ids={'W': 'edge1-00', "E": "edge2-00", 'S': 'edge3-00', 'N': 'edge4-00'}):
    directions = phase.split('_')
    affected_lanes = []
    for direction in directions:
        for k, v in four_lane_ids.items():
            if v.strip() != '' and direction.startswith(k):
                for lane_no in direction_lane_dict[direction]:
                    affected_lanes.append("%s_%d" % (v, lane_no))
                    # affacted_lanes.append("%s_%d" % (v, 0))
    if affected_lanes == []:
        raise("Please check your phase and lane_number_dict in phase_affacted_lane()!")
    return affected_lanes


'''
input: central nodeid "node0", surrounding nodes WESN: [1,2,3,4]
output: four_lane_ids={'W':'edge1-0',"E":"edge2-0",'S':'edge4-0','N':'edge3-0'})
--- my modification ---
output: four_lane_ids={'W':'edge1-0',"E":"edge2-0",'S':'edge3-0','N':'edge4-0'})
'''

# it looks useless
def find_surrounding_lane_WESN(central_node_id, WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    tree = ET.parse('./data/one_run/cross.net.xml')
    root = tree.getroot()
    four_lane_ids_dict = collections.OrderedDict()
    for k, v in WESN_node_ids.items():
        four_lane_ids_dict[k] = root.find("./edge[@id='edge%s-%s']" % (v, central_node_id)).get('id')
    return four_lane_ids_dict


'''
coordinate mapper
'''

# it looks useless
def coordinate_mapper(x1, y1, x2, y2):
    x1 = int(x1 / grid_width)
    y1 = int(y1 / grid_width)
    x2 = int(x2 / grid_width)
    y2 = int(y2 / grid_width)
    x_max = x1 if x1 > x2 else x2
    x_min = x1 if x1 < x2 else x2
    y_max = y1 if y1 > y2 else y2
    y_min = y1 if y1 < x2 else y2
    length_num_grids = int(area_length / grid_width)
    width_num_grids = int(area_width / grid_width)
    return length_num_grids - y_max, length_num_grids - y_min, x_min, x_max

# it looks useless
def get_phase_affected_lane_traffic_max_volume(phase="NSG_SNG", tl_node_id="00",
                                 WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    four_lane_ids_dict = find_surrounding_lane_WESN(central_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
    directions = phase.split('_')
    traffic_volume_start_end = []
    for direction in directions:
        traffic_volume_start_end.append([four_lane_ids_dict[direction[0]],four_lane_ids_dict[direction[1]]])
    tree = ET.parse('./data/one_run/cross.rou.xml')
    root = tree.getroot()
    phase_volumes = []
    for lane_id in traffic_volume_start_end:
        to_lane_id="edge%s-%s"%(lane_id[1].split('-')[1],lane_id[1].split('-')[0][4:])
        time_begin = root.find("./flow[@from='%s'][@to='%s']" % (lane_id[0], to_lane_id)).get('begin')
        time_end = root.find("./flow[@from='%s'][@to='%s']" % (lane_id[0], to_lane_id)).get('end')
        volume = root.find("./flow[@from='%s'][@to='%s']" % (lane_id[0], to_lane_id)).get('number')
        phase_volumes.append((float(time_end)-float(time_begin))/float(volume))
    return max(phase_volumes)

# it looks useless
def phase_affected_lane_position(phase="NSG_SNG", tl_node_id="00",
                                 WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    '''
    input: NSG_SNG ,central nodeid "node0", surrounding nodes WESN: {"W":"1", "E":"2", "S":"3", "N":"4"}
    output: edge-ids, 4_0_0, 4_0_1, 3_0_0, 3_0_1
    [[ 98,  100,  204,  301],[ 102, 104, 104, 198]]
    '''
    four_lane_ids_dict = find_surrounding_lane_WESN(central_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
    affected_lanes = phase_affected_lane(phase=phase, four_lane_ids=four_lane_ids_dict)
    tree = ET.parse('./data/one_run/cross.net.xml')
    root = tree.getroot()
    indexes = []
    for lane_id in affected_lanes:
        lane_shape = root.find("./edge[@to='node%s']/lane[@id='%s']" % (tl_node_id, lane_id)).get('shape')
        lane_x1 = float(lane_shape.split(" ")[0].split(",")[0])
        lane_y1 = float(lane_shape.split(" ")[0].split(",")[1])
        lane_x2 = float(lane_shape.split(" ")[1].split(",")[0])
        lane_y2 = float(lane_shape.split(" ")[1].split(",")[1])
        ind_x1, ind_x2, ind_y1, ind_y2 = coordinate_mapper(lane_x1, lane_y1, lane_x2, lane_y2)
        indexes.append([ind_x1, ind_x2 + 1, ind_y1, ind_y2 + 1])
    return indexes

# it looks useless
def phases_affected_lane_positions(phases=["NSG_SNG_NWG_SEG", "NEG_SWG_NWG_SEG"], tl_node_id="00",
                                  WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    parameterArray = []
    for phase in phases:
        parameterArray += phase_affected_lane_position(phase=phase, tl_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
    return parameterArray


def vehicle_location_mapper(coordinate):
    transformX = math.floor(coordinate[0] / grid_width)
    transformY = math.floor((area_length - coordinate[1]) / grid_width)
    length_num_grids = int(area_length/grid_width)
    transformY = length_num_grids-1 if transformY == length_num_grids else transformY
    transformX = length_num_grids-1 if transformX == length_num_grids else transformX
    tempTransformTuple = (transformY, transformX)
    return tempTransformTuple

# it looks useless
def translateAction(action):
    result = 0
    for i in range(len(action)):
        result += (i + 1) * action[i]
    return result



# not global
def changeTrafficLight_7(eng, node_id,current_phase=0):  # [WNG_ESG_WSG_ENG_NWG_SEG]
    DIC_PHASE_MAP = {
        1: 2,
        2: 3,
        3: 4,
        4: 1
    }
    next_phase = DIC_PHASE_MAP[current_phase]
    next_phase_time_eclipsed = 0
    eng.set_tl_phase(node_id, next_phase)
    return next_phase, next_phase_time_eclipsed



def get_phase_vector(controlSignal, current_phase):
    # TODO: different lane number?
    controlSignal2phase = {
        # 2 phases:
        "grrr gGGG grrr gGGG".replace(" ", ""): 'WNG_ESG_EWG_WEG_WSG_ENG',
        "gGGG grrr gGGG grrr".replace(" ", ""): 'NSG_NEG_SNG_SWG_NWG_SEG',
        # 4 phases: (NSG, NSLG, WEG, NSLG)
        "gGGr grrr gGGr grrr".replace(" ", ""): 'NSG_SNG_NWG_SEG',
        "grrG grrr grrG grrr".replace(" ", ""): 'NEG_SWG_NWG_SEG',
        "grrr gGGr grrr gGGr".replace(" ", ""): 'WEG_EWG_WSG_ENG',
        "grrr grrG grrr grrG".replace(" ", ""): 'WNG_ESG_WSG_ENG'
    }
    direction_list = ["NWG", "WSG", "SEG", "ENG", "NSG", "SNG", "EWG", "WEG", "NEG", "WNG", "SWG", "ESG"]

    phase = controlSignal[current_phase] # index -> grrr...
    phase = controlSignal2phase[phase].split("_") # grrr... -> WEG...
    phase_vector = [0] * len(direction_list)
    for direction in phase:
        phase_vector[direction_list.index(direction)] = 1
    return np.array(phase_vector)

# it looks useless; TODO: 
# TODO: now we regard each tl same as node00: we should consider offset
# TODO: more patterns
def getMapOfCertainTrafficLight(current_phase=0, tl_node_id="00", area_length=600):
    current_phases_light_7 = [phases_light_7[current_phase]]
    parameterArray = phases_affected_lane_positions(phases=current_phases_light_7, tl_node_id=tl_node_id)
    length_num_grids = int(area_length / grid_width)
    resultTrained = np.zeros((length_num_grids, length_num_grids))
    for affected_road in parameterArray:
        resultTrained[affected_road[0]:affected_road[1], affected_road[2]:affected_road[3]] = 1 # 127 TODO:???
    return resultTrained



# it looks useless
def calculate_reward(tempLastVehicleStateList):
    waitedTime = 0
    stop_count = 0
    for key, vehicle_dict in tempLastVehicleStateList.items():
        if tempLastVehicleStateList[key]['speed'] < 5:
            waitedTime += 1
            #waitedTime += (1 +math.pow(tempLastVehicleStateList[key]['waitedTime']/50,2))
        if tempLastVehicleStateList[key]['former_speed'] > 0.5 and tempLastVehicleStateList[key]['speed'] < 0.5:
            stop_count += (tempLastVehicleStateList[key]['stop_count']-tempLastVehicleStateList[key]['former_stop_count'])
    #PI = (waitedTime + 10 * stop_count) / len(tempLastVehicleStateList) if len(tempLastVehicleStateList)!=0 else 0
    PI = waitedTime/len(tempLastVehicleStateList) if len(tempLastVehicleStateList)!=0 else 0
    return - PI

# TODO
MASK_IN_MAP = np.zeros((State.D_MAP_FEATURE[0], State.D_MAP_FEATURE[1]))
MASK_OUT_MAP = np.zeros((State.D_MAP_FEATURE[0], State.D_MAP_FEATURE[1]))
MASK_IN_MAP[0:75, 75-3:75] = 1
MASK_IN_MAP[75:150, 75:75+3] = 1
MASK_IN_MAP[75:75+3, 0:75] = 1
MASK_IN_MAP[75-3:75, 75:150] = 1
MASK_OUT_MAP[0:75, 75:75+3] = 1
MASK_OUT_MAP[75:150, 75-3:75] = 1
MASK_OUT_MAP[75-3:75, 0:75] = 1
MASK_OUT_MAP[75:75+3, 75:150] = 1
def getMapOfVehicles(node_id, vehicle_dict, cur_phase, next_phase):
    '''
    get the vehicle positions as NIPS paper
    :param area_length:
    :return: numpy narray
    '''
    # channel 0 & 1: position in & out
    # channel 2 & 3: speed in & out
    # channel 4 & 5: recount_waiting_time in & out
    # channel 6: cur_phase in
    # channel 7: next_phase in
    length_num_grids = int(area_length / grid_width)
    mapOfCars = np.zeros(State.D_MAP_FEATURE)

    position_map = np.zeros((State.D_MAP_FEATURE[0], State.D_MAP_FEATURE[1]))
    speed_map = np.zeros((State.D_MAP_FEATURE[0], State.D_MAP_FEATURE[1]))
    wait_time_map = np.zeros((State.D_MAP_FEATURE[0], State.D_MAP_FEATURE[1]))

    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        x, y = traci.vehicle.getPosition(vehicle_id)  # (double,double),tuple
        transform_tuple = vehicle_location_mapper(
            (x-coordinate_offset[node_id][0], y-coordinate_offset[node_id][1]))  # call the function

        if transform_tuple[0] in range(length_num_grids) and transform_tuple[1] in range(length_num_grids):
            # position
            position_map[transform_tuple[0], transform_tuple[1]] = 1
            traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
            speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64) # VAR_SPEED = 0x40
            speed_map[transform_tuple[0], transform_tuple[1]] = speed
            if vehicle_id in vehicle_dict:
                # recount_waiting_time
                # wait_time_map[transform_tuple[0], transform_tuple[1]] = vehicle_dict[vehicle_id].recount_waiting_time
                # delta_recount_waiting_time
                if speed < 0.1:
                    wait_time_map[transform_tuple[0], transform_tuple[1]] = 1

    mapOfCars[:,:,0] = np.multiply(position_map, MASK_IN_MAP)
    mapOfCars[:,:,1] = np.multiply(position_map, MASK_OUT_MAP)
    mapOfCars[:,:,2] = np.multiply(speed_map, MASK_IN_MAP)
    mapOfCars[:,:,3] = np.multiply(speed_map, MASK_OUT_MAP)
    mapOfCars[:,:,4] = np.multiply(wait_time_map, MASK_IN_MAP)
    mapOfCars[:,:,5] = np.multiply(wait_time_map, MASK_OUT_MAP)
    mapOfCars[:,:,6] = getMapOfCertainTrafficLight(cur_phase, tl_node_id="00")
    mapOfCars[:,:,7] = getMapOfCertainTrafficLight(next_phase, tl_node_id="00")

    if K.image_data_format() == 'channels_first':
        mapOfCars = np.transpose(mapOfCars, (2, 0, 1))
    return mapOfCars

def restrict_reward(reward,func="unstrict"):
    if func == "linear":
        bound = -50
        reward = 0 if reward < bound else (reward/(-bound) + 1)
    elif func == "neg_log":
        reward = math.log(-reward+1)
    else:
        pass

    return reward

# not global
def log_rewards(eng, vehicle_dict, action, rewards_info_dict, file_name, true_reward, timestamp, rewards_detail_dict_list, node_id, reward_indicator, warm_up, global_dic_waiting_time_vehicles):

    reward, reward_detail_dict = get_rewards_from_sumo(eng, vehicle_dict, action, rewards_info_dict, node_id, reward_indicator, warm_up, global_dic_waiting_time_vehicles)
    list_reward_keys = np.sort(list(reward_detail_dict.keys()))
    reward_str = "{0}, {1}, {2}".format(node_id, timestamp, action)
    for reward_key in true_reward:
        reward_str = reward_str + ", {0}".format(reward_detail_dict[reward_key][2])
    reward_str += '\n'

    fp = open(file_name, "a")
    fp.write(reward_str)
    fp.close()
    rewards_detail_dict_list.append(reward_detail_dict)

# not global
def log_rewards_control(rewards_info_dict, rewards_this_node_info_dict, file_name, timestamp, rewards_detail_dict_list, rewards_detail_this_node_dict_list, node_id, neighbor, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict):

    reward_main, reward_aux, reward_detail_dict, reward_this_node_detail_dict = get_control_rewards(rewards_info_dict, rewards_this_node_info_dict, node_id, neighbor, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict)

    list_reward_keys_aux = np.sort(list(reward_detail_dict.keys()))
    reward_str_aux = "{0}, {1}".format(node_id, timestamp)
    for reward_key in list_reward_keys_aux:
        reward_str_aux = reward_str_aux + ", {0}".format(reward_detail_dict[reward_key][2])
    reward_str_aux += '\n'

    list_reward_keys_main = np.sort(list(reward_this_node_detail_dict.keys()))
    reward_str_main = "{0}, {1}".format(node_id, timestamp)
    for reward_key in list_reward_keys_main:
        reward_str_main = reward_str_main + ", {0}".format(reward_this_node_detail_dict[reward_key][2])
    reward_str_main += '\n\n'

    fp = open(file_name, "a")
    fp.write(reward_str_aux)
    fp.write(reward_str_main)
    fp.close()
    rewards_detail_dict_list.append(reward_detail_dict)
    rewards_detail_this_node_dict_list.append(reward_this_node_detail_dict)

# not global
def get_rewards_from_sumo(eng, vehicle_dict, action, rewards_info_dict, node_id, reward_indicator, warm_up, global_dic_waiting_time_vehicles):
    reward = 0
    import copy
    reward_detail_dict = copy.deepcopy(rewards_info_dict)
    vehicle_id_entering_list = get_vehicle_id_entering(eng, node_id)
    if warm_up:
        node_index = get_node_id_list().index(node_id)
        if reward_indicator[node_index] == 0:
            reward_detail_dict['queue_length'].append(get_overall_queue_length(eng, global_listLanes[node_id]))
            reward_detail_dict['wait_time'].append(0)
            reward_detail_dict['delay'].append(0)

        elif reward_indicator[node_index] == 1:
            reward_detail_dict['queue_length'].append(0)
            reward_detail_dict['wait_time'].append(get_overall_waiting_time(eng, global_listLanes[node_id], global_dic_waiting_time_vehicles))
            reward_detail_dict['delay'].append(0)

        elif reward_indicator[node_index] == 2:
            reward_detail_dict['queue_length'].append(0)
            reward_detail_dict['wait_time'].append(0)
            reward_detail_dict['delay'].append(get_overall_delay(eng, global_listLanes[node_id]))
    else:
        if reward_indicator[node_id] == 0:
            reward_detail_dict['queue_length'].append(get_overall_queue_length(eng, global_listLanes[node_id]))
            reward_detail_dict['wait_time'].append(0)
            reward_detail_dict['delay'].append(0)

        elif reward_indicator[node_id] == 1:
            reward_detail_dict['queue_length'].append(0)
            reward_detail_dict['wait_time'].append(get_overall_waiting_time(eng, global_listLanes[node_id], global_dic_waiting_time_vehicles))
            reward_detail_dict['delay'].append(0)

        elif reward_indicator[node_id] == 2:
            reward_detail_dict['queue_length'].append(0)
            reward_detail_dict['wait_time'].append(0)
            reward_detail_dict['delay'].append(get_overall_delay(eng, global_listLanes[node_id]))

    reward_detail_dict['emergency'].append(0)
    reward_detail_dict['duration'].append(0)
    reward_detail_dict['flickering'].append(get_flickering(action))
    reward_detail_dict['partial_duration'].append(0)
    reward_detail_dict['num_of_vehicles_left'].append(0)
    reward_detail_dict['duration_of_vehicles_left'].append(0)

    for k, v in reward_detail_dict.items():
        if v[0]:  # True or False
            reward += v[1]*v[2]
    reward = restrict_reward(reward)#,func="linear")
    return reward, reward_detail_dict

def get_control_rewards(rewards_info_dict, rewards_this_node_info_dict, node_id, neighbor, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict):
    reward = 0
    import copy
    travel_time_list = []
    reward_detail_dict = copy.deepcopy(rewards_info_dict)   # main: this node
    reward_this_node_detail_dict = copy.deepcopy(rewards_this_node_info_dict)   # aux: local
    if len(list(all_vehicles_location_enter_time_dict[node_id].values())) == 0:
        local_travel_time_this_node = 0
    else:
        local_travel_time_this_node = np.sum(list(all_vehicles_location_enter_time_dict[node_id].values()))
    reward_this_node_detail_dict['local_travel_time'].append(local_travel_time_this_node)
    reward_this_node_detail_dict['average_local_travel_time'].append(0)

    if len(list(all_vehicles_this_node_enter_time_dict[node_id].values())) == 0:
        travel_time_this_node = 0
    else:
        travel_time_this_node = np.sum(list(all_vehicles_this_node_enter_time_dict[node_id].values()))
    reward_detail_dict['this_node_travel_time'].append(travel_time_this_node)

    for k, v in reward_detail_dict.items():
        if v[0]:  # True or False
            reward += v[1]*v[2]
    reward_main = restrict_reward(reward)#,func="linear")

    for k, v in reward_this_node_detail_dict.items():
        if v[0]:  # True or False
            reward += v[1]*v[2]
    reward_aux = restrict_reward(reward)#,func="linear")
    return reward_main, reward_aux, reward_detail_dict, reward_this_node_detail_dict

# not global
def get_rewards_from_dict_list(rewards_detail_dict_list):
    reward = 0
    for i in range(len(rewards_detail_dict_list)):
        for k, v in rewards_detail_dict_list[i].items():
            if v[0]:  # True or False
                reward += v[1] * v[2]
    reward = restrict_reward(reward)
    return reward

def get_overall_queue_length(eng, listLanes):
    overall_queue_length = 0
    global_vehicles = eng.get_lane_waiting_vehicle_count()
    for lane in listLanes:
        overall_queue_length += global_vehicles[lane]
    return overall_queue_length


def get_vehicle_speed(listVehicles):
    list_vehicle_speed = []
    for vehicle_id in listVehicles:
        list_vehicle_speed.append(traci.vehicle.getSpeed(vehicle_id))


def get_overall_waiting_time(eng, listLanes, global_dic_waiting_time_vehicles):
    overall_waiting_time = 0
    for lane in listLanes:
        for vehicle in global_dic_waiting_time_vehicles[lane]:
            overall_waiting_time += global_dic_waiting_time_vehicles[lane][vehicle] / 60.0
    return overall_waiting_time

def get_overall_recount_waiting_time(node_id, listLanes, vehicle_dict):
    overall_recount_waiting_time = 0
    for lane in listLanes:
        vehicle_id_lane_list = traci.lane.getLastStepVehicleIDs(lane)
        for vehicle_id, vehicle in vehicle_dict.items():
            if vehicle_id in vehicle_id_lane_list:
                overall_recount_waiting_time += vehicle.recount_waiting_time
    return overall_recount_waiting_time / 60

def get_overall_delta_waiting_time(node_id, listLanes, vehicle_dict):
    overall_delta_waiting_time = 0
    for lane in listLanes:
        vehicle_id_lane_list = traci.lane.getLastStepVehicleIDs(lane)
        for vehicle_id in vehicle_id_lane_list:
            traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
            speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64) # VAR_SPEED = 0x40
            if speed < 0.1:
                overall_delta_waiting_time += 1
    return overall_delta_waiting_time

def get_overall_delay(eng, listLanes):
    overall_delay = 0
    global_vehicle_speed = eng.get_vehicle_speed()
    global_lane_vehicle = eng.get_lane_vehicles()
    for lane in listLanes:
        vehicle_of_this_lane = global_lane_vehicle[lane]
        speed_of_this_lane = [global_vehicle_speed[vehicle] for vehicle in vehicle_of_this_lane]
        if len(speed_of_this_lane) != 0:
            mean_speed = sum(speed_of_this_lane) / max(len(speed_of_this_lane), 1)
            max_speed = max(speed_of_this_lane)
            overall_delay += 1 - mean_speed / max(max_speed, 1)
    return overall_delay

def get_flickering(action):
    return action

# calculate number of emergency stops by vehicle
def get_num_of_emergency_stops(vehicle_dict):
    emergency_stops = 0
    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
        current_speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64) # VAR_SPEED = 0x40
        if (vehicle_id in vehicle_dict.keys()):
            vehicle_former_state = vehicle_dict[vehicle_id]
            if current_speed - vehicle_former_state.speed < -4.5:
                emergency_stops += 1
        else:
            # print("##New car coming")
            if current_speed - Vehicles.initial_speed < -4.5:
                emergency_stops += 1
    if len(vehicle_dict) > 0:
        return emergency_stops/len(vehicle_dict)
    else:
        return 0

def get_partial_travel_time_duration(eng, vehicle_dict, vehicle_id_list):
    travel_time_duration = 0
    for vehicle_id in vehicle_id_list:
        if (vehicle_id in vehicle_dict.keys()) and (vehicle_dict[vehicle_id].first_stop_time != -1):
            travel_time_duration += (eng.get_current_time() - vehicle_dict[vehicle_id].first_stop_time) / 60.0
    if len(vehicle_id_list) > 0:
        return travel_time_duration  # /len(vehicle_id_list)
    else:
        return 0


def get_travel_time_duration(eng, vehicle_dict, vehicle_id_list):
    travel_time_duration = 0
    for vehicle_id in vehicle_id_list:
        if (vehicle_id in vehicle_dict.keys()):
            travel_time_duration += (eng.get_current_time() - vehicle_dict[vehicle_id].enter_time) / 60.0
    if len(vehicle_id_list) > 0:
        return travel_time_duration  # /len(vehicle_id_list)
    else:
        return 0


### added! Colight's calculating method of average travel time
def update_dic_lane_vehicle_arrive_leave_time(eng, global_list_node_lane_vehicle):
    vehicles_each_lane_new = eng.get_lane_vehicles()    # new
    for node_id_1 in get_node_id_list():
        for lane in global_all_lanes_each_node[node_id_1]:
            s_old = set(global_list_node_lane_vehicle[lane])
            s_new = set(vehicles_each_lane_new[lane])
            if lane in global_listLanes[node_id_1]:
                for vehicle_id in s_old - s_new:  # it has left the lane
                    lane_vehicle_arrive_leave_time_dict[node_id_1][vehicle_id]["leave_time"] = eng.get_current_time()
                    lane_vehicle_arrive_leave_time_dict[node_id_1][vehicle_id]["run_time"] = lane_vehicle_arrive_leave_time_dict[node_id_1][vehicle_id]["leave_time"] - lane_vehicle_arrive_leave_time_dict[node_id_1][vehicle_id]["arrive_time"]
            for vehicle_id in s_new - s_old:
                lane_vehicle_arrive_leave_time_dict[node_id_1][vehicle_id] = {}
                lane_vehicle_arrive_leave_time_dict[node_id_1][vehicle_id]["arrive_time"] = eng.get_current_time()
    global_list_node_lane_vehicle = vehicles_each_lane_new

    return global_list_node_lane_vehicle

def update_vehicles_state(eng, global_dic_vehicles):
    global_dic_vehicle_set = set(global_dic_vehicles.keys())  # old list
    # list_lane_new_vehicles = eng.get_lane_vehicles()
    # global_speed = eng. get_vehicle_speed()
    # list_new_vehicles = list(global_speed.keys())
    # [list_new_vehicles.extend(vehicle_id) for vehicle_id in list_lane_new_vehicles.values()]
    list_new_vehicles = eng.get_vehicles(include_waiting=True)
    global_vehicle_id_set = set(list_new_vehicles)  # new list

    # all_new_list = traci.vehicle.getIDList()
    for vehicle_id in global_dic_vehicle_set - global_vehicle_id_set:  # it has left the whole network
        all_vehicles_enter_time_dict[vehicle_id] = eng.get_current_time() - global_dic_vehicles[vehicle_id]
        del (global_dic_vehicles[vehicle_id])  # old -= (old-new): old = old intersect new

    for vehicle_id in global_vehicle_id_set - global_dic_vehicle_set:
        global_dic_vehicles[vehicle_id] = eng.get_current_time()

    return global_dic_vehicles

def find_neighbor_lanes():
    node_id_list = get_node_id_list()
    if not hasattr(find_neighbor_lanes, 'neighbor_lanes_list'):
        find_neighbor_lanes.neighbor_lanes_list = collections.OrderedDict()
        for node_id_1 in node_id_list:
            find_neighbor_lanes.neighbor_lanes_list[node_id_1] = []
            neighbor_this_node = find_neighbors()[node_id_1]
            for node_id_index in neighbor_this_node:
                if node_id_index != -1:
                    node_id_2 = node_id_list[node_id_index]
                    listLanes = global_listLanes[node_id_2]
                    for lane in listLanes:
                        find_neighbor_lanes.neighbor_lanes_list[node_id_1].append(lane)
    return find_neighbor_lanes.neighbor_lanes_list

def update_vehicles_location(eng, global_dic_location_vehicles, global_dic_this_node_vehicles, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict, global_dic_waiting_time_vehicles): # old_all
    global_dic_vehicle_location_set = collections.OrderedDict()  # old
    global_vehicle_id_location_set = collections.OrderedDict()  # new
    global_current_location_vehicles = collections.OrderedDict()  # new_all

    global_dic_vehicle_this_node_set = collections.OrderedDict()  # old
    global_vehicle_id_this_node_set = collections.OrderedDict()  # new
    global_current_this_node_vehicles = collections.OrderedDict()  # new_all

    global_vehicles_speed = eng.get_vehicle_speed()
    # global_still_vehicles = {k: v for k, v in global_vehicles_speed.items() if v <= 0.1}
    vehicle_id_of_each_lane = collections.OrderedDict()
    m1 = eng.get_lane_vehicles()

    for node_id_1 in get_node_id_list():
        global_current_location_vehicles[node_id_1] = []
        global_current_this_node_vehicles[node_id_1] = []

    for node_id_1, listLanes in global_listLanes.items():
        for lane in listLanes:
            vehicle_id_of_each_lane[lane] = m1[lane]
            each_lane_vehicleID = m1[lane]
            for vehicleID in each_lane_vehicleID:
                global_current_this_node_vehicles[node_id_1].append(vehicleID)

    neighbor_lanes = find_neighbor_lanes()
    for node_id_1, neighbor_lane in neighbor_lanes.items():
        for lane in neighbor_lane:
            each_lane_vehicleID = m1[lane]
            for vehicleID in each_lane_vehicleID:
                global_current_location_vehicles[node_id_1].append(vehicleID)


    for node_id_1 in get_node_id_list():
        global_dic_vehicle_location_set[node_id_1] = set(global_dic_location_vehicles[node_id_1].keys())  # old list
        # 此时刻node_id_1控制的lanes上的车辆id
        global_vehicle_id_location_set[node_id_1] = set(global_current_location_vehicles[node_id_1])  # new list

        global_dic_vehicle_this_node_set[node_id_1] = set(global_dic_this_node_vehicles[node_id_1].keys())
        global_vehicle_id_this_node_set[node_id_1] = set(global_current_this_node_vehicles[node_id_1])

        for vehicle_id in global_dic_vehicle_location_set[node_id_1] - global_vehicle_id_location_set[node_id_1]:  # it has left the whole network
            all_vehicles_location_enter_time_dict[node_id_1][vehicle_id] = eng.get_current_time() - \
                                                                           global_dic_location_vehicles[node_id_1][
                                                                               vehicle_id]
            del (global_dic_location_vehicles[node_id_1][vehicle_id])  # old -= (old-new): old = old intersect new
        for vehicle_id in global_vehicle_id_location_set[node_id_1] - global_dic_vehicle_location_set[node_id_1]:
            global_dic_location_vehicles[node_id_1][vehicle_id] = eng.get_current_time()


        for vehicle_id in global_dic_vehicle_this_node_set[node_id_1] - global_vehicle_id_this_node_set[node_id_1]:  # it has left the whole network
            all_vehicles_this_node_enter_time_dict[node_id_1][vehicle_id] = eng.get_current_time() - \
                                                                           global_dic_this_node_vehicles[node_id_1][
                                                                               vehicle_id]
            del (global_dic_this_node_vehicles[node_id_1][vehicle_id])  # old -= (old-new): old = old intersect new
        for vehicle_id in global_vehicle_id_this_node_set[node_id_1] - global_dic_vehicle_this_node_set[node_id_1]:
            global_dic_this_node_vehicles[node_id_1][vehicle_id] = eng.get_current_time()


    for lane in vehicle_id_of_each_lane.keys():
        global_wait_time_old = set(global_dic_waiting_time_vehicles[lane].keys())  # old list
        # 此时刻node_id_1控制的lanes上的车辆id
        global_wait_time_new = set(vehicle_id_of_each_lane[lane])  # new list

        for vehicle_id in global_wait_time_old - global_wait_time_new:  # it has left the whole network
            del (global_dic_waiting_time_vehicles[lane][vehicle_id])  # old -= (old-new): old = old intersect new

        for vehicle_id in global_wait_time_new - global_wait_time_old:
            if global_vehicles_speed[vehicle_id] <= 0.1:
                global_dic_waiting_time_vehicles[lane][vehicle_id] = 1

        intersection = [i for i in global_wait_time_new if i in global_wait_time_old]
        for vehicle_id in intersection:
            if global_vehicles_speed[vehicle_id] <= 0.1:
                global_dic_waiting_time_vehicles[lane][vehicle_id] += 1
            else:
                del (global_dic_waiting_time_vehicles[lane][vehicle_id])

    return global_dic_location_vehicles, global_dic_this_node_vehicles, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict, global_dic_waiting_time_vehicles

# --- my modification ---
# add params
def status_calculator(eng, global_vehicle_id_list):
    status = collections.OrderedDict()

    global_lane_queue = eng.get_lane_waiting_vehicle_count()
    global_lane_vihicle_count = eng.get_lane_vehicle_count()
    # global_lane_waiting_time = eng.

    for node_id, listLanes in global_listLanes.items():
        # --- my modification ---
        laneQueueTracker = []
        laneNumVehiclesTracker = []
        laneWaitingTracker = []

        # ================= COUNT HALTED VEHICLES (I.E. QUEUE SIZE) (12 elements)
        for lane in listLanes:
            laneNumVehiclesTracker.append(global_lane_vihicle_count[lane])
        if len(listLanes) < State.D_NUM_OF_VEHICLES[0]:
            laneNumVehiclesTracker.extend([0 for _ in range(State.D_NUM_OF_VEHICLES[0] - len(listLanes))])

        # ================ count vehicles in lane
        for lane in listLanes:
            laneQueueTracker.append(global_lane_queue[lane])
        if len(listLanes) < State.D_QUEUE_LENGTH[0]:
            laneQueueTracker.extend([0 for _ in range(State.D_QUEUE_LENGTH[0] - len(listLanes))])

        # ================ cum waiting time in minutes
        for lane in listLanes:
            laneWaitingTracker.append(0)
        if len(listLanes) < State.D_WAITING_TIME[0]:
            laneWaitingTracker.extend([0 for _ in range(State.D_WAITING_TIME[0] - len(listLanes))])

        # vehicle_id_list = global_vehicle_id_list[node_id]
        # ================ get position matrix of vehicles on lanes
        # need paramter: current_phase
        # cur_phase = current_phase[node_id]
        # next_phase = (current_phase[node_id]+1) % len(get_node_phases(node_id))
        # mapOfCars = getMapOfVehicles(node_id, vehicle_id_list, cur_phase, next_phase)
        mapOfCars = None

        status[node_id] = [laneQueueTracker, laneNumVehiclesTracker, laneWaitingTracker, mapOfCars]

    return status

def get_vehicle_id_entering(eng, node_id):
    vehicle_id_entering = []
    m2 = eng.get_lane_vehicles()
    for lane in entering_lanes[node_id]:
        vehicle_id_entering.extend(m2[lane])

    return vehicle_id_entering


def get_vehicle_id_leaving(vehicle_dict, node_id):
    vehicle_id_leaving = []
    vehicle_id_entering = get_vehicle_id_entering(node_id)
    for vehicle_id in vehicle_dict.keys():
        if not(vehicle_id in vehicle_id_entering) and vehicle_dict[vehicle_id].entering:
            vehicle_id_leaving.append(vehicle_id)

    return vehicle_id_leaving


# it looks useless
def get_car_on_red_and_green(cur_phase):
    # --- my modification ---
    # listLanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2',
                 # 'edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
    vehicle_red = []
    vehicle_green = []
    if cur_phase == 1:
        # red_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2']
        # green_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
        red_lanes = ['edge1-00_0', 'edge1-00_1', 'edge1-00_2', 'edge2-00_0', 'edge2-00_1', 'edge2-00_2']
        green_lanes = ['edge3-00_0', 'edge3-00_1', 'edge3-00_2', 'edge4-00_0', 'edge4-00_1', 'edge4-00_2']
    else:
        # red_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
        # green_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2']
        red_lanes = ['edge3-00_0', 'edge3-00_1', 'edge3-00_2', 'edge4-00_0', 'edge4-00_1', 'edge4-00_2']
        green_lanes = ['edge1-00_0', 'edge1-00_1', 'edge1-00_2', 'edge2-00_0', 'edge2-00_1', 'edge2-00_2']
    for lane in red_lanes:
        vehicle_red.append(traci.lane.getLastStepVehicleNumber(lane))
    for lane in green_lanes:
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
        omega = 0
        for vehicle_id in vehicle_ids:
            traci.vehicle.subscribe(vehicle_id, (tc.VAR_DISTANCE, tc.VAR_LANEPOSITION))
            distance = traci.vehicle.getSubscriptionResults(vehicle_id).get(132)
            if distance > 100:
                omega += 1
        vehicle_green.append(omega)

    return max(vehicle_red), max(vehicle_green)

# it looks useless
# def get_status_img(current_phase,tl_node_id):
#     mapOfCars = getMapOfVehicles(tl_node_id)

#     current_observation = [mapOfCars]
#     return current_observation


def run(eng, joint_action, current_phase, current_phase_duration, global_vehicle_dict, global_dic_location_vehicles, global_dic_this_node_vehicles, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict, global_dic_waiting_time_vehicles, global_list_node_lane_vehicle, rewards_info_dict, true_reward, f_log_rewards, rewards_detail_dict_list, reward_indicator, warm_up):
    global_return_phase = collections.OrderedDict()
    global_return_phase_duration = collections.OrderedDict()
    for node_id, action in joint_action.items():
        global_return_phase[node_id] = current_phase[node_id]
        global_return_phase_duration[node_id] = current_phase_duration[node_id]


    Yellow = 0

    yellow_node_id = []
    for node_id, action in joint_action.items():
        if action == 1:
            yellow_node_id.append(node_id)

    if yellow_node_id:
        for i in range(3):
            for node_id, action in joint_action.items():
                if action == 1:
                    eng.set_tl_phase(node_id, Yellow)
                else:
                    global_return_phase_duration[node_id] += 1
                timestamp = eng.get_current_time()
                log_rewards(eng, None, action, rewards_info_dict, f_log_rewards, true_reward, timestamp+1, rewards_detail_dict_list[node_id], node_id, reward_indicator, warm_up, global_dic_waiting_time_vehicles)
            eng.next_step()
            global_vehicle_dict = update_vehicles_state(eng, global_vehicle_dict)
            if not warm_up:
                global_dic_location_vehicles, global_dic_this_node_vehicles, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict, global_dic_waiting_time_vehicles \
                    = update_vehicles_location(eng, global_dic_location_vehicles, global_dic_this_node_vehicles, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict, global_dic_waiting_time_vehicles)
                global_list_node_lane_vehicle = update_dic_lane_vehicle_arrive_leave_time(eng, global_list_node_lane_vehicle)

        for node_id in yellow_node_id:
            global_return_phase[node_id], _ = changeTrafficLight_7(eng=eng, node_id=node_id, current_phase=current_phase[node_id])  # change traffic light in SUMO according to actionToPerform
            global_return_phase_duration[node_id] = 0


    timestamp = eng.get_current_time()
    eng.next_step()
    for node_id, action in joint_action.items():
        log_rewards(eng, None, action, rewards_info_dict, f_log_rewards, true_reward, timestamp, rewards_detail_dict_list[node_id], node_id, reward_indicator, warm_up, global_dic_waiting_time_vehicles)
    global_vehicle_dict = update_vehicles_state(eng, global_vehicle_dict)
    if not warm_up:
        global_dic_location_vehicles, global_dic_this_node_vehicles, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict, global_dic_waiting_time_vehicles \
            = update_vehicles_location(eng, global_dic_location_vehicles, global_dic_this_node_vehicles, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict, global_dic_waiting_time_vehicles)
        global_list_node_lane_vehicle = update_dic_lane_vehicle_arrive_leave_time(eng, global_list_node_lane_vehicle)
    for node_id in global_return_phase_duration.keys():
        global_return_phase_duration[node_id] += 1
    return global_return_phase, global_return_phase_duration, global_vehicle_dict, global_dic_location_vehicles, global_dic_this_node_vehicles, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict, global_dic_waiting_time_vehicles, global_list_node_lane_vehicle


def run_control(eng, rewards_info_dict, rewards_this_node_info_dict, f_log_rewards_control, rewards_detail_dict_list, rewards_detail_this_node_dict_list, all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict):
    timestamp = eng.get_current_time()
    neighbors = find_neighbors()
    for node_id_1 in get_node_id_list():
        log_rewards_control(rewards_info_dict, rewards_this_node_info_dict, f_log_rewards_control, timestamp, rewards_detail_dict_list[node_id_1], rewards_detail_this_node_dict_list[node_id_1], node_id_1, neighbors[node_id_1], all_vehicles_location_enter_time_dict, all_vehicles_this_node_enter_time_dict)


# it looks useless
def get_base_min_time(traffic_volumes,min_phase_time):
    traffic_volumes=np.array([36,72,0])
    min_phase_times=np.array([10,35,35])
    for i, min_phase_time in enumerate(min_phase_times):
        ratio=min_phase_time/traffic_volumes[i]
        traffic_volumes_ratio=traffic_volumes/ratio

# it looks useless
# def phase_vector_to_number(phase_vector,phases_light=phases_light_7):
#     phase_vector_7 = []
#     result = -1
#     for i in range(len(phases_light)):
#         phase_vector_7.append(str(get_phase_vector(i)))
#     if phase_vector in phase_vector_7:
#         return phase_vector_7.index(phase_vector)
#     else:
#         raise ("Phase vector %s is not in phases_light %s"%(phase_vector,str(phase_vector_7)))



if __name__ == '__main__':
    pass
    print(get_phase_vector(0))
    print(get_phase_vector(1))
    # phase_vector_to_number('[0 1 0 1 0 0 1 1 0 1 0 1]')
    pass
    # traci.close()
