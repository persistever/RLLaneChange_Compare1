# coding:utf-8

import math
import numpy as np
import traci
import traci.constants as tc
from surrounding import Surrounding

LANE_WIDTH = 3.2
RADAR_LIMIT = 200
SPEED_LIMIT_HIGH = 35
SPEED_LIMIT_LOW = 0


class EgoVehicle:
    @staticmethod
    def _form_mission(m_type, c_type, ax, ay, vx, vy):
        return {'m_type': m_type, 'c_type': c_type, 'axCtl': ax, 'ayCtl': ay, 'vxCtl': vx, 'vyCtl': vy}

    def __init__(self, vehicle_id):
        self.id = vehicle_id
        self.data = None  # 从subscribe订阅的所有数据
        self.surroundings = None
        self.preX = 0  # 之前的一个位置，用来估算纵向车速
        self.preY = 0  # 之前的一个位置，用来横向车速
        self.x = 0  # 当前的x全局坐标
        self.y = 0  # 当前的y全局坐标
        self.yLane = 0  # 车辆在当前车道的相对横向位置
        self.vx0 = 0
        self.vx = 0
        self.vy = 0
        self.preLaneIndex = -1
        self.laneIndex = -1
        self.goalLaneIndex = -1
        self.laneID = ''
        self.edgeID = ''
        self.nextEdgeID = ''
        self.nLane = 1
        self.nNextLane = 4
        self.laneX = 0
        self.laneY = 0
        self.timeStep = 0.01
        self.goalY = 0
        self.axCtl = 0
        self.ayCtl = 0
        self.vyCtl = 0
        self.vxCtl = 1 - self.vx0
        self.angleCtl = 90
        self.lastChangeLaneTime = 0
        self.missionList = []
        self.neighbourVehicleList = []
        self.midFrontVehicleList = []
        self.midRearVehicleList = []
        self.leadingVehicle = None
        self.followingVehicle = None
        self.gapFrontVehicle = None
        self.gapRearVehicle = None
        self.yBeforeLaneChange = 0
        self.state = 0
        self.outOfRoad = False
        self.edgeList = ['gneE0', 'HuiheJ1', 'gneE1', 'HuiheJ2', 'gneE2', 'Zadao1', 'Zadao2']
        self.laneSpeedLimitList = [0.0, 33.3, 22.0, 27.0, 33.0, 0.0]
        self.laneNumberDict = {}

    def subscribe_ego_vehicle(self):
        traci.vehicle.subscribe(self.id, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ROAD_ID))
        self.surroundings = Surrounding("ego")
        self.surroundings.surrounding_init()
        self.laneNumberDict = self.surroundings.get_all_edge_lane_number_dict()

    def fresh_data(self):
        self.data = traci.vehicle.getSubscriptionResults(self.id)

        if self.data is not None:
            self._set_xy()
            if self.x > 0:
                self._set_speed()
                self._set_road_id()
                self._set_lane_index()
                self._set_n_lane()
                self._set_next_n_lane()
                self._set_y_lane_lateral()
                self._set_angle()
                self.surroundings.get_surroundings(self.edgeID, self.x, self.y)
                self.neighbourVehicleList = self.surroundings.get_neighbor_list()

        if self.neighbourVehicleList is not None:
            self.midFrontVehicleList = self.surroundings.get_mid_leader_neighbor_list()
            self.midRearVehicleList = self.surroundings.get_mid_follower_neighbor_list()

        self._set_leading_vehicle()
        self._set_following_vehicle()

        if self.state == 2 and len(self.missionList) != 0:  # 如果是准备换道的任务,需要不断更新Gap前后车的信息
            is_need_rear_virtual = False  # 表示是否需要虚拟数据去预测gap后车
            is_need_front_virtual = False  # 表示是否需要虚拟数据去预测gap前车
            gap_rear_vehicle_x = 0
            gap_rear_vehicle_y = 0
            gap_rear_vehicle_vx = 0
            gap_front_vehicle_x = 0
            gap_front_vehicle_y = 0
            gap_front_vehicle_vx = 0
            if self.gapRearVehicle['virtual'] is False:  # 如果不是虚拟车,即为真实车
                gap_rear_vehicle = traci.vehicle.getSubscriptionResults(self.gapRearVehicle['name'])
                if gap_rear_vehicle is not None:
                    gap_rear_vehicle_x = gap_rear_vehicle[tc.VAR_POSITION][0]
                    gap_rear_vehicle_y = gap_rear_vehicle[tc.VAR_POSITION][1]
                    gap_rear_vehicle_vx = gap_rear_vehicle[tc.VAR_SPEED]
                    if self.gapRearVehicle['lane_index_relative'] == 0:  # 如果选择的是左车道的Gap
                        if -0.5 <= gap_rear_vehicle_y - self.y <= 4.3:  # 判断该真实车是否还在该车道上
                            is_need_rear_virtual = False
                        else:
                            is_need_rear_virtual = True
                    elif self.gapRearVehicle['lane_index_relative'] == 2:  # 如果选择的是右车道的Gap
                        if -4.3 <= gap_rear_vehicle_y - self.y <= 0.5:
                            is_need_rear_virtual = False
                        else:
                            is_need_rear_virtual = True
                else:  # 有可能虽然不是虚拟车,但这个时候已经获取不到该车的信息
                    is_need_rear_virtual = True
            else:  # Gap后车是虚拟车
                is_need_rear_virtual = True

            if is_need_rear_virtual is False:  # 真实车,直接获取
                self.gapRearVehicle['position_x'] = gap_rear_vehicle_x
                self.gapRearVehicle['position_y'] = gap_rear_vehicle_y
                self.gapRearVehicle['speed'] = gap_rear_vehicle_vx
                self.gapRearVehicle['relative_position_x'] = gap_rear_vehicle_x - self.x
                self.gapRearVehicle['relative_position_y'] = gap_rear_vehicle_y - self.y
            elif is_need_rear_virtual is True:  # 虚拟车,车速用data_process里面传过来的, 或者是之前真实车遗留下来的
                self.gapRearVehicle['relative_position_x'] += (self.gapRearVehicle['speed'] - self.vx) * self.timeStep
                self.gapRearVehicle['position_x'] = self.x + self.gapRearVehicle['relative_position_x']
                self.gapRearVehicle['position_y'] = self.y + self.gapRearVehicle['relative_position_y']

            if self.gapFrontVehicle['virtual'] is False:
                gap_front_vehicle = traci.vehicle.getSubscriptionResults(self.gapFrontVehicle['name'])
                if gap_front_vehicle is not None:
                    gap_front_vehicle_x = gap_front_vehicle[tc.VAR_POSITION][0]
                    gap_front_vehicle_y = gap_front_vehicle[tc.VAR_POSITION][1]
                    gap_front_vehicle_vx = gap_front_vehicle[tc.VAR_SPEED]
                    if self.gapFrontVehicle['lane_index_relative'] == 0:
                        if -0.5 <= gap_front_vehicle_y - self.y <= 5.0:
                            is_need_front_virtual = False
                        else:
                            is_need_front_virtual = True
                    elif self.gapFrontVehicle['lane_index_relative'] == 2:
                        if -5.0 <= gap_front_vehicle_y - self.y <= 0.5:
                            is_need_front_virtual = False
                        else:
                            is_need_front_virtual = True
                else:
                    is_need_front_virtual = True
            else:
                is_need_front_virtual = True

            if is_need_front_virtual is False:
                self.gapFrontVehicle['position_x'] = gap_front_vehicle_x
                self.gapFrontVehicle['position_y'] = gap_front_vehicle_y
                self.gapFrontVehicle['speed'] = gap_front_vehicle_vx
                self.gapFrontVehicle['relative_position_x'] = self.gapFrontVehicle['position_x'] - self.x
                self.gapFrontVehicle['relative_position_y'] = self.gapFrontVehicle['position_y'] - self.y
            elif is_need_front_virtual is True:
                self.gapFrontVehicle['relative_position_x'] += (self.gapFrontVehicle['speed'] - self.vx) * self.timeStep
                self.gapFrontVehicle['position_x'] = self.x + self.gapFrontVehicle['relative_position_x']
                self.gapFrontVehicle['position_y'] = self.y + self.gapFrontVehicle['relative_position_y']

    def print_data(self):
        print("自车信息："+str(self.data)+' 车速： '+str(self.vx)+' edgeID: '+str(self.edgeID)+' laneIndex: '+str(self.laneIndex))
        print("他车信息"+str(self.neighbourVehicleList))
        # print("车道index: "+str(self.laneIndex))
        # print("道路ID: "+str(self.edgeID))
        # print("目标车道index: "+str(self.goalLaneIndex))
        # print("Gap前车信息"+str(self.gapFrontVehicle))
        # print("Gap后车信息"+str(self.gapRearVehicle))
        # print("前车信息: "+str(self.leadingVehicle))
        # if len(self.missionList) != 0:
        #     print("当前任务"+str(self.missionList[0]))
        # print("下一个edge的车道数："+str(self.nNextLane))

    def get_lmr_speed_limit(self):  # 获取道路限速信息
        temp_index = self.laneIndex + 1
        if self.laneIndex < 0:
            temp_index = 1
        elif self.laneIndex > 3:
            temp_index = 4
        return self.laneSpeedLimitList[temp_index-1:temp_index+2]

    def print_current_lane_index(self):
        print("当前车道: "+str(self.laneIndex))

    def print_goal_lane_index(self):
        print("目标车道: "+str(self.goalLaneIndex))

    def _set_xy(self):
        self.preX = self.x  # 记录上一个Step的位置,用来计算车速
        self.preY = self.y
        self.x = self.data[tc.VAR_POSITION][0]
        self.y = self.data[tc.VAR_POSITION][1]

    def _set_speed(self):
        self.vx = (self.x - self.preX) / self.timeStep
        self.vy = (self.y - self.preY) / self.timeStep
        self.vx0 = self.data[tc.VAR_SPEED]

    def _set_lane_index(self):
        self.laneIndex = self.nLane - math.ceil(-self.y / LANE_WIDTH)  # 用横向位置硬算第几个车道

    def _set_y_lane_lateral(self):
        self.yLane = self.y - (- self.nLane + self.laneIndex + 0.5) * LANE_WIDTH

    def _set_angle(self):
        if self.vx != 0:
            self.angleCtl = 90 - math.atan(self.vy/self.vx)/math.pi*180.0
        else:
            self.angleCtl = 90

    def _set_road_id(self):
        if self.edgeID != self.data[tc.VAR_ROAD_ID]:
            self.edgeID = self.data[tc.VAR_ROAD_ID]
            if self.y < -LANE_WIDTH * self.nNextLane or self.y > 0:  # 在edgeID切换的时候有可能会横向出界,判断一下
                self.outOfRoad = True

    def _set_n_lane(self):
        if self.edgeID.find('Huihe') == -1 and self.edgeID.find('Zadao') == -1:  #没有看到Huihe和Zadao,说明在正常道路上
            self.nLane = self.laneNumberDict[self.edgeID]
        else:
            self.nLane = 4

    def _set_next_n_lane(self):
        if self.edgeID.find(':HuiheJ1') != -1:
            edge_index = 1
        elif self.edgeID.find(':HuiheJ2') != -1:
            edge_index = 3
        elif self.edgeID.find('Zadao') != -1:
            edge_index = 4
        else:
            edge_index = self.edgeList.index(self.edgeID)

        next_edge_index = edge_index + 1
        if next_edge_index == 1:
            self.nNextLane = 4
        elif next_edge_index == 3:
            self.nNextLane = 4
        elif next_edge_index > 4:
            self.nNextLane = 4
        else:
            self.nNextLane = self.laneNumberDict[self.edgeList[next_edge_index]]
        # print("当前车道: " + str(self.edgeList[edge_index]) + " 下一车道:" + str(self.edgeList[next_edge_index]))

    def _set_leading_vehicle(self):
        self.leadingVehicle = {}
        self.midFrontVehicleList.sort(key=lambda x: x['relative_position_x'])  # 按照从近到远排序
        if len(self.midFrontVehicleList) != 0:
            self.leadingVehicle = self.midFrontVehicleList[0]
            self.leadingVehicle['virtual'] = False
        else:
            self.leadingVehicle['virtual'] = True
            self.leadingVehicle['name'] = 'virtual_l'
            self.leadingVehicle['position_x'] = self.x + RADAR_LIMIT
            self.leadingVehicle['position_y'] = self.y
            self.leadingVehicle['speed'] = 30
        self.leadingVehicle['relative_position_x'] = self.leadingVehicle['position_x'] - self.x
        self.leadingVehicle['relative_position_y'] = self.leadingVehicle['position_y'] - self.y

    def _set_following_vehicle(self):
        self.followingVehicle = {}
        self.midRearVehicleList.sort(key=lambda x: x['relative_position_x'], reverse=True) # 按照从近到远排序
        if len(self.midRearVehicleList) != 0:
            self.followingVehicle = self.midRearVehicleList[0]
            self.followingVehicle['virtual'] = False
        else:
            self.followingVehicle['virtual'] = True
            self.followingVehicle['name'] = 'virtual_f'
            self.followingVehicle['position_x'] = self.x - RADAR_LIMIT
            self.followingVehicle['position_y'] = self.y
            self.followingVehicle['speed'] = 30
        self.followingVehicle['relative_position_x'] = self.followingVehicle['position_x'] - self.x
        self.followingVehicle['relative_position_y'] = self.followingVehicle['position_y'] - self.y

    def set_busy(self):
        self.state = 1

    def get_n_lane(self):
        return self.nLane

    def get_next_n_lane(self):
        return self.nNextLane

    def get_state(self):
        return self.state

    def get_speed(self):
        return self.vx

    def get_lane_index(self):
        return self.laneIndex

    def is_outof_map(self):
        if self.x >= 2800.0:
            return True
        else:
            return False

    def drive(self):
        if len(self.missionList) == 0:
            if self.vxCtl > SPEED_LIMIT_HIGH:
                self.vxCtl = SPEED_LIMIT_HIGH
            if self.vxCtl < SPEED_LIMIT_LOW:
                self.vxCtl = SPEED_LIMIT_LOW
            traci.vehicle.moveToXY(self.id, 'gneE0', 2, self.x + self.timeStep * self.vxCtl,
                                   self.y, 90, 2)
        else:
            temp_check_type = self.missionList[0]["c_type"]
            complete_flag = 0
            if temp_check_type == 1:
                if self.has_pre_change_to_lane_complete():
                    del self.missionList[0]
                    self.change_to_lane()
            elif temp_check_type == 2:
                if self.has_lane_change_complete():
                    del self.missionList[0]
                    self.post_change_to_lane()
            elif temp_check_type == 3:
                if self.has_post_change_to_lane():
                    complete_flag = 1
                    self.state = 0
                    del self.missionList[0]
            elif temp_check_type == 4:
                if self.has_lane_keep_step1():
                    del self.missionList[0]
                    self.lane_keep_step2()
            elif temp_check_type == 5:
                if self.has_lane_keep_step2():
                    complete_flag = 1
                    self.state = 0
                    del self.missionList[0]

            if complete_flag == 0:
                if self.missionList[0]['m_type'] == 1:
                    self.pre_change_to_lane()
                elif self.missionList[0]['m_type'] == 3:
                    self.post_change_to_lane()
                elif self.missionList[0]['m_type'] == 4:
                    self.lane_keep_step1()
                elif self.missionList[0]['m_type'] == 5:
                    self.lane_keep_step2()
                self.axCtl = self.missionList[0]['axCtl']
                self.ayCtl = self.missionList[0]['ayCtl']
                if self.missionList[0]['vxCtl'] is None:
                    self.vxCtl = self.vxCtl + self.timeStep * self.axCtl
                else:
                    self.vxCtl = self.missionList[0]['vxCtl'] + self.timeStep * self.axCtl
                if self.missionList[0]['vyCtl'] is None:
                    self.vyCtl = self.vyCtl + self.timeStep * self.ayCtl
                else:
                    self.vyCtl = self.missionList[0]['vyCtl'] + self.timeStep * self.ayCtl
            else:
                self.vxCtl = self.vx
                self.vyCtl = 0
                self.axCtl = 0
                self.ayCtl = 0

            if self.vxCtl > SPEED_LIMIT_HIGH:
                self.vxCtl = SPEED_LIMIT_HIGH
            if self.vxCtl < SPEED_LIMIT_LOW:
                self.vxCtl = SPEED_LIMIT_LOW
            traci.vehicle.moveToXY(self.id, 'gneE0', 2, self.x + self.timeStep * self.vxCtl,
                                   self.y + self.timeStep * self.vyCtl, self.angleCtl, 2)

    def lane_change_plan(self, gap_front_vehicle, gap_rear_vehicle):
        self.missionList.append(self._form_mission(1, 1, 0, 0, None, None))
        self.missionList.append(self._form_mission(2, 2, 0, 0, None, None))
        self.missionList.append(self._form_mission(3, 3, 0, 0, None, None))
        self.gapFrontVehicle = gap_front_vehicle
        self.gapRearVehicle = gap_rear_vehicle
        if gap_front_vehicle['virtual'] is False:
            traci.vehicle.subscribe(gap_front_vehicle['name'], (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_LANE_INDEX))
        if gap_rear_vehicle['virtual'] is False:
            traci.vehicle.subscribe(gap_rear_vehicle['name'], (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_LANE_INDEX))
        self.pre_change_to_lane()
        temp_goal_index = gap_front_vehicle['lane_index_relative']
        if temp_goal_index == 0:
            self.goalLaneIndex = self.laneIndex + 1
        elif temp_goal_index == 1:
            self.goalLaneIndex = self.laneIndex
        elif temp_goal_index == 2:
            self.goalLaneIndex = self.laneIndex - 1
        self.state = 2
        self.yBeforeLaneChange = self.y

    def pre_change_to_lane(self):
        mean_x = 0.5 * (self.gapRearVehicle['relative_position_x'] + self.gapFrontVehicle['relative_position_x'])
        temp_ax = 0.0
        if mean_x < -120:
            temp_ax = -4.0
        elif -120 <= mean_x < -50:
            temp_ax = -2.0
        elif -50 <= mean_x < 0:
            temp_ax = -1.0
        elif 0 <= mean_x < 50:
            temp_ax = 1.0
        elif 50 <= mean_x < 120:
            temp_ax = 2.0
        else:
            temp_ax = 4.0
        self.missionList[0]['axCtl'] = temp_ax

    def has_pre_change_to_lane_complete(self):
        gap_rear_vehicle_speed = self.gapRearVehicle['speed']
        gap_front_vehicle_speed = self.gapFrontVehicle['speed']
        rear_position_threshold = np.clip(2.0+8.0/33.0*gap_rear_vehicle_speed, 2.0, 10.0)
        front_position_threshold = np.clip(2.0+8.0/33.0*gap_front_vehicle_speed, 2.0, 10.0)
        if self.gapRearVehicle['position_x']+rear_position_threshold < self.x < \
                self.gapFrontVehicle['position_x']-front_position_threshold:
            return True
        else:
            return False

    def change_to_lane(self):
        if self.goalLaneIndex > self.laneIndex:
            self.missionList[0]['vyCtl'] = LANE_WIDTH / (1.0/60.0*abs(self.vx)+3)
        elif self.goalLaneIndex < self.laneIndex:
            self.missionList[0]['vyCtl'] = - LANE_WIDTH / (1.0/60.0*abs(self.vx)+3)
        else:
            self.missionList[0]['vyCtl'] = 0

    def has_lane_change_complete(self):
        if self.laneIndex == self.goalLaneIndex and abs(self.yLane) < 0.1:
            self.preLaneIndex = self.laneIndex
            self.vyCtl = 0
            return True
        else:
            return False

    def post_change_to_lane(self):
        if self.leadingVehicle['position_x'] < self.gapFrontVehicle['position_x']:
            temp_distance = self.leadingVehicle['position_x'] - self.x
            temp_relative_speed = self.leadingVehicle['speed'] - self.vx
        else:
            temp_distance = self.gapFrontVehicle['position_x']-self.x
            temp_relative_speed = self.gapFrontVehicle['speed']-self.vx
        temp_ax = 0.0
        if temp_relative_speed > 0:
            if temp_distance > 100:
                temp_ax = 4.0
            elif 50 <= temp_distance < 100:
                temp_ax = 2.0
            elif temp_distance >= 0:
                temp_ax = 1.0
            else:
                temp_ax = -8.0
        elif temp_relative_speed < 0:
            if temp_distance > 100:
                temp_ax = -2.0
            elif 50 <= temp_distance < 100:
                temp_ax = -4.0
            elif temp_distance >= 0:
                temp_ax = -8.0
            else:
                temp_ax = -8.0
        self.missionList[0]['axCtl'] = temp_ax

    def has_post_change_to_lane(self):
        if self.leadingVehicle['position_x'] < self.gapFrontVehicle['position_x']:
            if self.leadingVehicle['speed']-1.0 < self.vx <= self.leadingVehicle['speed']:
                # if self.gapFrontVehicle['virtual'] is False:
                #     traci.vehicle.unsubscribe(self.gapFrontVehicle['name'])
                return True
            else:
                return False
        else:
            if self.gapFrontVehicle['speed']-1.0 < self.vx <= self.gapFrontVehicle['speed']:
                # if self.gapFrontVehicle['virtual'] is False:
                #     traci.vehicle.unsubscribe(self.gapFrontVehicle['name'])
                return True
            else:
                return False

    def lane_keep_plan(self):
        self.missionList.append(self._form_mission(4, 4, 0, 0, None, None))
        self.missionList.append(self._form_mission(5, 5, 0, 0, None, None))
        self.lane_keep_step1()
        self.state = 1

    def lane_keep_step1(self):
        temp_distance = self.leadingVehicle['position_x'] - self.x
        temp_relative_speed = self.leadingVehicle['speed'] - self.vx
        temp_ax = 0.0
        safe_distance = np.max([4, self.leadingVehicle['speed']*3.0])
        if temp_relative_speed > 0:
            if temp_distance >= safe_distance * 1:
                temp_ax = np.clip((temp_distance-safe_distance)*safe_distance*temp_relative_speed/50000, 0, 4)
            else:
                temp_ax = np.clip((temp_distance-safe_distance)*safe_distance*temp_relative_speed/16.5, -8, 0)
        elif temp_relative_speed <= 0:
            if temp_distance >= safe_distance * 1.5:
                temp_ax = np.clip((temp_distance - safe_distance) * safe_distance * (-temp_relative_speed) / 50000, 0, 2)
            else:
                temp_ax = np.clip((temp_distance-safe_distance)*safe_distance*(-temp_relative_speed)/5.0, -8, 0)
        self.missionList[0]['axCtl'] = temp_ax

    def has_lane_keep_step1(self):
        temp_distance = self.leadingVehicle['position_x'] - self.x
        safe_distance = np.max([4, self.leadingVehicle['speed']*3.0])
        if self.leadingVehicle['virtual'] is False:
            if np.max([0, safe_distance-10.0]) < temp_distance < safe_distance:
                return True
            else:
                return False
        else:
            if self.vx > 25:
                return True
            else:
                return False

    def lane_keep_step2(self):
        temp_relative_speed = self.leadingVehicle['speed'] - self.vx
        temp_ax = 0.0
        if temp_relative_speed > 10.0:
            temp_ax = 4.0
        elif 4 <= temp_relative_speed < 10.0:
            temp_ax = 2.0
        elif 0 < temp_relative_speed < 4:
            temp_ax = 1.0
        elif temp_relative_speed < -10.0:
            temp_ax = -4.0
        elif -10 <= temp_relative_speed < -4.0:
            temp_ax = -2.0
        elif -4.0 <= temp_relative_speed < 0:
            temp_ax = -1.0
        self.missionList[0]['axCtl'] = temp_ax

    def has_lane_keep_step2(self):
        temp_relative_speed = self.leadingVehicle['speed'] - self.vx
        if -0.1 < temp_relative_speed < 0.1:
            return True
        else:
            return False

    def clear_mission(self):
        self.missionList = []

    def clear_gap_vehicle(self):
        self.gapFrontVehicle = None
        self.gapRearVehicle = None

    def check_collision(self):
        flag = 0
        if len(self.neighbourVehicleList) > 0:
            for item in self.neighbourVehicleList:
                if item['name'] != "ego":
                    temp_distance = math.pow(item['relative_position_x'], 2)+math.pow(item['relative_position_y'], 2)
                    if temp_distance < 2.0:
                        flag += 1
        if flag == 0:
            return False
        else:
            return True

    def check_can_change_lane(self, action_high):
        if self.goalLaneIndex > (self.nLane - 1) or self.goalLaneIndex < 0:
            return False
        else:
            return True

    def check_can_insert_into_gap(self):
        if self.gapFrontVehicle['relative_position_x'] - self.gapRearVehicle['relative_position_x'] < 10:
            return False
        else:
            return True

    def check_change_lane_successful(self):
        if abs(self.yBeforeLaneChange - self.y) >= 2.0:
            if self.gapRearVehicle['relative_position_x'] < 0 < self.gapFrontVehicle['relative_position_x']:
                return True
            else:
                return False
        else:
            return False

    def check_outof_road(self):
        if -self.nLane * LANE_WIDTH < self.y < 0 and self.outOfRoad is False:
            return False
        else:
            return True

    def is_safe(self):
        flag = True
        if len(self.neighbourVehicleList) > 0:
            for item in self.neighbourVehicleList:
                if item['name'] != "ego":
                    # print(item)
                    if -1.6 < item['relative_position_y'] < 1.6 and -0.5 < item['relative_position_x'] < 5.0:
                        flag = False
                        break
                    elif (item['relative_position_y'] < -1.6 or item['relative_position_y'] > 1.6) and self.vyCtl != 0 \
                            and math.pow(item['relative_position_x'], 2)+math.pow(item['relative_position_y'], 2) < 5.0:
                        flag = False
                        break
        return flag

