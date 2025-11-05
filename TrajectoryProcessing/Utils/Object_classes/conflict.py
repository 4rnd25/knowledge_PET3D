"""
Created on May 7 2025 08:24

@author: ISAC - pettirsch
"""


class Conflict:
    def __init__(self, timeStampVideo, timeStampVideoMicrosec, timeStamp, timeStampMicrosec, indicator, idVehicle1,
                 idVehicle2, vehicle_class_1, vehicle_class_2, vehicle_cluster_1, vehicle_cluster_2, posX, posY, value,
                 maneuverType=None, ruleFlag=None, LOFVeh1=None, LOFVeh2=None, maneuverType_manual=None,
                 ruleFlag_manual=None, checked_manual=None):
        self.timeStampVideo = timeStampVideo
        self.timeStampVideoMicrosec = timeStampVideoMicrosec
        self.timeStamp = timeStamp
        self.timeStampMicrosec = timeStampMicrosec
        self.indicator = indicator
        self.idVehicle1 = idVehicle1
        self.idVehicle2 = idVehicle2
        self.vehicle_class_1 = vehicle_class_1
        self.vehicle_class_2 = vehicle_class_2
        self.vehicle_cluster_1 = vehicle_cluster_1
        self.vehicle_cluster_2 = vehicle_cluster_2
        self.posX = posX
        self.posY = posY
        self.value = value
        self.maneuverType = maneuverType
        self.ruleFlag = ruleFlag
        self.LOFVeh1 = LOFVeh1
        self.LOFVeh2 = LOFVeh2
        self.maneuverType_manual = maneuverType_manual
        self.ruleFlag_manual = ruleFlag_manual
        self.checked_manual = checked_manual

    def get_label(self):
        # YEARMonthDAYTHHMMSS_class_is_class_id_indicator_value
        value_rounded = round(self.value, 2)
        return f"{self.timeStampVideo.strftime('%Y%m%dT%H%M%S')}_{self.maneuverType}_{self.vehicle_class_1}_{self.idVehicle1}_{self.vehicle_cluster_1}_{self.vehicle_class_2}_{self.idVehicle2}_{self.vehicle_cluster_2}_{self.indicator}_{value_rounded}"

    def get_conflict_time(self):
        timestamp = self.timeStampVideo
        return timestamp.replace(microsecond=self.timeStampVideoMicrosec)
