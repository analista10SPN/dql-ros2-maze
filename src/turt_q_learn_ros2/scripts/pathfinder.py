#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from matplotlib import pyplot
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import os
import numpy as np
import math
import bisect

mapPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/map/map.pgm"

class Cell:
    def __init__(self, f=0, g=0, h=0, walkable=False, y=0, x=0):
        self.G, self.F, self.H = g, f, h
        self.y, self.x = y, x
        self.parent = None
        self.walkable = walkable

    def __lt__(self, other):
        return self.F < other.F

    def __repr__(self):
        return 'F({})'.format(self.F)

class PathFinder(Node):
    def __init__(self):
        super().__init__('pathfinder')
        self.collRadius = .3
        with open(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/map/map.pgm", 'rb') as f:
            self.map = readPgm(f)
        
        self.originOffset = (-10, -10)
        self.resolution = .05
        self.path = []
        self.pathColour = 128

        # Publisher
        self.action_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Start pathfinding process
        self.aStar([168, 168], [200, 210])
        self.drivePath()

    def genPathMap(self):
        pixRadius = int(round(self.collRadius / self.resolution))
        self.pathMap = np.array(self.map.tolist(), dtype=np.int16)
        wallList = []
        for y, row in enumerate(self.map):
            for x, cell in enumerate(row):
                if cell == 205:
                    self.pathMap[y][x] = -1
                elif cell == 0:
                    for y2 in range(y - pixRadius, y + pixRadius):
                        for x2 in range(x - pixRadius, x + pixRadius):
                            if x2 >= 0 and y2 >= 0:
                                if self.map[y2][x2] != 0 and math.sqrt((x2 - x) ** 2 + (y2 - y) ** 2) < pixRadius:
                                    self.pathMap[y2][x2] = -1
                    wallList.append([y, x])
                elif cell == 254:
                    self.pathMap[y][x] = 1

        for coord in wallList:
            self.pathMap[coord[0]][coord[1]] = -1

    def genAStarMap(self, target):
        self.aMap = []
        for y, row in enumerate(self.map):
            newRow = []
            for x in range(len(row)):
                h = (min(abs(x - target[1]), abs(y - target[0])) * 14) + (abs(abs(x - target[1]) - abs(y - target[0])) * 10)
                newRow.append(Cell(h, 0, h, self.map[y][x] == 254, y, x))
            self.aMap.append(newRow)

    def aStar(self, goal, start):
        self.genAStarMap(goal)
        closedList, openList = [], [self.aMap[start[0]][start[1]]]

        while openList:
            current = openList.pop(0)
            closedList.append(current)
            self.get_logger().info(f"Checking {current.y}, {current.x}")
            if current.y == goal[0] and current.x == goal[1]:
                self.path = getPath(current)
                self.pathToWaypoints()
                return

            for y in range(-1, 2):
                for x in range(-1, 2):
                    if (y + current.y) >= 0 and (x + current.x) >= 0:
                        checkCell = self.aMap[y + current.y][x + current.x]
                        if not checkCell.walkable or checkCell in closedList:
                            continue
                        else:
                            newG = current.G + (10 if abs(y) - abs(x) else 14)
                            if checkCell not in openList:
                                checkCell.parent = current
                                checkCell.G = newG
                                checkCell.F = checkCell.G + checkCell.H
                                bisect.insort_left(openList, checkCell)
                            elif newG < checkCell.G:
                                checkCell.G = newG
                                checkCell.F = checkCell.G + checkCell.H
                                checkCell.parent = current
                                openList.sort(key=lambda el: el.F)

        self.get_logger().error("NO PATH")

    def drivePath(self):
        for wayPoint in self.path:
            self.turnToPoint(wayPoint)
            self.driveToPoint(wayPoint)

    def turnToPoint(self, point):
        while True:
            heading = self.getHeading(point, False)
            self.get_logger().info(f"Heading: {heading}")
            self.action_publisher.publish(genTwist(0, heading))
            if abs(heading) < .1:
                return

    def driveToPoint(self, point):
        while True:
            heading, distance = self.getHeading(point, True)
            self.get_logger().info(f"Heading: {heading}")
            self.action_publisher.publish(genTwist(.15, heading))
            if distance < .1:
                self.action_publisher.publish(genTwist(0, 0))
                return

    def getHeading(self, point, isDriving):
        odomData = None
        while odomData is None:
            try:
                odomData = self.wait_for_message('odom', Odometry, timeout=5.0)
            except Exception as e:
                pass

        modelX = odomData.pose.pose.position.x
        modelY = odomData.pose.pose.position.y

        goalAngle = math.atan2(point[0] - modelY, point[1] - modelX)
        modelAngle = odomData.pose.pose.orientation

        yaw = math.atan2(+2.0 * (modelAngle.w * modelAngle.z + modelAngle.x * modelAngle.y),
                         1.0 - 2.0 * (modelAngle.y * modelAngle.y + modelAngle.z * modelAngle.z))

        if isDriving:
            return goalAngle - yaw, math.hypot(point[1] - modelX, point[0] - modelY)
        else:
            return goalAngle - yaw

def getPath(node):
    nodeList = []
    while node.parent:
        nodeList.append([node.y, node.x])
        node = node.parent
    nodeList.append([node.y, node.x])
    return list(reversed(nodeList))

def readPgm(pgmf):
    assert pgmf.readline() == b'P5\n'
    _ = pgmf.readline()
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255
    return np.fromfile(pgmf, dtype=np.uint8).reshape((height, width))

def genTwist(speed, z):
    retTwist = Twist()
    retTwist.linear.x = speed
    retTwist.angular.z = z
    return retTwist

def main(args=None):
    rclpy.init(args=args)
    pathfinder = PathFinder()
    rclpy.spin(pathfinder)

    pathfinder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
