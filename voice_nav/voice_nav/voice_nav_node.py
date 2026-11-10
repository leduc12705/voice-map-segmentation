#!/usr/bin/env python3
"""Nhan /voice_text -> phan tich -> gui lan luot NavigateToPose toi Nav2."""
import math
from collections import deque

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from voice_nav.command_parser import CommandParser


class VoiceNavNode(Node):
    def __init__(self):
        super().__init__("voice_nav_node")
        self.declare_parameter("zones_file", "zones.yaml")
        self.declare_parameter("near_tolerance", 0.6)  # m: da o gan diem dau thi bo qua
        with open(self.get_parameter("zones_file").value, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.frame = cfg.get("map_frame", "map")
        self.zones = cfg["zones"]
        self.parser = CommandParser(self.zones)

        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.status_pub = self.create_publisher(String, "voice_nav/status", 10)
        self.create_subscription(String, "voice_text", self.on_text, 10)

        self.queue = deque()
        self.goal_handle = None
        self.gen = 0  # tang moi khi co lenh moi de bo qua callback cu
        self.say(f"San sang. Cac khu vuc: {', '.join(z['name'] for z in self.zones)}")

    def say(self, msg):
        self.get_logger().info(msg)
        self.status_pub.publish(String(data=msg))

    # ---------- xu ly lenh ----------
    def on_text(self, msg):
        res = self.parser.parse(msg.data)
        if res.kind == "unknown_zone":
            names = ", ".join(z["name"] for z in self.zones)
            self.say(f"Khong co khu vuc '{', '.join(res.unknown)}'. Cac khu vuc hien co: {names}")
        elif res.kind == "unrecognized":
            self.say(f"Khong hieu lenh: '{msg.data}'")
        elif res.kind == "cancel":
            self.cancel_all()
            self.say("Da huy, robot dung lai.")
        else:
            route = self.trim_route(res.route)
            self.say("Lo trinh: " + " -> ".join(z["name"] for z in route))
            self.start_route(route)

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(self.frame, "base_link", rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    def trim_route(self, route):
        """Neu robot da o gan diem dau tien thi khong can quay lai diem do."""
        xy = self.robot_xy()
        if xy and len(route) > 1:
            g = route[0]["goal"]
            if math.hypot(g["x"] - xy[0], g["y"] - xy[1]) < self.get_parameter("near_tolerance").value:
                return route[1:]
        return route

    # ---------- dieu huong ----------
    def cancel_all(self):
        self.gen += 1
        self.queue.clear()
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None

    def start_route(self, route):
        self.cancel_all()
        if not self.client.wait_for_server(timeout_sec=3.0):
            self.say("Nav2 (navigate_to_pose) chua san sang.")
            return
        self.queue = deque(route)
        self.send_next(self.gen)

    def send_next(self, gen):
        if gen != self.gen:
            return
        if not self.queue:
            self.say("Hoan thanh lo trinh.")
            return
        zone = self.queue.popleft()
        g = zone["goal"]
        pose = PoseStamped()
        pose.header.frame_id = self.frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = float(g["x"]), float(g["y"])
        yaw = float(g.get("yaw", 0.0))
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.say(f"Dang di den {zone['name']} ({g['x']:.2f}, {g['y']:.2f})")
        fut = self.client.send_goal_async(goal)
        fut.add_done_callback(lambda f: self.on_accept(f, zone, gen))

    def on_accept(self, fut, zone, gen):
        if gen != self.gen:
            return
        h = fut.result()
        if not h.accepted:
            self.say(f"Nav2 tu choi muc tieu {zone['name']}.")
            self.queue.clear()
            return
        self.goal_handle = h
        h.get_result_async().add_done_callback(lambda f: self.on_result(f, zone, gen))

    def on_result(self, fut, zone, gen):
        if gen != self.gen:
            return
        status = fut.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.say(f"Da den {zone['name']}.")
            self.send_next(gen)
        else:
            self.say(f"Khong den duoc {zone['name']} (status={status}). Huy phan con lai.")
            self.queue.clear()


def main():
    rclpy.init()
    node = VoiceNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
