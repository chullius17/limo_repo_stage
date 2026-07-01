#!/usr/bin/env python3

import math
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped, PoseArray
from action_msgs.msg import GoalStatus

from limo_interfaces.action import Mission
from limo_interfaces.srv import MissionCommand


# =====================================================
# CLIENT
# =====================================================

class MissionClient(Node):

    def __init__(self):
        super().__init__('mission_client')

        self.cb_group = ReentrantCallbackGroup()

        # GUI SERVER
        self._service = self.create_service(
            MissionCommand,
            '/mission_cmd',
            self._cmd_callback
        )

        # ACTION CLIENT
        self._client = ActionClient(
            self,
            Mission,
            '/mission',
            callback_group=self.cb_group
        )

        # PAUSE TOPIC
        self._pause_pub = self.create_publisher(
            Bool,
            '/mission/pause',
            10
        )

        # VISUALIZATION TOPIC
        self._queued_goals_pub = self.create_publisher(
            PoseArray,
            '/mission/queued_goals',
            10
        )

        # STATE PUBLISHER
        self.state_gui_pub = self.create_publisher(
            String,
            '/mission/state',
            10
        )

        # GOAL PUBLISHER
        self.goal_gui_pub = self.create_publisher(
            String,
            '/mission/goal',
            10
        )

        # STATE
        self._goals: List[PoseStamped] = []
        self._active_goal = None

        # SUPPORT VARIABLES
        self.last_state = None
        self.last_goal = None

        self.get_logger().info("MissionClient ready")

    # =====================================================
    # GUI COMMAND RECEPTION
    # =====================================================

    def _cmd_callback(self, request, response):
        cmd = request.command.strip().split()

        if not cmd:
            response.success = False
            response.message = "empty command"
            return response

        action = cmd[0].lower()

        try:

            # ---------------- ADD ----------------
            if action == "add":
                if len(cmd) < 3:
                    raise ValueError("usage: add x y yaw")

                x = float(cmd[1])
                y = float(cmd[2])
                yaw = float(cmd[3]) if len(cmd) > 3 else 0.0

                self.add(x, y, yaw)

                response.success = True
                response.message = "goal added"
                return response

            # ---------------- SEND ----------------
            elif action == "send":
                self.send()
                response.success = True
                response.message = "mission sent"
                return response

            # ---------------- ABORT ----------------
            elif action == "abort":
                self.abort()
                response.success = True
                response.message = "mission aborted"
                return response

            # ---------------- PAUSE ----------------
            elif action == "pause":
                self.pause()
                response.success = True
                response.message = "paused"
                return response

            # ---------------- RESUME ----------------
            elif action == "resume":
                self.resume()
                response.success = True
                response.message = "resumed"
                return response

            else:
                response.success = False
                response.message = f"unknown command: {action}"
                return response

        except Exception as e:
            response.success = False
            response.message = str(e)
            return response

    # =====================================================
    # VISUALIZATION HELPER
    # =====================================================

    def _publish_queued(self):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'map'
        pa.poses = [ps.pose for ps in self._goals]
        self._queued_goals_pub.publish(pa)

    # =====================================================
    # GOALS
    # =====================================================

    def add(self, x: float, y: float, yaw_deg: float = 0.0):
        ps = PoseStamped()
        ps.header.frame_id = "map"

        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0

        yaw = math.radians(yaw_deg)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        ps.pose.orientation.z = math.sin(yaw / 2.0)

        self._goals.append(ps)
        self._publish_queued()

        print(f"[ADD] ({x:.2f}, {y:.2f}, {yaw_deg:.1f} deg) "
              f"tot={len(self._goals)}")

    # =====================================================
    # ACTION CONTROL
    # =====================================================

    def send(self):
        """abort + restart mission"""

        if not self._goals:
            print("[WARN] nessun goal da inviare")
            return

        self.abort()

        if not self._client.wait_for_server(timeout_sec=5.0):
            print("[ERROR] Mission server non disponibile")
            return

        goal_msg = Mission.Goal()
        goal_msg.goals = list(self._goals)

        print(f"[SEND] invio missione ({len(self._goals)} goal)")

        future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_cb
        )

        future.add_done_callback(self._goal_response_cb)
        self._goals.clear()
        self._publish_queued()

    def abort(self):
        """cancel missione corrente"""
        if self._active_goal is None:
            print("[WARN] nessuna missione attiva")
            return

        print("[ABORT] richiesta cancellazione")
        self._active_goal.cancel_goal_async()
        self._active_goal = None

    # =====================================================
    # CALLBACK ACTION
    # =====================================================

    def _goal_response_cb(self, future):
        handle = future.result()

        if not handle.accepted:
            print("[ERROR] missione rifiutata")
            return

        self._active_goal = handle
        print("[INFO] missione accettata")

        result_future = handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        res = future.result()
        status = res.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            print("[OK] mission completata")
        elif status == GoalStatus.STATUS_CANCELED:
            print("[INFO] mission cancellata")
        else:
            print(f"[WARN] status={status}")

        self._active_goal = None

    def _feedback_cb(self, msg):
        fb = msg.feedback
        pose = fb.current_goal

        x = round(pose.pose.position.x, 3)
        y = round(pose.pose.position.y, 3)

        q = pose.pose.orientation
        yaw = round(
            math.degrees(
                math.atan2(
                    2.0 * (q.w * q.z),
                    1.0 - 2.0 * (q.z * q.z)
                )
            ),
            1
        )

        current_state = fb.state
        current_goal = (x, y, yaw)

        if (
            current_state == self.last_state
            and current_goal == self.last_goal
        ):
            return

        # ───────── STATE TOPIC ─────────
        state_msg = String()
        state_msg.data = str(fb.state)
        self.state_gui_pub.publish(state_msg)

        # ───────── GOAL TOPIC ─────────
        goal_msg = String()
        goal_msg.data = f"{x:.2f},{y:.2f},{yaw:.1f}"
        self.goal_gui_pub.publish(goal_msg)

        self.last_state = current_state
        self.last_goal = current_goal

    # =====================================================
    # PAUSE / RESUME
    # =====================================================

    def pause(self):
        self._pause_pub.publish(Bool(data=True))
        print("[PAUSE] attivata")

    def resume(self):
        self._pause_pub.publish(Bool(data=False))
        print("[RESUME] attivata")


# =====================================================
# MAIN
# =====================================================

def main(args=None):
    rclpy.init(args=args)

    node = MissionClient()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()