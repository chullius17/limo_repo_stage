#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
from action_msgs.msg import GoalStatus

from limo_interfaces.srv import GetSequencePlan
from limo_interfaces.action import Mission, FollowSequencePlan

from enum import Enum
import time


# =====================================================
# MISSION STATE
# =====================================================

class MissionState(Enum):
    WAITING = "WAITING"
    PLANNING = "PLANNING"
    MOVING = "MOVING"
    PAUSED = "PAUSED"


# =====================================================
# MAIN NODE
# =====================================================

class MissionCoordinator(Node):

    def __init__(self):
        super().__init__('mission_coordinator')

        # ---------------- PARAMETERS ----------------
        self.declare_parameter('plan_service', '/plan_sequence_path')
        self.declare_parameter('follow_action', '/follow_sequence_plan')

        self.plan_service = self.get_parameter('plan_service').value
        self.follow_action = self.get_parameter('follow_action').value

        # ---------------- STATE ----------------
        self.state = MissionState.WAITING
        self.abort_requested = False
        self.follow_handle = None

        # done sync (execute_callback waits for this)
        self.done_event = rclpy.task.Future()
        self.result = Mission.Result()

        # ---------------- CALLBACK GROUP ----------------
        self.cb_group = ReentrantCallbackGroup()

        # ---------------- CLIENTS ----------------
        self.plan_client = self.create_client(
            GetSequencePlan,
            self.plan_service,
            callback_group=self.cb_group
        )

        self.reset_client = self.create_client(
            Trigger, 
            '/mission/reset_viz', 
            callback_group=self.cb_group
            )

        self.follow_client = ActionClient(
            self,
            FollowSequencePlan,
            self.follow_action,
            callback_group=self.cb_group
        )

        # ---------------- COMMUNICATION ----------------
        self.enable_pub = self.create_publisher(Bool, '/mission/enable', 10)

        self.pause_sub = self.create_subscription(
            Bool,
            '/mission/pause',
            self._on_pause,
            10
        )

        self.ordered_goals_pub = self.create_publisher(PoseArray, '/limo/mission/goals_astar', 10)
        self.goals_pub = self.create_publisher(PoseArray, 'limo/mission/goals', 10)
        self.paths_pub = self.create_publisher(Path, 'limo/mission/paths', 10)

        # ---------------- ACTION SERVER ----------------
        self.action_server = ActionServer(
            self,
            Mission,
            '/mission',
            execute_callback=self._execute,
            goal_callback=self._accept_goal,
            cancel_callback=self._cancel_goal,
            callback_group=self.cb_group
        )

        self.get_logger().info("MissionCoordinator READY")

    # =====================================================
    # STATE MANAGEMENT
    # =====================================================

    def _set_state(self, new_state: MissionState, goal_handle=None):
        self.get_logger().info(f"[STATE] {self.state.value} → {new_state.value}")
        self.state = new_state

        if goal_handle:
            fb = Mission.Feedback()
            fb.state = self.state.value
            goal_handle.publish_feedback(fb)

    def _log_event(self, msg: str):
        self.get_logger().warn(f"[MISSION] {msg}")

    # =====================================================
    # ACTION CALLBACKS
    # =====================================================

    def _accept_goal(self, goal_request):
        self.get_logger().info(f"Received mission with {len(goal_request.goals)} goals")
        return rclpy.action.GoalResponse.ACCEPT

    def _cancel_goal(self, goal_handle):
        self.get_logger().warn("Mission cancel requested")
        self.abort_requested = True

        self._call_reset()

        if self.follow_handle:
            self.follow_handle.cancel_goal_async()

        return rclpy.action.CancelResponse.ACCEPT

    # =====================================================
    # EXECUTION ENTRY POINT
    # =====================================================

    def _execute(self, goal_handle):
        self.abort_requested = False
        self.done_event      = rclpy.task.Future()
        self.result          = Mission.Result()

        self._set_state(MissionState.PLANNING, goal_handle)

        pa = PoseArray()
        pa.header.stamp    = self.get_clock().now().to_msg()
        pa.header.frame_id = 'odom'
        pa.poses = [g.pose for g in goal_handle.request.goals]
        self.goals_pub.publish(pa)

        if not self.plan_client.wait_for_service(timeout_sec=5.0):
            self._finalize(goal_handle, succeeded=False)
            return self.result

        req       = GetSequencePlan.Request()
        req.goals = list(goal_handle.request.goals)

        future = self.plan_client.call_async(req)
        future.add_done_callback(lambda f: self._on_plan_done(f, goal_handle))

        # ── BLOCCA QUI finché _finalize non setta done_event ──────────────
        while not self.done_event.done():
            time.sleep(0.05)

        return self.result
    
    # =====================================================
    # PLANNING RESULT
    # =====================================================

    def _on_plan_done(self, future, goal_handle):
        if self.abort_requested:
            self._abort(goal_handle)
            return

        plan = future.result()

        # pubblica goal ordinati
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'odom'
        pa.poses = [g.pose for g in plan.ordered_goals]
        self.ordered_goals_pub.publish(pa)

        # pubblica ogni segmento di path
        for path in plan.paths:
            self.paths_pub.publish(path)

        self._set_state(MissionState.MOVING, goal_handle)
        self._start_follow(goal_handle, plan.ordered_goals, plan.paths)

    # =====================================================
    # FOLLOW ACTION
    # =====================================================

    def _start_follow(self, goal_handle, ordered_goals, paths):

        if not self.follow_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Follow action server unavailable")
            self._abort(goal_handle)
            return

        goal = FollowSequencePlan.Goal()
        goal.ordered_goals = ordered_goals
        goal.paths = paths

        future = self.follow_client.send_goal_async(goal)
        future.add_done_callback(lambda f: self._on_follow_response(f, goal_handle))

    def _on_follow_response(self, future, goal_handle):

        handle = future.result()

        if not handle.accepted:
            self.get_logger().error("Follow goal rejected")
            self._abort(goal_handle)
            return

        self.follow_handle = handle

        result_future = handle.get_result_async()
        result_future.add_done_callback(lambda f: self._on_follow_done(f, goal_handle))

    def _on_follow_done(self, future, goal_handle):
        result = future.result()
        follow_status = result.status

        if self.abort_requested:
            # Il cancel del follow è andato a buon fine, ora abortisco il mission goal
            self._log_event("ABORTED (follow canceled)")
            self.enable_pub.publish(Bool(data=False))
            self._call_reset()
            goal_handle.abort()
            self._set_state(MissionState.WAITING)
            if not self.done_event.done():
                self.done_event.set_result(self.result)
            return

        if follow_status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(f"Follow ended with status={follow_status}, aborting mission")
            self._abort(goal_handle)
            return

        self._log_event("DONE")
        goal_handle.succeed()
        self._set_state(MissionState.WAITING, goal_handle)
        if not self.done_event.done():
            self.done_event.set_result(self.result)

    # =====================================================
    # PAUSE HANDLING
    # =====================================================

    def _on_pause(self, msg: Bool, goal_handle):
        if self.state not in (MissionState.MOVING, MissionState.PAUSED):
            return

        if msg.data:
            self._set_state(MissionState.PAUSED, goal_handle)
        else:
            self._set_state(MissionState.MOVING, goal_handle)

        # pause=True  → enable=False
        # pause=False → enable=True
        self.enable_pub.publish(Bool(data=not msg.data))

    # =====================================================
    # ABORT LOGIC (UNIFIED)
    # =====================================================

    def _abort(self, goal_handle):
        self.abort_requested = True

        if self.follow_handle:
            self.follow_handle.cancel_goal_async()
            self.follow_handle = None

        self.enable_pub.publish(Bool(data=False))
        self._call_reset()
        self._log_event("ABORTED")

        # Controlla lo stato prima di chiamare abort/canceled
        status = goal_handle.status
        if status == GoalStatus.STATUS_EXECUTING:
            goal_handle.abort()
        elif status == GoalStatus.STATUS_CANCELING:
            goal_handle.canceled()   # rispetta il protocollo ROS2

        self._set_state(MissionState.WAITING)
        if not self.done_event.done():
            self.done_event.set_result(self.result)

    def _call_reset(self):
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("reset_viz service not available, skipping.")
            return
        future = self.reset_client.call_async(Trigger.Request())
        future.add_done_callback(lambda f: self.get_logger().info("Visualizer reset."))

    # =====================================================
    # SHUTDOWN
    # =====================================================

    def destroy_node(self):
        self.get_logger().info("Shutting down MissionCoordinator...")

        self.abort_requested = True

        if self.follow_handle:
            try:
                self.follow_handle.cancel_goal_async()
            except Exception:
                pass

        try:
            self.action_server.destroy()
        except Exception:
            pass

        try:
            self.follow_client.destroy()
        except Exception:
            pass

        super().destroy_node()

# =====================================================
# MAIN
# =====================================================

def main(args=None):
    rclpy.init(args=args)

    node = MissionCoordinator()

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
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