import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path
from action_msgs.msg import GoalStatus

from limo_interfaces.srv import GetSequencePlan, GenerateControlPlot
from limo_interfaces.action import Mission, FollowSequencePlan

from enum import Enum
import threading
import time


# =====================================================
# Mission State Definition
# =====================================================

class MissionState(Enum):
    WAITING = "WAITING"
    PLANNING = "PLANNING"
    MOVING = "MOVING"
    PAUSED = "PAUSED"


# =====================================================
# Mission Coordinator Node
# =====================================================

class MissionCoordinator(Node):

    def __init__(self):
        super().__init__('mission_coordinator')

        # ---------------- PARAMETERS ----------------
        self.declare_parameter('plan_service', '/plan_sequence_path')
        self.declare_parameter('follow_action', '/follow_sequence_plan')

        self.plan_service = self.get_parameter('plan_service').value
        self.follow_action = self.get_parameter('follow_action').value

        # ---------------- INTERNAL STATE ----------------
        self.state = MissionState.WAITING
        self.abort_requested = False
        self.follow_handle = None
        self.follow_goal_future = None
        self.state_switch = False

        # Pause flag (used by external topic)
        self.pause_requested = False

        # Execute synchronization future
        self.done_event = None
        self.result = Mission.Result()

        # Prevent double finalization (IMPORTANT)
        self._finalized = False

        # ---------------- CALLBACK GROUP ----------------
        self.cb_group = ReentrantCallbackGroup()

        # ---------------- ACTION CLIENTS ----------------
        self.plan_client = self.create_client(
            GetSequencePlan,
            self.plan_service,
            callback_group=self.cb_group
        )

        self.follow_client = ActionClient(
            self,
            FollowSequencePlan,
            self.follow_action,
            callback_group=self.cb_group
        )

        self.plot_client = self.create_client(
            GenerateControlPlot,
            '/mission/generate_control_plot',
            callback_group=self.cb_group
        )
        
        self.declare_parameter('plot_output_dir', '')
        self.plot_output_dir = self.get_parameter('plot_output_dir').value


        # ---------------- TOPICS ----------------
        self.pause_sub = self.create_subscription(
            Bool,
            '/mission/pause',
            self._on_pause,
            10
        )

        self.enable_pub = self.create_publisher(Bool, '/mission/enable', 10)
        self.goals_pub = self.create_publisher(PoseArray, '/limo/mission/goals', 10)
        self.ordered_goals_pub = self.create_publisher(PoseArray, '/limo/mission/goals_astar', 10)
        self.paths_pub = self.create_publisher(Path, '/limo/mission/paths', 10)

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

    def _set_state(self, new_state: MissionState):
        """Update internal mission state and trigger feedback update."""
        self.get_logger().info(f"[STATE] {self.state.value} → {new_state.value}")
        self.state = new_state
        self.state_switch = True


# =====================================================
# ACTION CALLBACKS
# =====================================================

    def _accept_goal(self, goal_request):
        self.get_logger().info(
            f"Received mission with {len(goal_request.goals)} goals"
        )
        return rclpy.action.GoalResponse.ACCEPT

    def _cancel_goal(self, goal_handle):
        self.get_logger().warn("Mission cancel requested")
        self.abort_requested = True

        if self.follow_handle:
            self.follow_handle.cancel_goal_async()

        return rclpy.action.CancelResponse.ACCEPT


# =====================================================
# EXECUTION LOOP
# =====================================================

    def _execute(self, goal_handle):
        """Main mission execution entry point."""

        self.abort_requested = False
        self.pause_requested = False
        self._finalized = False

        self.done_event = threading.Event()
        self.result = Mission.Result()

        self._set_state(MissionState.PLANNING)

        # Publish initial goals
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'odom'
        pa.poses = [g.pose for g in goal_handle.request.goals]
        self.goals_pub.publish(pa)

        # Call planner service
        if not self.plan_client.wait_for_service(timeout_sec=5.0):
            self._finalize(goal_handle, "ABORT", "Planner unavailable")
            return self.result

        req = GetSequencePlan.Request()
        req.goals = list(goal_handle.request.goals)

        future = self.plan_client.call_async(req)
        future.add_done_callback(
            lambda f: self._on_plan_done(f, goal_handle)
        )

        # ---------------- EXECUTION WAIT LOOP ----------------
        while not self.done_event.is_set():

            # Pause behavior: only affects state + robot enable
            if self.pause_requested:
                if self.state is not MissionState.PAUSED:
                    self._set_state(MissionState.PAUSED)
            else:
                if self.state == MissionState.PAUSED:
                    self._set_state(MissionState.MOVING)

            # Publish continuous feedback
            if self.state_switch:
                fb = Mission.Feedback()
                fb.state = self.state.value
                goal_handle.publish_feedback(fb)

                self.state_switch = False

            time.sleep(0.1)

        return self.result


# =====================================================
# PLANNER CALLBACK
# =====================================================

    def _on_plan_done(self, future, goal_handle):

        if self.abort_requested:
            self._finalize(goal_handle, "ABORT", "Aborted during planning")
            return

        try:
            plan = future.result()
        except Exception as e:
            self._finalize(goal_handle, "ABORT", f"Planner error: {e}")
            return

        # Publish ordered goals
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'odom'
        pa.poses = [g.pose for g in plan.ordered_goals]
        self.ordered_goals_pub.publish(pa)

        # Publish paths
        for path in plan.paths:
            self.paths_pub.publish(path)

        self._set_state(MissionState.MOVING)
        self._start_follow(goal_handle, plan.ordered_goals, plan.paths)


# =====================================================
# FOLLOW ACTION
# =====================================================

    def _start_follow(self, goal_handle, ordered_goals, paths):

        if not self.follow_client.wait_for_server(timeout_sec=5.0):
            self._finalize(goal_handle, "ABORT", "Follow server unavailable")
            return

        goal = FollowSequencePlan.Goal()
        goal.ordered_goals = ordered_goals
        goal.paths = paths

        future = self.follow_client.send_goal_async(
            goal,
            feedback_callback=lambda fb: self._on_follow_feedback(fb, goal_handle)
        )
        self.follow_goal_future = future
        future.add_done_callback(
            lambda f: self._on_follow_response(f, goal_handle)
        )

    def _on_follow_response(self, future, goal_handle):

        handle = future.result()
        self.follow_goal_future = None

        if not handle.accepted:
            self._finalize(goal_handle, "ABORT", "Follow rejected")
            return

        self.follow_handle = handle

        if self.abort_requested:
            self.get_logger().warn("Abort già richiesto → cancel follow immediato")
            handle.cancel_goal_async()
            # non aspettare il result, _finalize arriverà da _on_follow_done
            # oppure forza qui:
            self._finalize(goal_handle, "ABORT", "Aborted before follow started")
            return

        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._on_follow_done(f, goal_handle)
        )

    
    def _on_follow_feedback(self, feedback_msg, goal_handle):
        inner = feedback_msg.feedback
        fb = Mission.Feedback()
        fb.state = self.state.value
        fb.current_goal = inner.next_goal_pose
        goal_handle.publish_feedback(fb)

        self.get_logger().info(
            f"[FOLLOW] goal={inner.current_goal_index} | "
            f"d_err={inner.distance_error:+.3f}m  a_err={inner.angular_error:+.4f}rad | "
            f"v={inner.linear_velocity:+.3f}m/s  ω={inner.angular_velocity:+.4f}rad/s | "
            f"d_goal={inner.distance_to_next_goal:.3f}m"
        )


# =====================================================
# FOLLOW RESULT
# =====================================================

    def _on_follow_done(self, future, goal_handle):
 
        if self._finalized:
            return
    
        result = future.result()
        status = result.status
    
        if self.abort_requested:
            self._finalize(goal_handle, "ABORT", "Follow canceled")
            return
    
        if status != GoalStatus.STATUS_SUCCEEDED:
            self._finalize(goal_handle, "ABORT", f"Follow failed ({status})")
            return
    
        # ── genera i plot prima di finalizzare ──
        inner = result.result  # FollowSequencePlan.Result
    
        if self.plot_client.wait_for_service(timeout_sec=2.0):
            req = GenerateControlPlot.Request()
            req.distance_errors   = list(inner.distance_errors)
            req.angular_errors    = list(inner.angular_errors)
            req.linear_velocities = list(inner.linear_velocities)
            req.angular_velocities = list(inner.angular_velocities)
            req.goal_reach_times  = list(inner.goal_reach_times)
            req.n_goals           = len(inner.goal_reach_times)
            req.output_dir        = self.plot_output_dir
    
            plot_future = self.plot_client.call_async(req)
            plot_future.add_done_callback(self._on_plot_done)
        else:
            self.get_logger().warn("ControlPlotService non disponibile, skip plot")
    
        self._finalize(goal_handle, "SUCCESS", "Mission completed")
 
 
    def _on_plot_done(self, future):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f"[PLOT] {res.message}")
                for p in res.output_paths:
                    self.get_logger().info(f"  → {p}")
            else:
                self.get_logger().warn(f"[PLOT] failed: {res.message}")
        except Exception as e:
            self.get_logger().error(f"[PLOT] exception: {e}")


# =====================================================
# PAUSE HANDLING
# =====================================================

    def _on_pause(self, msg: Bool, *args):
        self.pause_requested = msg.data

        if msg.data:
            self.get_logger().info("Pause requested")
        else:
            self.get_logger().info("Resume requested")

        self.enable_pub.publish(Bool(data=not msg.data))
        

# =====================================================
# FINALIZE (SINGLE EXIT POINT)
# =====================================================

    def _finalize(self, goal_handle, mode: str, message: str = ""):

        if self._finalized:
            return

        self._finalized = True
        self.get_logger().warn(f"[FINALIZE] {mode} - {message}")

        # Stop follow action if active
        if self.follow_handle:
            try:
                self.follow_handle.cancel_goal_async()
            except Exception:
                pass
            self.follow_handle = None

        # Set result
        self.result.success = (mode == "SUCCESS")

        # Update ROS action state
        if mode == "SUCCESS":
            goal_handle.succeed()
        elif mode == "ABORT":
            goal_handle.abort()
        elif mode == "CANCEL":
            goal_handle.canceled()

        # Unblock execute loop
        if self.done_event and not self.done_event.is_set():
            self.done_event.set()

        self._set_state(MissionState.WAITING)


# =====================================================
# MAIN
# =====================================================

def main(args=None):
    rclpy.init(args=args)
    node = MissionCoordinator()
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