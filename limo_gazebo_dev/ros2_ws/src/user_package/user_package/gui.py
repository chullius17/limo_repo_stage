#!/usr/bin/env python3

import sys
import threading
from sensor_msgs.msg import Image
from std_msgs.msg import String
from limo_interfaces.srv import MissionCommand

import rclpy
from rclpy.node import Node

# Correct and explicit configuration for PyQt5
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QEvent
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# TODO: fix pause logic in order to implement pause toggle switch

class MissionGuiNode(Node):
    """ROS 2 Node that handles data communication for the GUI."""

    def __init__(self, gui_signals):
        super().__init__('mission_gui_node')
        self.signals = gui_signals

        # SERVICE CLIENT
        self.cli = self.create_client(MissionCommand, '/mission_cmd')
        
        # SUBSCRIPTIONS
        self.state_sub = self.create_subscription(
            String,
            '/mission/state',
            self._state_callback,
            10
        )
        self.goal_sub = self.create_subscription(
            String,
            '/mission/goal',
            self._callback_goal,
            10
        )
        self.image_sub = self.create_subscription(
            Image,
            '/limo/mission_visualizer/image',
            self._image_callback,
            10
        )

    def send_cmd_async(self, command_str: str):
        """Sends a service request asynchronously without blocking the GUI."""
        if not self.cli.service_is_ready():
            self.signals.append_msg.emit("[GUI ERROR] Service /mission_cmd not available")
            return

        req = MissionCommand.Request()
        req.command = command_str
        
        future = self.cli.call_async(req)
        future.add_done_callback(self._service_response_callback)

    def _service_response_callback(self, future):
        try:
            # Code comments in English as requested
            response = future.result()
            prefix = "[SUCCESS]" if response.success else "[FAILED]"
            self.signals.append_msg.emit(f"{prefix} {response.message}")
        except Exception as e:
            self.signals.append_msg.emit(f"[SERVICE ERROR] {str(e)}")

    def _state_callback(self, msg: String):
        self.signals.update_state.emit(msg.data)

    def _callback_goal(self, msg: String):
        self.signals.update_goal.emit(msg.data)

    def _image_callback(self, msg: Image):
        if msg.encoding == 'rgb8':
            # Use QImage.Format_RGB888 for compatibility with older PyQt5 versions
            q_img = QImage(msg.data, msg.width, msg.height, msg.step, QImage.Format_RGB888)
            self.signals.update_image.emit(q_img.copy())
            
        elif msg.encoding == 'bgr8':
            bgr_data = bytearray(msg.data)
            for i in range(0, len(bgr_data), 3):
                bgr_data[i], bgr_data[i+2] = bgr_data[i+2], bgr_data[i]
                
            q_img = QImage(bytes(bgr_data), msg.width, msg.height, msg.step, QImage.Format_RGB888)
            self.signals.update_image.emit(q_img.copy())


class GuiSignals(QWidget):
    """Thread-safe signals to communicate from ROS 2 thread to PyQt thread."""
    update_state = pyqtSignal(str)
    update_goal = pyqtSignal(str)
    update_image = pyqtSignal(QImage)
    append_msg = pyqtSignal(str)


class MissionMainWindow(QMainWindow):
    """Main Window UI matching the custom layout wireframe."""

    def __init__(self, ros_node, gui_signals):
        super().__init__()
        self.node = ros_node
        self.signals = gui_signals

        self.setWindowTitle("Limo Mission Control Panel")
        self.resize(800, 650)

        # Connect signals to slots
        self.signals.update_state.connect(self.set_state)
        self.signals.update_goal.connect(self.set_goal)
        self.signals.update_image.connect(self.set_image)
        self.signals.append_msg.connect(self.log_message)

        self._init_ui()

    def _init_ui(self):
        master_widget = QWidget()
        self.setCentralWidget(master_widget)
        global_layout = QVBoxLayout(master_widget)

        top_panels_layout = QHBoxLayout()

        # =====================================================
        # LEFT COLUMN
        # =====================================================
        left_layout = QVBoxLayout()

        self.image_label = QLabel("/limo/mission_visualizer/image")
        self.image_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        left_layout.addWidget(self.image_label, stretch=4)

        pr_layout = QHBoxLayout()
        self.btn_pause = QPushButton("PAUSE")
        self.btn_resume = QPushButton("RESUME")
        
        # Style configuration for PAUSE and RESUME buttons
        self.btn_pause.setStyleSheet(
            "background-color: #f57c00; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        self.btn_resume.setStyleSheet(
            "background-color: #388e3c; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        
        self.btn_pause.clicked.connect(lambda: self.node.send_cmd_async("pause"))
        self.btn_resume.clicked.connect(lambda: self.node.send_cmd_async("resume"))
        
        pr_layout.addWidget(self.btn_pause)
        pr_layout.addWidget(self.btn_resume)
        left_layout.addLayout(pr_layout)

        telemetry_box = QWidget()
        telemetry_box.setStyleSheet("border: 1px solid black; background-color: #fafafa;")
        telemetry_layout = QVBoxLayout(telemetry_box)
        
        self.lbl_state = QLabel("STATE: Unknown")
        self.lbl_goal = QLabel("GOAL: None")
        self.lbl_state.setStyleSheet("border: none; font-weight: bold; color: black;")
        self.lbl_goal.setStyleSheet("border: none; font-weight: bold; color: black;")
        
        telemetry_layout.addWidget(self.lbl_state)
        telemetry_layout.addWidget(self.lbl_goal)
        left_layout.addWidget(telemetry_box)

        # =====================================================
        # RIGHT COLUMN
        # =====================================================
        right_layout = QVBoxLayout()

        self.cmd_input = QTextEdit()
        self.cmd_input.setPlaceholderText("Write the add command here...\nExample:\nadd 1.0 2.5 90\n\n[Press ENTER to insert the goal]")
        self.cmd_input.setStyleSheet("border: 1px solid black; background-color: #ffffff; color: #000000;")
        self.cmd_input.installEventFilter(self)
        right_layout.addWidget(self.cmd_input, stretch=4)

        self.btn_send = QPushButton("SEND")
        self.btn_abort = QPushButton("ABORT")
        
        # Style configuration for SEND and ABORT buttons
        self.btn_send.setStyleSheet(
            "background-color: #0056b3; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        self.btn_abort.setStyleSheet(
            "background-color: #d32f2f; color: white; font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        
        self.btn_send.clicked.connect(lambda: self.node.send_cmd_async("send"))
        self.btn_abort.clicked.connect(lambda: self.node.send_cmd_async("abort"))
        
        right_layout.addWidget(self.btn_send)
        right_layout.addWidget(self.btn_abort)

        top_panels_layout.addLayout(left_layout, stretch=3)
        top_panels_layout.addLayout(right_layout, stretch=1)
        global_layout.addLayout(top_panels_layout, stretch=5)

        # =====================================================
        # BOTTOM MESSAGE AREA
        # =====================================================
        self.msg_log = QTextEdit()
        self.msg_log.setReadOnly(True)
        self.msg_log.setPlaceholderText("msgs output log Window...")
        self.msg_log.setStyleSheet("border: 1px solid black; background-color: #2b2b2b; color: #a9b7c6;")
        self.msg_log.setFixedHeight(130)
        
        global_layout.addWidget(self.msg_log)

    # =====================================================
    # TRANSMISSION LOGIC AND EVENTS
    # =====================================================

    def _send_current_input(self):
        """Retrieves the entire text written in the command area and transmits it."""
        command = self.cmd_input.toPlainText().strip()
        if command:
            self.node.send_cmd_async(command)
            self.cmd_input.clear()

    @pyqtSlot(str)
    def set_state(self, state_str):
        self.lbl_state.setText(f"STATE: {state_str}")

    @pyqtSlot(str)
    def set_goal(self, goal_str):
        self.lbl_goal.setText(f"GOAL: {goal_str}")

    @pyqtSlot(QImage)
    def set_image(self, q_img):
        if q_img.isNull():
            return
            
        if self.image_label.text():
            self.image_label.setText("")

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.update()

    @pyqtSlot(str)
    def log_message(self, text):
        self.msg_log.append(text)

    def eventFilter(self, obj, event):
        """Intercepts a single Enter key press to immediately send the add command."""
        if obj is self.cmd_input and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._send_current_input()
                return True
        return super().eventFilter(obj, event)


# =====================================================
# RUNNER
# =====================================================

def main(args=None):
    rclpy.init(args=args)

    app = QApplication(sys.argv)
    gui_signals = GuiSignals()

    node = MissionGuiNode(gui_signals)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    window = MissionMainWindow(node, gui_signals)
    window.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()