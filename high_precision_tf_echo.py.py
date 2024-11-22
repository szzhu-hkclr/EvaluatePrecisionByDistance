#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped

class HighPrecisionTfEcho(Node):

    def __init__(self, source_frame, target_frame):
        super().__init__('high_precision_tf_echo')
        self.source_frame = source_frame
        self.target_frame = target_frame

        # Create a TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to periodically query and print the transform
        self.timer = self.create_timer(0.1, self.print_transform_with_high_precision)  # 10 Hz

    def print_transform_with_high_precision(self):
        try:
            # Lookup the transform
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time())  # Get the latest available transform

            # Extract translation
            translation = transform.transform.translation
            translation_x = translation.x
            translation_y = translation.y
            translation_z = translation.z

            # Extract rotation (quaternion)
            rotation = transform.transform.rotation
            rotation_x = rotation.x
            rotation_y = rotation.y
            rotation_z = rotation.z
            rotation_w = rotation.w

            # Print with high precision
            self.get_logger().info(f"Translation: [{translation_x:.9f}, {translation_y:.9f}, {translation_z:.9f}]")
            self.get_logger().info(f"Rotation (Quaternion): [{rotation_x:.9f}, {rotation_y:.9f}, {rotation_z:.9f}, {rotation_w:.9f}]")

        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {str(e)}")


def main(args=None):
    rclpy.init(args=args)

    # Replace these with your source and target frames
    source_frame = 'source_frame_name'
    target_frame = 'target_frame_name'

    # Create and spin the node
    node = HighPrecisionTfEcho(source_frame, target_frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Shutdown and cleanup
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()