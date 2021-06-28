#!/usr/bin/env/python3
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

DUMMY_FIELD_PREFIX = '__'
# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/lidar_front/points_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.con_lidar_front = np.asarray([], dtype=np.float64, order ='C')


    def fields_to_dtype(self, fields, point_step):
        '''Convert a list of PointFields to a numpy record datatype.
        '''
        offset = 0
        np_dtype_list = []
        for f in fields:
            while offset < f.offset:
                # might be extra padding between fields
                np_dtype_list.append(
                    ('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1

            dtype = pftype_to_nptype[f.datatype]
            if f.count != 1:
                dtype = np.dtype((dtype, f.count))

            np_dtype_list.append((f.name, dtype))
            offset += pftype_sizes[f.datatype] * f.count

        # might be extra padding between points
        while offset < point_step:
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        return np_dtype_list

    def pointcloud2_to_array(self, cloud_msg, squeeze=True):
        # construct a numpy record type equivalent to the point type of this cloud
        dtype_list = self.fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)
        # parse the cloud into an array
        cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)
        # remove the dummy fields that were added
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
        if squeeze and cloud_msg.height == 1:
            return np.reshape(cloud_arr, (cloud_msg.width,))
        else:
            return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

    def listener_callback(self, cloud_msg):

        lidar_front = self.pointcloud2_to_array(cloud_msg)
        lidar_front_flt = np.array(lidar_front, dtype=np.float64)

        self.get_logger().info(f'Data Type: {type(self.con_lidar_front)}')
        self.get_logger().info(f'Data Type: {lidar_front.dtype}')

        self.con_lidar_front = (np.concatenate((lidar_front_flt, self.con_lidar_front), dtype=np.float64))



def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
