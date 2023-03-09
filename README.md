# Unscented-Kalman-Filter

Implemented an Unscented Kalman Filter (UKF) to track the orientation of a robot in three-dimensions. 


The observations from an inertial measurement unit (IMU) consists of a data from a gyroscope (angular velocity) and an accelerometer (accelerations in body frame). The ground truth of the orientation is given by a motion-capture system called “Vicon”.

The UKF is developed for the IMU data by using the Vicon data for calibration and tuning of this filter.

Roll | Pitch | Yaw 
--- | --- | ---
 ![](https://user-images.githubusercontent.com/68454938/223841646-379f876f-9cf8-4951-b6be-ecf1e0f7175a.png) | ![](https://user-images.githubusercontent.com/68454938/223841633-f0d165d9-aefb-49df-b371-1622891c4b8b.png) | ![](https://user-images.githubusercontent.com/68454938/223841636-c840512e-a59b-48f4-9e60-73653146c923.png)

Angular Velocity along x | Angular Velocity along y | Angular Velocity along z
--- | --- | ---
![](https://user-images.githubusercontent.com/68454938/223842989-e6f818ff-ebb9-4419-a4e8-9e2adb2b836d.png) | ![](https://user-images.githubusercontent.com/68454938/223843015-d514b605-56fb-46df-a8ee-c597676195d8.png) | ![](https://user-images.githubusercontent.com/68454938/223843025-20b9f6f3-051a-4076-9516-88de8377d53f.png)

Quaternion | Quaternion along x | Quaternion along y | Quaternion along z 
--- | --- | --- | ---
![Figure_4](https://user-images.githubusercontent.com/68454938/223843052-7b33d456-5030-4ce5-9b3b-8fb94b313910.png) | ![Figure_5](https://user-images.githubusercontent.com/68454938/223843275-31a2697b-999a-47a0-a178-cf524709e40b.png) | ![Figure_6](https://user-images.githubusercontent.com/68454938/223843064-e7bac4e5-26ee-4901-8d6f-2e3263c693cc.png) | ![Figure_7](https://user-images.githubusercontent.com/68454938/223843074-58e9d547-8d98-40aa-b65c-ce4de525c44d.png)

# Reference
E. Kraft, "A quaternion-based unscented Kalman filter for orientation tracking," Sixth International Conference of Information Fusion, 2003. Proceedings of the, Cairns, Queensland, Australia, 2003, pp. 47-54, doi: 10.1109/ICIF.2003.177425.
