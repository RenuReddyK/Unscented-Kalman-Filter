import numpy as np
import scipy
from scipy import io
from quaternion import Quaternion
import math
import matplotlib.pyplot as plt

#roll pitch and yaw using an unscented kalman filter

def estimate_rot(data_num=1):

    #Dimensions:
    # T = 5645  
    # Vicon data = (3, 3, 5561)  
    # Acceleration values = (3, 5645)  
    # Gyroscope values = (3, 5645)

    # #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = np.array(imu['vals'][0:3,:], dtype=np.float64)

    gyro = np.array(imu['vals'][3:6,:], dtype=np.float64)
    T = np.shape(imu['ts'])[1]

    # Initial covariance of the state, dynamics noise and measurement noise are assumed as the following:  
    Cov_k_k = 0.1 * np.diag(np.ones(6))
    R_t = np.diag([9, 9, 9, 5, 5, 5])  # dynamics noise
    Q = np.diag([0.009,0.009,0.009, 0.27, 0.27, 0.27]) # measurement noise 
    
    # Caliberated values
    a_bias_x = 510.80923077
    a_bias_y =  500.99384615
    a_bias_z = 505.90153846

    a_sens_x = 32.64027712389943
    a_sens_y = 32.64027712389943
    a_sens_z = 32.64027712389943

    g_sens_x = 181.71408595333332 
    g_sens_y = 208.87758669 
    g_sens_z =  209.29583632166666 
    #print((g_sens_x + g_sens_y)/2)
    
    g_bias_x = 369.68769231
    g_bias_y = 373.56769231
    g_bias_z = 375.37076923

    updated_acc = np.zeros(accel.shape)
    updated_gyro = np.zeros(gyro.shape)

    updated_acc[0] = -(accel[0,:] - a_bias_x) * 3300/(1023 * a_sens_x)
    updated_acc[1] = -(accel[1,:] - a_bias_y) * 3300/(1023 * a_sens_y)
    updated_acc[2] = (accel[2,:] - a_bias_z) * 3300/(1023*a_sens_z)
    
    updated_gyro[0] = (gyro[1,:] - g_bias_y) * 3300/(1023 * g_sens_y)
    updated_gyro[1] = (gyro[2,:] - g_bias_z) * 3300/(1023 * g_sens_z)
    updated_gyro[2] = (gyro[0,:] - g_bias_x) * 3300/(1023 * g_sens_x)

    # State (Estimate of the state is given by the mean and covariance)
    
    # Quaternions
    quat = Quaternion()
    vec = Quaternion().vec().reshape((3,1))
    
    # Angular velocity
    w_ =  0.5 * np.ones(3)

    # State vector
    x = np.hstack((quat.q,w_))[:,np.newaxis]        # x --> (7, 1)
    aaa = Quaternion(x[0], x[1:4].reshape((3,)))
    angle_i = aaa.euler_angles()

    roll = [angle_i[0]]
    pitch = [angle_i[1]]
    yaw = [angle_i[2]]
    Time = np.array(imu['ts']).T
    n = 6
    Y_i = np.zeros((7, 12))
    for timestep in range(1,T):
        # Sigma points 
        W = np.hstack((scipy.linalg.sqrtm(n*(Cov_k_k + Q)), -scipy.linalg.sqrtm(n*(Cov_k_k + Q))))        # W --> (6, 12)
        w_ = x[4:,0]
        X = np.zeros((7,12))

        # H1 Measurement  
        q_delta = Quaternion()
        scalar = x[0,0]
        vec = x[1:4,0]
        delta_t = Time[timestep] -  Time[timestep-1]
        quat = Quaternion( scalar, vec)
        for i in range(12):
            quat.from_axis_angle(W[:3,i])
            a = Quaternion(scalar, vec.reshape((3,))) * quat
            b = w_ + W[3:,i]
            X[:,i] = np.real((np.hstack(((Quaternion(scalar, vec.reshape((3,))) * quat).q, b))).T)       # X --> (7, 12)
            
            #X[:,i] =AB

            # print("AB")
            # print(AB)
            # try:
            #     X[:,i] = AB
            # except:
            #     print(f"\n a: {a}")
            #     print(f"\n b: {b}")
            #     print(f"\n b: {type(b)}")
            #     print(f"\n AB: {AB}")
            C = (x[4:,] * delta_t).reshape((3,))
            q_delta.from_axis_angle(C)
    
            # Transforming sigma points and finding mean
            q_k_w_delta = a * q_delta
            Y_i[:4, i] = q_k_w_delta.q
            Y_i[4:, i] = X[4:, i]

        q_prev = Quaternion(Y_i[0,0], Y_i[1:4, 0].reshape(3,))

        error_vec = np.zeros((3,12))
        for i in range(80):                    # Max_steps = 500 
            for k in range(12):
                e = Quaternion(Y_i[0, k], Y_i[1:4, k]) * q_prev.inv()
                if e.q[0] > 1:
                    if abs(e.q[0] - 1.0) < 0.0001:
                        e.q[0] = 1
                if e.q[0] < -1:
                    if abs(e.q[0] + 1.0) < 0.0001:
                        e.q[0] = -1
                error_vec[:, k] = e.axis_angle()
            
            error_mean = np.mean(error_vec, axis=1)
            error_mean_q = Quaternion()
            error_mean_q.from_axis_angle(error_mean) 
            
            # Estimate of the quaternion for the next step iteration
            q_prev = error_mean_q * q_prev
            if np.linalg.norm(error_mean) < 0.01 :    
                break

        # A Priori State Vector Covariance
        r_wi =  error_vec             # q_i * Quaternion.inv(q_est)
        w_wi = np.zeros((3,12))
        w_wi = Y_i[4:,:] - np.mean(Y_i[4:,:], axis =1).reshape((-1,1))     
        W_i = np.vstack((r_wi, w_wi))     # W_i --> (6, 12)
        
        # updating the mean and covariance after gradient descent
        x[:4,0] = (q_prev.q)
        x[4:,0] = (np.mean(Y_i[4:,:],axis =1))
        Cov_k_k = (1/12) * (np.matmul(W_i, W_i.T))
        g_quaternion = Quaternion(0, [0, 0, 9.81])

        #Calculating Z 
        Z = np.zeros((6,12))
        for i in range(12):
            qua = Quaternion(Y_i[0, i] ,Y_i[1:4, i])
            g_prime = Quaternion.inv(qua) * g_quaternion * qua
            Z[:, i] = (np.vstack((g_prime.vec().reshape((3,1)), np.array(Y_i[4:,i]).reshape((3,1))))).reshape((6,))
        z_mean = np.mean(Z, axis =1).reshape((6,1))

        # Measurement estimate covariance 
        P_zz =  (1/12) * np.matmul((Z-z_mean), (Z-z_mean).T)
        P_vv = P_zz + R_t

        # Cross correlation matrix
        P_xz = (1/12) * np.matmul(W_i, (Z - z_mean).T)

        # Computing the Kalman gain
        K = np.matmul(P_xz,  np.linalg.inv(P_vv))
        
        # Update sensor reading
        # Updating the mean and covariance using the latest observation
        Cov_FINAL = Cov_k_k - np.matmul(np.matmul(K, P_vv), K.T)
        q1 = Quaternion(x[0,0], x[1:4,0].reshape((3,)))
            
        # Update of the a posteriori estimate 
        updated_acc_val = np.vstack((updated_acc[0,timestep-1], updated_acc[1,timestep-1], updated_acc[2,timestep-1]))
        updated_gyro_val = np.vstack((updated_gyro[0,timestep-1],updated_gyro[1,timestep-1],updated_gyro[2,timestep-1]))
        SENSOR = np.vstack((updated_acc_val, updated_gyro_val)) #. reshape((-1,1))
        v_k = SENSOR.reshape((6,1)) - z_mean
        m = np.matmul(K, v_k)
        
        q2 = Quaternion()
        q2.from_axis_angle(m[:3,:].reshape((3,)))
        x_k =  q2 * q1

        Angles = Quaternion(x_k.scalar(), x_k.vec()).euler_angles()
        roll.append(Angles[0])
        pitch.append(Angles[1])
        yaw.append(Angles[2])

        Cov_k_k = Cov_FINAL
        x = np.hstack(( x_k.q, x[4:,0] + m[3:,0] ))[:,np.newaxis]
        w_ = x[4:,:].reshape((3,))  + m[3:,:]
        
    
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    vicon_R = vicon['rots']
    vicon_alpha = np.arctan2(vicon_R[1,0,:], vicon_R[0,0,:]) #yaw
    vicon_beta = np.arctan2(-1*vicon_R[2,0,:], np.sqrt(np.square(vicon_R[2,1,:]) + np.square(vicon_R[2,2,:]))) #pitch
    vicon_gamma = np.arctan2(vicon_R[2,1,:], vicon_R[2,2,:]) #roll    
            
    x = np.arange(0,5561)
    plt.title("Vicon pitch vs Predicted pitch")
    plt.xlabel("Time")
    plt.ylabel("Pitch angle")
    
    plt.plot(x,vicon_beta[:5561], color='deeppink', label='Vicon Pitch')
    plt.plot(x,pitch[:5561], color='teal', label='Predicted Pitch')
    plt.legend()
    plt.show()

    x = np.arange(0,5561)
    plt.title("Vicon yaw vs Predicted yaw")
    plt.xlabel("Time")
    plt.ylabel("Yaw angle")
    
    plt.plot(x,vicon_alpha[:5561], color='deeppink', label='Vicon Yaw')
    plt.plot(x,yaw[:5561], color='teal', label='Predicted Yaw')
    plt.legend()
    plt.show()

    x = np.arange(0,5561)
    plt.title("Vicon roll vs Predicted roll")
    plt.xlabel("Time")
    plt.ylabel("Roll angle")
    
    plt.plot(x,vicon_gamma[:5561], color='deeppink', label='Vicon Roll')
    plt.plot(x,roll[:5561], color='teal', label='Predicted Roll')
    plt.legend()
    plt.show()

    #----------------Caliberation-----------------------
    # IMU Rectification (Ax, Ay directions and order of Wz)
    # Rectified_accel = [accel[0,:],accel[1,:],accel[2,:]]
    # Rectified_gyro = [gyro[1,:],gyro[2,:],gyro[0,:]] 
    # Rectified_gyro = gyro
    # # # Acceleration caliberation (the magnitude = 9.81)
    # Rectified_accel = np.array(Rectified_accel)  # (3, 5645)
    # Beta_acc = np.mean(Rectified_accel[:,:650],axis =1)
    # # # print(Beta_acc)
    # Beta_acc[2] = (Beta_acc[1]+Beta_acc[0])/2
    # a = np.sqrt((np.square((accel[0,:650]-Beta_acc[0])*3300/1023) + np.square((accel[1,:650]-Beta_acc[1]) *3300/1023) + np.square((accel[2,:650]-Beta_acc[2])*3300/1023))/(9.81*9.81))
    # Alpha_acc = np.mean(a)
    # # print(Beta_acc)
    # # print(Alpha_acc)
    # # Gyroscope caliberation 
    # Beta_gyro = np.mean(Rectified_gyro[:,:650],axis =1)
    # print(Beta_gyro)

    # w_A = np.zeros((650,1)) # yaw
    # w_B = np.zeros((650,1)) # pitch
    # w_C = np.zeros((650,1)) # roll
    # Alpha_gyro_x = np.zeros((649,1))
    # Alpha_gyro_y = np.zeros((649,1))
    # Alpha_gyro_z = np.zeros((649,1))
    # A = np.arctan2(vicon["rots"][1,0,:],vicon["rots"][0,0,:]) 
    # B = np.arctan2(-vicon["rots"][2,0,:], np.sqrt(np.square(vicon["rots"][2,1,:])+ np.square(vicon["rots"][2,2,:])))
    # C = np.arctan2(vicon["rots"][2,1,:], vicon["rots"][2,2,:])
    # # True angular velocity
    # for i in range(1,650):
    #     w_A[i-1] = A[i] - A[i-1]        # here delta T =1 
    #     w_B[i-1] = B[i] - B[i-1]        # here delta T =1 
    #     w_C[i-1] = C[i] - C[i-1]        # here delta T =1 
    #     Alpha_gyro_x[i-1] = ((Rectified_gyro[0,i] - Beta_gyro[0]) * 3300 )/(w_A[i-1] * 1023)
    #     Alpha_gyro_y[i-1] = ((Rectified_gyro[1,i] - Beta_gyro[1]) * 3300 )/(w_C[i-1] * 1023)
    #     Alpha_gyro_z[i-1] = ((Rectified_gyro[2,i] - Beta_gyro[2]) * 3300 )/(w_B[i-1] * 1023)
    # Alpha_gyro = np.hstack((np.mean(Alpha_gyro_x[np.where((np.abs(Alpha_gyro_x) < 1e4))]), np.mean(Alpha_gyro_y[np.where((np.abs(Alpha_gyro_y) < 1e4))]), np.mean(Alpha_gyro_z[np.where((np.abs(Alpha_gyro_z) < 1e4))])))
    # Alpha_gyro = np.absolute(Alpha_gyro)
    #----------------------------------------------------------------------------------------------------------------------------
    

    return roll,pitch,yaw


if __name__ == "__main__":
    estimate_rot()