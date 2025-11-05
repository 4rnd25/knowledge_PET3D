"""
Created on Jan 20 2025 11:11

@author: ISAC - pettirsch
"""

import numpy as np

import numpy as np



class UKFVehicleTracker:
    def __init__(self, initial_pos, initial_yaw, initial_speed, dt=0.1, process_noise=1e-2, measurement_noise=1e-1,
                 velocity_smoothing = 0.8, acc_smoothing = 0.8, yaw_smoothing = 0.8, yaw_rate_smoothing = 0.8):
        """
        UKF for vehicle tracking with position, velocity, acceleration, yaw, and yaw rate.

        Args:
            dt (float): Time step (seconds).
            process_noise (float): Process noise covariance.
            measurement_noise (float): Position measurement noise covariance.
        """
        self.dt = dt  # Time step

        # State vector: [x, y, z, v, a, yaw, yaw_rate]
        if initial_pos is None or initial_speed is None:
            self.state = np.zeros(7)
        else:
            self.state = np.array([initial_pos[0], initial_pos[1], initial_pos[2], initial_speed/3.6, 0, initial_yaw, 0])

        # State covariance matrix
        self.P = np.eye(7) * 1.0

        # Process noise matrix (allows adaptation to rapid motion)
        self.Q = np.eye(7) * process_noise
        self.Q[3, 3] = 0.5  # Increase process noise for velocity
        self.Q[4, 4] = 0.3  # Increase process noise for acceleration
        self.Q[5, 5] = 0.1  # Increase process noise for yaw
        self.Q[6, 6] = 0.2  # Increase process noise for yaw rate

        # Measurement noise matrix (only for position)
        self.R = np.eye(3) * measurement_noise

        # Sigma point parameters
        self.alpha = 0.1  # Spread of sigma points
        self.kappa = 0  # Secondary scaling
        self.beta = 2  # Optimized for Gaussian distributions
        self.n = 7  # State dimension
        self.lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n

        # Compute sigma point weights
        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wc = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha ** 2 + self.beta)

        # Store previous measurements for velocity & yaw estimation
        self.previous_position = self.state[:3]
        self.previous_velocity = self.state[3]
        self.previous_yaw = self.state[5]
        self.previous_acc = self.state[4]
        self.previous_yaw_rate = self.state[6]

        self.velocity_smoothing = velocity_smoothing
        self.acc_smoothing = acc_smoothing
        self.yaw_smoothing = yaw_smoothing
        self.yaw_rate_smoothing = yaw_rate_smoothing

    def generate_sigma_points(self):
        """
        Generate sigma points for the Unscented Transform.
        """
        # Regularize and enforce symmetry
        self.P = (self.P + self.P.T) / 2  # Force symmetry
        self.P += np.eye(self.n) * 1e-6  # Small regularization for numerical stability

        # Compute square root using SVD instead of Cholesky
        U, S_diag, Vt = np.linalg.svd((self.n + self.lambda_) * self.P)
        S = np.dot(U, np.diag(np.sqrt(S_diag)))

        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.state

        for i in range(self.n):
            sigma_points[i + 1] = self.state + S[i]
            sigma_points[self.n + i + 1] = self.state - S[i]

        return sigma_points

    def motion_model(self, state):
        """
        Predicts the next state using the nonlinear motion model.
        """
        x, y, z, v, a, yaw, yaw_rate = state

        # Compute new velocity
        v_new = v + a * self.dt

        # Compute new yaw angle
        yaw_new = yaw + yaw_rate * self.dt
        yaw_new = (yaw_new + np.pi) % (2 * np.pi) - np.pi  # Normalize yaw

        # Update position using velocity and yaw
        if np.abs(yaw_rate) > 1e-3:  # Turning motion
            x_new = x + (v_new / yaw_rate) * (np.sin(yaw_new) - np.sin(yaw))
            y_new = y + (v_new / yaw_rate) * (-np.cos(yaw_new) + np.cos(yaw))
        else:  # Straight-line motion
            x_new = x + v_new * np.cos(yaw) * self.dt
            y_new = y + v_new * np.sin(yaw) * self.dt

        z_new = z  # No vertical motion assumed

        return np.array([x_new, y_new, z_new, v_new, a, yaw_new, yaw_rate])

    def predict(self):
        """
        Predicts the next state using the Unscented Transform.
        """
        sigma_points = self.generate_sigma_points()

        # Propagate each sigma point through the motion model
        transformed_sigma_points = np.array([self.motion_model(sp) for sp in sigma_points])

        # Compute predicted state mean
        self.state = np.sum(self.Wm[:, np.newaxis] * transformed_sigma_points, axis=0)

        # Compute predicted state covariance
        self.P = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff = transformed_sigma_points[i] - self.state
            self.P += self.Wc[i] * np.outer(diff, diff)
        self.P += self.Q  # Add process noise

        return self.state

    def update(self, measurement):
        """
        Updates the state using the Unscented Transform.

        Args:
            measurement (ndarray): Observed [x, y, z] position.
        """
        sigma_points = self.generate_sigma_points()

        # Transform sigma points into measurement space
        transformed_sigma_points = sigma_points[:, :3]  # Extract only (x, y, z)

        # Compute predicted measurement mean
        z_pred = np.sum(self.Wm[:, np.newaxis] * transformed_sigma_points, axis=0)

        # Compute innovation covariance
        S = np.zeros((3, 3))
        for i in range(2 * self.n + 1):
            diff = transformed_sigma_points[i] - z_pred
            S += self.Wc[i] * np.outer(diff, diff)
        S += self.R  # Add measurement noise

        # Compute cross covariance
        cross_cov = np.zeros((self.n, 3))
        for i in range(2 * self.n + 1):
            state_diff = sigma_points[i] - self.state
            measurement_diff = transformed_sigma_points[i] - z_pred
            cross_cov += self.Wc[i] * np.outer(state_diff, measurement_diff)

        # Compute Kalman gain
        K = cross_cov @ np.linalg.inv(S)

        # Update state
        measurement = measurement.reshape(-1)
        innovation = measurement - z_pred
        self.state += K @ innovation

        # Estimate velocity and acceleration from position changes
        velocity_est = (self.state[:3] - self.previous_position) / self.dt
        velocity_est_scalar = np.linalg.norm(velocity_est)
        self.state[3] = self.velocity_smoothing * self.previous_velocity + (
                    1 - self.velocity_smoothing) * velocity_est_scalar

        acceleration_est = (self.state[3] - self.previous_velocity) / self.dt
        self.state[4] = self.acc_smoothing * self.previous_acc + (1 - self.acc_smoothing) * acceleration_est



        yaw_est = np.arctan2(velocity_est[1], velocity_est[0])
        yaw_est = (yaw_est + np.pi) % 2*np.pi - np.pi
        self.state[5] = self.yaw_smoothing * self.previous_yaw + (1 - self.yaw_smoothing) * yaw_est
        yaw_rate_est = (self.state[5] - self.previous_yaw) / self.dt
        self.state[6] = self.yaw_rate_smoothing * self.previous_yaw_rate + (1 - self.yaw_rate_smoothing) * yaw_rate_est

        # Update stored previous values
        self.previous_position = self.state[:3]
        self.previous_velocity = self.state[3]
        self.previous_acc = self.state[4]
        self.previous_yaw = self.state[5]
        self.previous_yaw_rate = self.state[6]

        # Update state covariance
        self.P -= K @ S @ K.T

        return self.state

    def get_state(self):
        """
        Returns the estimated state [x, y, z, v, a, yaw, yaw_rate].
        """
        return self.state

    def get_covariance(self):
        """
        Returns the state covariance matrix.
        """
        return self.P


class KalmanFilter:
    def __init__(self, initial_pos=None, initial_speed=None, initial_yaw=None, dt=0.1, process_noise=1e-2,
                 measurement_noise=1e-1, yaw_smoothing_factor=0.8):
        """
        Kalman Filter for 3D motion tracking with velocity, acceleration, yaw, and yaw rate.

        Args:
            dt (float): Time step (seconds).
            process_noise (float): Process noise covariance.
            measurement_noise (float): Position measurement noise covariance.
        """
        self.dt = dt  # Time step
        self.yaw_smoothing_factor = yaw_smoothing_factor

        # State vector: [x, y, z, vx, vy, vz, ax, ay, az, yaw, yaw_rate]
        if initial_pos is None or initial_speed is None:
            self.state = np.zeros(11)
        else:
            vx = initial_speed / (3.6 * 30) * np.cos(initial_yaw)
            vy = initial_speed / (3.6 * 30) * np.sin(initial_yaw)
            self.state = np.array([initial_pos[0], initial_pos[1], initial_pos[2], vx, vy, 0, 0, 0, 0, initial_yaw, 0])

        # State covariance matrix
        self.P = np.eye(11) * 1.0

        # Process noise matrix (allows adaptation to rapid motion)
        self.Q = np.eye(11) * process_noise

        # Measurement noise matrix (only for position)
        self.R = np.eye(3) * measurement_noise

        # Measurement matrix (we only observe position)
        self.H = np.zeros((3, 11))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # z

        # State transition matrix
        self.F = np.eye(11)
        for i in range(3):  # Update position using velocity and acceleration
            self.F[i, i + 3] = self.dt
            self.F[i, i + 6] = 0.5 * self.dt ** 2
        for i in range(3):  # Update velocity using acceleration
            self.F[i + 3, i + 6] = self.dt
        self.F[9, 10] = self.dt  # Yaw update using yaw rate

    def predict(self):
        """
        Predict the next state based on the motion model.
        """
        self.previous_yaw = self.state[9]
        self.previous_velocity = self.state[3:6]

        # Extract state components
        x, y, z = self.state[0:3]
        vx, vy, vz = self.state[3:6]
        ax, ay, az = self.state[6:9]
        yaw, yaw_rate = self.state[9:11]

        # Compute new velocity based on acceleration
        vx += ax * self.dt
        vy += ay * self.dt
        vz += az * self.dt

        # Compute new position based on velocity and acceleration
        x += vx * self.dt + 0.5 * ax * self.dt ** 2
        y += vy * self.dt + 0.5 * ay * self.dt ** 2
        z += vz * self.dt + 0.5 * az * self.dt ** 2

        # Update yaw angle based on yaw rate
        yaw += yaw_rate * self.dt

        # Ensure yaw remains within [-π, π]
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

        # Compute new velocity direction based on yaw
        v_magnitude = np.sqrt(vx ** 2 + vy ** 2)
        vx = v_magnitude * np.cos(yaw)
        vy = v_magnitude * np.sin(yaw)

        # Update state
        self.state = np.array([x, y, z, vx, vy, vz, ax, ay, az, yaw, yaw_rate])

        # Predict state covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state

    def update(self, measurement):
        """
        Update the state based on the new position measurement.

        Args:
            measurement (ndarray): Observed [x, y, z] position.
        """
        measurement = measurement.reshape(-1)

        # Compute innovation (residual)
        y = measurement - (self.H @ self.state)

        # Compute innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Compute Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state with measurement
        test = (K @ y).flatten()
        self.state += (K @ y).flatten()

        # Estimate yaw using velocity direction
        self.estimate_yaw_from_velocity()

        # Estimate yaw rate using the change in yaw
        self.estimate_yaw_rate()

        # Update covariance matrix
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def estimate_yaw_from_velocity(self):
        """
        Estimates yaw from velocity and applies smoothing.
        """
        vx, vy = self.state[3], self.state[4]
        speed = np.linalg.norm([vx, vy])

        # Ignore yaw updates for very slow movement
        if speed < 0.2:
            self.state[9] = self.previous_yaw
            return  # Keep the last known yaw

        # Compute new yaw from velocity
        new_yaw = np.arctan2(vy, vx)
        new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize yaw

        # Apply smoothing to prevent jumps
        self.state[9] = self.yaw_smoothing_factor * self.previous_yaw + (1 - self.yaw_smoothing_factor) * new_yaw
        self.state[9] = (self.state[9] + np.pi) % (2 * np.pi) - np.pi  # Normalize yaw

    def estimate_yaw_rate(self):
        """
        Estimates yaw rate (ω) from yaw changes over time.
        """
        current_yaw = self.state[9]
        previous_yaw = self.previous_yaw

        # Compute yaw rate as the difference over time
        yaw_rate_est = (current_yaw - previous_yaw) / self.dt

        # Apply exponential smoothing for stability
        alpha = 0.5  # Smoothing factor
        self.state[10] = alpha * self.state[10] + (1 - alpha) * yaw_rate_est

        # Store previous yaw for the next update
        self.previous_yaw = current_yaw

    def estimate_acceleration(self):
        """
        Estimates acceleration (ax, ay, az) from velocity changes over time.
        """
        # Get current and previous velocity
        vx, vy, vz = self.state[3:6]
        vx_prev, vy_prev, vz_prev = self.previous_velocity

        # Compute acceleration as the difference over time
        ax_est = (vx - vx_prev) / self.dt
        ay_est = (vy - vy_prev) / self.dt
        az_est = (vz - vz_prev) / self.dt

        # Apply exponential smoothing for stability
        alpha = 0.5  # Smoothing factor
        self.state[6] = alpha * self.state[6] + (1 - alpha) * ax_est
        self.state[7] = alpha * self.state[7] + (1 - alpha) * ay_est
        self.state[8] = alpha * self.state[8] + (1 - alpha) * az_est

        # Store previous velocity for the next update
        self.previous_velocity = np.array([vx, vy, vz])

    def get_state(self):
        """
        Return the full state vector [x, y, z, vx, vy, vz, ax, ay, az, yaw, yaw_rate].
        """
        return self.state

    def get_covariance(self):
        """
        Return the state covariance matrix.
        """
        return self.P

# class KalmanFilter:
#     def __init__(self, dt=0.1, initial_pos=None, initial_speed=50, initial_yaw=None, process_noise=1e-2,
#                  measurement_noise=1e-1, yaw_smoothing_factor=0.8, velocity_smoothing_factor=0.8, adaptive_smoothing=True):
#         """
#         Kalman filter for 3D motion tracking with velocity-position connection.
#
#         Args:
#             dt (float): Time step between measurements.
#             initial_pos (ndarray or None): Initial [x, y, z] position.
#             initial_speed (float): Initial speed estimate in m/s.
#             initial_direction (ndarray or None): Initial movement direction as [dx, dy, dz].
#             process_noise (float): Process noise covariance.
#             measurement_noise (float): Position measurement noise covariance.
#         """
#         self.dt = dt  # Time step
#
#         # Initialize velocity from speed and direction if provided
#         if initial_pos is not None:
#             if initial_yaw is not None:
#                 vx = initial_speed / (3.6 * 30) * np.cos(initial_yaw)
#                 vy = initial_speed / (3.6*30) * np.sin(initial_yaw)
#                 vz = 0
#             else:
#                 vx, vy, vz = 0, 0, 0  # Default: no movement
#             self.state = np.array([initial_pos[0], initial_pos[1], initial_pos[2], vx, vy, vz])
#         else:
#             self.state = np.zeros(6)  # Default all zeros
#         self.yaw = initial_yaw if initial_yaw is not None else 0
#
#         # State covariance matrix
#         self.P = np.eye(6) * 1.0
#
#         # Process noise
#         self.Q = np.eye(6) * process_noise
#
#         # Measurement noise
#         self.R = np.eye(3) * measurement_noise  # x, y, z
#
#         # Measurement matrix (observe position)
#         self.H = np.zeros((3, 6))
#         self.H[0, 0] = 1  # x
#         self.H[1, 1] = 1  # y
#         self.H[2, 2] = 1  # z
#
#         # Transition matrix (connects velocity to position)
#         self.F = np.eye(6)
#         for i in range(3):  # Update position based on velocity
#             self.F[i, i + 3] = self.dt
#
#         self.decay_velocity_bool = False
#         self.yaw_smoothing_factor = yaw_smoothing_factor
#         self.velocity_smoothing_factor = velocity_smoothing_factor
#         self.adaptive_smoothing = adaptive_smoothing
#
#     def predict(self):
#         """
#         Predict the next state based on the motion model.
#         """
#         # Store previous position and velocity before prediction
#         self.previous_position = self.state[:3].copy()
#         self.previous_velocity = self.state[3:6].copy()
#
#         self.state = self.F @ self.state
#
#         # Predict new state covariance
#         self.P = self.F @ self.P @ self.F.T + self.Q
#
#         return self.state
#
#     def derive_yaw(self):
#         """
#         Compute yaw from velocity with smoothing.
#         """
#         vx, vy = self.state[3], self.state[4]
#         speed = np.linalg.norm([vx, vy])
#
#         # Ignore yaw updates for very slow movement
#         if speed < 0.2:
#             return self.yaw  # Keep last known yaw
#
#         # Compute new yaw from velocity
#         new_yaw = np.arctan2(vy, vx)
#
#         # Apply exponential smoothing to reduce yaw jumps
#         self.yaw = self.yaw_smoothing_factor * self.yaw + (1 - self.yaw_smoothing_factor) * new_yaw
#         self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize yaw
#
#         return self.yaw
#
#     def update(self, measurement):
#         """
#         Update the state based on the new position measurement.
#
#         Args:
#             measurement (ndarray): Observed [x, y, z] position.
#         """
#         # Ensure measurement is a column vector
#         measurement = measurement.reshape(-1)
#
#         # Compute innovation (residual)
#         y = measurement - (self.H @ self.state)
#
#         # Compute innovation covariance
#         S = self.H @ self.P @ self.H.T + self.R
#
#         # Compute Kalman gain
#         K = self.P @ self.H.T @ np.linalg.inv(S)
#
#         # Update the state
#         self.state += (K @ y).flatten()
#
#         # Update the velocity based on new position estimate
#         self.update_velocity()
#
#         # Update state covariance
#         I = np.eye(self.P.shape[0])
#         self.P = (I - K @ self.H) @ self.P
#
#     def update_velocity(self):
#         """
#         Update velocity using the difference between consecutive positions.
#         This ensures velocity and position are dynamically connected.
#         """
#         raw_velocity = (self.state[:3] - self.previous_position) / self.dt
#         acceleration = (raw_velocity - self.previous_velocity) / self.dt
#         acceleration_magnitude = np.linalg.norm(acceleration)
#
#         # Apply adaptive smoothing: less smoothing when acceleration is high
#         if self.adaptive_smoothing:
#             dynamic_smoothing_factor = np.clip(1 - acceleration_magnitude / 10, 0.3, 0.9)
#         else:
#             dynamic_smoothing_factor = self.velocity_smoothing_factor
#
#         # Exponential smoothing for velocity
#         self.state[3:6] = dynamic_smoothing_factor * self.previous_velocity + (
#                     1 - dynamic_smoothing_factor) * raw_velocity
#
#         # Store last known position, velocity, and acceleration
#         self.previous_position = self.state[:3].copy()
#         self.previous_velocity = self.state[3:6].copy()
#         self.previous_acceleration = acceleration.copy()
#
#     def decay_velocity(self):
#         """
#         Apply exponential velocity decay to slow down movement when no detection occurs.
#         """
#         self.decay_velocity_bool = True
#         self.state[3:] *= 0.8  # Decay velocity components
#
#     def set_decay_velocity(self, decay_velocity):
#         """
#         Enable or disable velocity decay.
#         """
#         self.decay_velocity_bool = decay_velocity
#
#     def get_covariance(self):
#         """
#         Return the state covariance matrix.
#         """
#         return self.P
#
#     def get_state(self):
#         """
#         Return the full state vector [x, y, z, vx, vy, vz].
#         """
#         return self.state

# class KalmanFilterYaw:
#     def __init__(self, dt=0.1, initial_pos=None, initial_speed=50, initial_yaw=0, process_noise=1e-2,
#                  measurement_noise=1e-1, yaw_measurement_noise=1.0):
#         """
#         Kalman filter with yaw and yaw rate integration.
#
#         Args:
#             dt (float): Time step between measurements.
#             process_noise (float): Process noise covariance.
#             measurement_noise (float): Position measurement noise covariance.
#             yaw_measurement_noise (float): Yaw measurement noise covariance.
#         """
#         self.dt = dt  # Time step
#
#         # State vector: [x, y, z, vx, vy, vz, yaw, yaw_rate]
#         if initial_pos is not None:
#             vx = initial_speed / (3.6*30) * np.cos(initial_yaw)
#             vy = initial_speed / (3.6*30) * np.sin(initial_yaw)
#             self.state = np.array([initial_pos[0], initial_pos[1], initial_pos[2], vx, vy, 0])
#         else:
#             vx = initial_speed / (3.6*30) * np.cos(initial_yaw)
#             vy = initial_speed / (3.6*30) * np.sin(initial_yaw)
#             self.state = np.array([0, 0, 0, vx, vy, 0])
#         self.last_valid_yaw = initial_yaw
#
#         # State covariance matrix
#         self.P = np.eye(6) * 1.0
#
#         # Process noise
#         self.Q = np.eye(6) * process_noise
#
#         # Measurement noise
#         self.R = np.eye(3) * measurement_noise  # x, y, z, yaw
#
#         # Measurement matrix (observe position and yaw)
#         self.H = np.zeros((3, 6))
#         self.H[0, 0] = 1  # x
#         self.H[1, 1] = 1  # y
#         self.H[2, 2] = 1  # z
#
#         # Transition matrix
#         self.F = np.eye(6)
#         for i in range(3):  # Update position based on velocity
#             self.F[i, i + 3] = self.dt
#
#         self.decay_velocity_bool = False
#
#     def predict(self):
#         """
#         Predict the state and covariance.
#         """
#         # Predict the next state
#         self.state = self.F @ self.state
#
#         # Predict the next state covariance
#         self.P = self.F @ self.P @ self.F.T + self.Q
#
#         return self.state
#
#     def update(self, measurement):
#         """
#         Update the state based on the new measurement.
#
#         Args:
#             measurement (ndarray): Observed [x, y, z].
#         """
#         # Ensure measurement is a column vector
#         measurement = measurement.reshape(-1)
#
#         # Blend yaw with motion-based estimation
#         #estimated_yaw = np.arctan2(self.state[4], self.state[3])  # vy, vx
#
#         # Full measurement vector [x, y, z, yaw]
#         #measurement_full = np.append(measurement, estimated_yaw)
#
#         # Compute innovation (residual)
#         y = measurement - (self.H @ self.state)
#
#         # Compute innovation covariance
#         S = self.H @ self.P @ self.H.T + self.R
#
#         # Compute Kalman gain
#         K = self.P @ self.H.T @ np.linalg.inv(S)
#
#         # Update the state
#         self.state += (K @ y).flatten()
#
#         # Update the state covariance
#         I = np.eye(self.P.shape[0])
#         self.P = (I - K @ self.H) @ self.P
#
#     def get_state(self):
#         """
#         Get the current state vector.
#
#         Returns:
#             ndarray: Current state [x, y, z, vx, vy, vz, yaw, yaw_rate].
#         """
#         return self.state.flatten()
#
#     def derive_yaw(self):
#         """
#         Derive yaw from velocity.
#         """
#         if self.decay_velocity_bool:
#             self.update_yaw(self.last_valid_yaw)
#             return self.last_valid_yaw
#
#         yaw =  np.arctan2(self.state[4], self.state[3])  # vy, vx
#         self.last_valid_yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
#         return self.last_valid_yaw
#
#     def update_yaw(self, yaw):
#         """
#         Update the yaw in the state vector.
#
#         Args:
#             yaw (float): Yaw angle.
#         """
#         abs_vel = np.linalg.norm(self.state[3:5])
#         self.state[3] = abs_vel * np.cos(yaw)
#         self.state[4] = abs_vel * np.sin(yaw)
#         self.last_valid_yaw = yaw
#
#
#     def decay_velocity(self):
#         self.decay_velocity_bool = True
#         self.state[3:] *= 0.8
#
#     def set_decay_velocity(self, decay_velocity):
#         self.decay_velocity_bool = decay_velocity
#
#     def get_covariance(self):
#         return self.P
#
#     def get_state(self):
#         return self.state
