import rospy
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['FreeSans', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 14


class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        d : float
            Half of the track width (m) of TurtleBot3 Burger.
        r : float
            Wheel radius (m) of the TurtleBot3 Burger.
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][0] is timestamp (s),
            u[i][1] is left wheel rotational speed (rad/s), and
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is timestamp (s),
            x[i][1] is bearing (rad),
            x[i][2] is translational position in x (m),
            x[i][3] is translational position in y (m),
            x[i][4] is left wheel rotational position (rad), and
            x[i][5] is right wheel rotational position (rad).
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][0] is timestamp (s),
            y[i][1] is translational position in x (m) when freeze_bearing:=true,
            y[i][1] is distance to the landmark (m) when freeze_bearing:=false,
            y[i][2] is translational position in y (m) when freeze_bearing:=true, and
            y[i][2] is relative bearing (rad) w.r.t. the landmark when
            freeze_bearing:=false.
        x_hat : list
            A list of estimated system states. It should follow the same format
            as x.
        dt : float
            Update frequency of the estimator.
        fig : Figure
            matplotlib Figure for real-time plotting.
        axd : dict
            A dictionary of matplotlib Axis for real-time plotting.
        ln* : Line
            matplotlib Line object for ground truth states.
        ln_*_hat : Line
            matplotlib Line object for estimated states.
        canvas_title : str
            Title of the real-time plot, which is chosen to be estimator type.
        sub_u : rospy.Subscriber
            ROS subscriber for system inputs.
        sub_x : rospy.Subscriber
            ROS subscriber for system states.
        sub_y : rospy.Subscriber
            ROS subscriber for system outputs.
        tmr_update : rospy.Timer
            ROS Timer for periodically invoking the estimator's update method.

    Notes
    ----------
        The frozen bearing is pi/4 and the landmark is positioned at (0.5, 0.5).
    """
    # noinspection PyTypeChecker
    def __init__(self):
        self.d = 0.08
        self.r = 0.033
        self.u = []
        self.x = []
        self.y = []
        self.x_hat = []  # Your estimates go here!
        self.dt = 0.1
        self.fig, self.axd = plt.subplot_mosaic(
            [['xy', 'phi'],
             ['xy', 'x'],
             ['xy', 'y'],
             ['xy', 'thl'],
             ['xy', 'thr']], figsize=(20.0, 10.0))
        self.ln_xy, = self.axd['xy'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_xy_hat, = self.axd['xy'].plot([], 'o-c', label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_y, = self.axd['y'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_y_hat, = self.axd['y'].plot([], 'o-c', label='Estimated')
        self.ln_thl, = self.axd['thl'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thl_hat, = self.axd['thl'].plot([], 'o-c', label='Estimated')
        self.ln_thr, = self.axd['thr'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thr_hat, = self.axd['thr'].plot([], 'o-c', label='Estimated')
        self.canvas_title = 'N/A'
        self.sub_u = rospy.Subscriber('u', Float32MultiArray, self.callback_u)
        self.sub_x = rospy.Subscriber('x', Float32MultiArray, self.callback_x)
        self.sub_y = rospy.Subscriber('y', Float32MultiArray, self.callback_y)
        self.tmr_update = rospy.Timer(rospy.Duration(self.dt), self.update)

    def callback_u(self, msg):
        self.u.append(msg.data)

    def callback_x(self, msg):
        self.x.append(msg.data)
        if len(self.x_hat) == 0:
            self.x_hat.append(msg.data)

    def callback_y(self, msg):
        self.y.append(msg.data)

    def update(self, _):
        raise NotImplementedError

    def plot_init(self):
        self.axd['xy'].set_title(self.canvas_title)
        self.axd['xy'].set_xlabel('x (m)')
        self.axd['xy'].set_ylabel('y (m)')
        self.axd['xy'].set_aspect('equal', adjustable='box')
        self.axd['xy'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].legend()
        self.axd['y'].set_ylabel('y (m)')
        self.axd['y'].legend()
        self.axd['thl'].set_ylabel('theta L (rad)')
        self.axd['thl'].legend()
        self.axd['thr'].set_ylabel('theta R (rad)')
        self.axd['thr'].set_xlabel('Time (s)')
        self.axd['thr'].legend()
        plt.tight_layout()

    def plot_update(self, _):
        self.plot_xyline(self.ln_xy, self.x)
        self.plot_xyline(self.ln_xy_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_yline(self.ln_y, self.x)
        self.plot_yline(self.ln_y_hat, self.x_hat)
        self.plot_thlline(self.ln_thl, self.x)
        self.plot_thlline(self.ln_thl_hat, self.x_hat)
        self.plot_thrline(self.ln_thr, self.x)
        self.plot_thrline(self.ln_thr_hat, self.x_hat)

    def plot_xyline(self, ln, data):
        if len(data):
            x = [d[2] for d in data]
            y = [d[3] for d in data]
            ln.set_data(x, y)
            self.resize_lim(self.axd['xy'], x, y)

    def plot_philine(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            phi = [d[1] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            x = [d[2] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_yline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            y = [d[3] for d in data]
            ln.set_data(t, y)
            self.resize_lim(self.axd['y'], t, y)

    def plot_thlline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thl = [d[4] for d in data]
            ln.set_data(t, thl)
            self.resize_lim(self.axd['thl'], t, thl)

    def plot_thrline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thr = [d[5] for d in data]
            ln.set_data(t, thr)
            self.resize_lim(self.axd['thr'], t, thr)

    # noinspection PyMethodMayBeStatic
    def resize_lim(self, ax, x, y):
        xlim = ax.get_xlim()
        ax.set_xlim([min(min(x) * 1.05, xlim[0]), max(max(x) * 1.05, xlim[1])])
        ylim = ax.get_ylim()
        ax.set_ylim([min(min(y) * 1.05, ylim[0]), max(max(y) * 1.05, ylim[1])])


class OracleObserver(Estimator):
    """Oracle observer which has access to the true state.

    This class is intended as a bare minimum example for you to understand how
    to work with the code.

    Example
    ----------
    To run the oracle observer:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=oracle_observer \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Oracle Observer'

    def update(self, _):
        self.x_hat.append(self.x[-1])


class DeadReckoning(Estimator):
    """Dead reckoning estimator.

    Your task is to implement the update method of this class using only the
    u attribute and x0. You will need to build a model of the unicycle model
    with the parameters provided to you in the lab doc. After building the
    model, use the provided inputs to estimate system state over time.

    The method should closely predict the state evolution if the system is
    free of noise. You may use this knowledge to verify your implementation.

    Example
    ----------
    To run dead reckoning:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=dead_reckoning \
            noise_injection:=true \
            freeze_bearing:=false
    For debugging, you can simulate a noise-free unicycle model by setting
    noise_injection:=false.
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Dead Reckoning'

    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may ONLY use self.u and self.x[0] for estimation
            # Previous state
            x_prev = self.x_hat[-1]
            u = self.u[-1]
            
            # Get time step
            dt = self.dt
            r = self.r
            d = self.d

            # Control inputs
            uL = u[1]
            uR = u[2]

            # State update using unicycle kinematics
            phi = x_prev[1] + (r / (2 * d)) * (uR - uL) * dt
            x = x_prev[2] + (r / 2) * (uL + uR) * np.cos(phi) * dt
            y = x_prev[3] + (r / 2) * (uL + uR) * np.sin(phi) * dt
            theta_L = x_prev[4] + uL * dt
            theta_R = x_prev[5] + uR * dt

            new_state = np.array([self.x[-1][0], phi, x, y, theta_L, theta_R])
            self.x_hat.append(new_state)


class KalmanFilter(Estimator):
    """Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    linear unicycle model at the default bearing of pi/4. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive Kalman filter update rule.

    Attributes:
    ----------
        phid : float
            Default bearing of the turtlebot fixed at pi / 4.

    Example
    ----------
    To run the Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=kalman_filter \
            noise_injection:=true \
            freeze_bearing:=true
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Kalman Filter'
        self.phid = np.pi / 4
        # TODO: Your implementation goes here!
        # You may define the A, C, Q, R, and P matrices below.
        dt = self.dt
        r = self.r
        d = self.d

        # We keep A as identity since we do the actual unicycle kinematics manually
        self.A = np.eye(6)

        # We won't rely heavily on B because we'll do manual kinematics in update()
        self.B = dt * np.array([
            [0, 0],
            [0, 0],
            [r / 2 * np.cos(self.phid), r / 2 * np.cos(self.phid)],
            [r / 2 * np.sin(self.phid), r / 2 * np.sin(self.phid)],
            [1, 0],
            [0, 1]
        ])

        # Measurement matrix C picks out x, y from the state
        #  index 2 -> x
        #  index 3 -> y
        self.C = np.array([
            [0, 0, 1, 0, 0, 0],  
            [0, 0, 0, 1, 0, 0]
        ])

        # Covariances
        self.Q = np.diag([0.01, 0.01, 0.1, 0.1, 0.01, 0.01])
        self.R = np.eye(2) * 0.1
        self.P = np.eye(6) * 0.01

    # noinspection DuplicatedCode
    # noinspection PyPep8Naming
    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may use self.u, self.y, and self.x[0] for estimation
            x_prev = np.array(self.x_hat[-1], dtype=float)
            u_curr = self.u[-1]
            y_curr = self.y[-1]

            dt = self.dt
            r = self.r
            d = self.d

            # Extract from x_prev for readability
            time_prev, phi_prev, x_prevpos, y_prevpos, thl_prev, thr_prev = x_prev

            # Compute linear & angular velocities from wheel speeds
            uL = u_curr[1]
            uR = u_curr[2]
            v = r * (uL + uR) / 2.0            # linear velocity
            w = r * (uR - uL) / (2.0 * d)      # angular velocity

            # Prediction via unicycle kinematics
            time_new = time_prev + dt
            phi_new  = phi_prev + w * dt
            x_newpos = x_prevpos + v * np.cos(phi_prev) * dt
            y_newpos = y_prevpos + v * np.sin(phi_prev) * dt
            thl_new  = thl_prev + uL * dt
            thr_new  = thr_prev + uR * dt

            x_predict = np.array([
                time_new,
                phi_new,
                x_newpos,
                y_newpos,
                thl_new,
                thr_new
            ], dtype=float)

            # Covariance prediction
            P_predict = self.A @ self.P @ self.A.T + self.Q

            # Compute predicted measurement: [x_predict[2], x_predict[3]]
            y_predict = self.C @ x_predict  # => [ x_newpos, y_newpos ]

            # Kalman Gain (with small regularization)
            S = self.C @ P_predict @ self.C.T + self.R + np.eye(2) * 1e-6
            K = P_predict @ self.C.T @ np.linalg.inv(S)

            # Innovation (measurement - prediction)
            # y_curr: [t, x_meas, y_meas], so y_curr[1] = x_meas, y_curr[2] = y_meas
            innovation = np.array([y_curr[1], y_curr[2]]) - y_predict

            # State correction
            x_new = x_predict + K @ innovation
            x_new[0] = time_new  # lock in the predicted time

            # Covariance update
            P_new = (np.eye(6) - K @ self.C) @ P_predict

            # Store the updated estimate
            self.x_hat.append(x_new)
            self.P = P_new

# noinspection PyPep8Naming
class ExtendedKalmanFilter(Estimator):
    """Extended Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    unicycle model and linearize it at every operating point. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive extended Kalman filter update rule.

    Hint: You may want to reuse your code from DeadReckoning class and
    KalmanFilter class.

    Attributes:
    ----------
        landmark : tuple
            A tuple of the coordinates of the landmark.
            landmark[0] is the x coordinate.
            landmark[1] is the y coordinate.

    Example
    ----------
    To run the extended Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=extended_kalman_filter \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Extended Kalman Filter'
        self.landmark = (0.5, 0.5)
        # TODO: Your implementation goes here!
        # You may define the Q, R, and P matrices below.

        self.Q = np.diag([
            0,    # time (if you store it)
            0.20,    # phi
            0.25,    # x
            0.09,    # y
            0,   # theta_L
            0    # theta_R
        ])

        # (B) Measurement noise R (moderate)
        self.R = np.diag([
            0.05,    # distance
            0.01     # bearing
        ])

        # (C) Initial covariance P (moderate)
        self.P = np.diag([
            0,  # time
            0.2,   # phi
            0.2,   # x
            0.1,   # y
            0,  # theta_L
            0   # theta_R
        ])

    # noinspection DuplicatedCode
    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may use self.u, self.y, and self.x[0] for estimation
            x_prev = np.array(self.x_hat[-1], dtype=float)
            u_curr = self.u[-1]
            y_curr = self.y[-1]

            dt = self.dt
            r  = self.r
            d  = self.d
            lx, ly = self.landmark

            # Extract old states and control input
            t_old, phi_old, x_old, y_old, thL_old, thR_old = x_prev
            uL = u_curr[1]
            uR = u_curr[2]

            # Compute linear and angular velocities
            v = r * (uL + uR) / 2.0
            w = r * (uR - uL) / (2.0 * d)

            # Predict new states using unicycle kinematics
            t_new    = t_old + dt
            phi_new  = phi_old + w * dt
            x_new    = x_old + v * np.cos(phi_old) * dt
            y_new    = y_old + v * np.sin(phi_old) * dt
            thL_new  = thL_old + uL * dt
            thR_new  = thR_old + uR * dt

            x_predict = np.array([
                t_new,
                phi_new,
                x_new,
                y_new,
                thL_new,
                thR_new
            ], dtype=float)

            # Linearize f about (x_prev, u_curr) -> A = df/dx
            A = self.jacobian_f(x_prev, u_curr)

            # Predict Covariance
            P_predict = A @ self.P @ A.T + self.Q

            # Measurement Prediction: y_pred = h(x_predict)
            y_pred = self.h(x_predict)

            # Linearize h about x_predict -> C = dh/dx
            C = self.jacobian_h(x_predict)

            # Kalman Gain
            S = C @ P_predict @ C.T + self.R + np.eye(2) * 1e-3
            K = P_predict @ C.T @ np.linalg.inv(S)

            # y_curr = [t, dist_meas, bearing_meas]
            innovation = np.array([y_curr[1], y_curr[2]]) - y_pred
            innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi

            # Update State & Covariance
            x_new_est = x_predict + K @ innovation
            x_new_est[0] = t_new
            x_new_est[1] = (x_new_est[1] + np.pi) % (2 * np.pi) - np.pi

            P_new = (np.eye(6) - K @ C) @ P_predict

            self.x_hat.append(x_new_est)
            self.P = P_new
    
    def jacobian_f(self, x_state, u):
        dt = self.dt
        r = self.r
        d = self.d

        _, phi_old, x_old, y_old, _, _ = x_state
        uL = u[1]
        uR = u[2]

        v = r * (uL + uR) / 2.0
        w = r * (uR - uL) / (2.0 * d)

        A = np.eye(6)
        A[2, 1] = -v * dt * np.sin(phi_old)
        A[3, 1] = v * dt * np.cos(phi_old)

        return A

    def h(self, x_state):
        _, phi, x, y, _, _ = x_state
        lx, ly = self.landmark

        dx = lx - x
        dy = ly - y

        dist = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx) - phi
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]

        return np.array([dist, bearing], dtype=float)

    def jacobian_h(self, x_state):
        """
        Jacobian of h w.r.t x for [distance, bearing].
        x_state = [t, phi, x, y, thL, thR].
        Returns C (2x6).
        """
        _, phi, x, y, _, _ = x_state
        lx, ly = self.landmark

        dx = lx - x
        dy = ly - y
        dist = np.sqrt(dx**2 + dy**2)

        C = np.zeros((2, 6))

        # distance partials
        if dist > 1e-9:  # avoid division by zero
            C[0, 2] = -(dx) / dist   # partial dist / partial x
            C[0, 3] = -(dy) / dist   # partial dist / partial y

        denom = dx**2 + dy**2
        if denom > 1e-9:
            C[1, 1] = -1.0  # partial wrt phi
            C[1, 2] = dy / denom
            C[1, 3] = -dx / denom

        return C
