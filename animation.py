
# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from scipy.linalg import svd


# =========================
# Calibration Parameters
# =========================

# --- Dataset 1 (README1.txt) ---
ds1_gyro_matrix = np.eye(3)
ds1_gyro_offset = np.array([-2.242224,  2.963463, -0.718397])

ds1_acc_matrix = np.array([
    [ 1.000966,  -0.002326418, -0.0006995499],
    [-0.002326379, 0.9787045,  -0.001540918 ],
    [-0.0006995811,-0.001540928, 1.00403    ]
])
ds1_acc_offset = np.array([-3.929942, -13.74679, 60.67546])

ds1_magn_matrix = np.array([
    [ 0.9192851, -0.02325168,  0.003480837],
    [-0.02325175, 0.914876,    0.004257396],
    [ 0.003481006,0.004257583, 0.8748001  ]
])
ds1_magn_offset = np.array([-95.67974, -244.9142, 17.71132])


# --- Dataset 2 (README5.txt)---
ds2_gyro_matrix = np.eye(3)
ds2_gyro_offset = np.array([-2.804399,  1.793105,  0.3411708])

ds2_acc_matrix = np.array([
    [1.002982,   9.415505e-05,  0.004346743],
    [9.04459e-05,1.002731,     -0.001444198],
    [0.004346536,-0.001444751,  1.030587   ]
])
ds2_acc_offset = np.array([3.602701, -20.96658, 54.97186])

ds2_magn_matrix = np.array([
    [ 1.013437,  -0.04728858, -0.001861475],
    [-0.04728862, 1.004832,    0.008222118],
    [-0.001861605,0.008221965, 0.9439077  ]
])
ds2_magn_offset = np.array([-150.4098, 74.62431, 630.9805])

def load_dataset(filename):
    return pd.read_csv(
        filename,
        delimiter="\t",
        dtype={
            "LogMode": int,
            "LogFreq": int,
            "Timestamp": int,
            "AccX": float, "AccY": float, "AccZ": float,
            "GyroX": float, "GyroY": float, "GyroZ": float,
            "MagnX": float, "MagnY": float, "MagnZ": float,
        },
    )

def calibrate_data(df, cols, matrix, offset):
    if len(cols) != 3:
        raise ValueError("Exactly three columns must be specified.")

    M = np.asarray(matrix, dtype=float)
    b = np.asarray(offset, dtype=float)

    if M.shape != (3, 3):
        raise ValueError("Calibration matrix must be 3x3.")
    if b.shape != (3,):
        raise ValueError("Offset vector must have length 3.")

    X = df[cols].to_numpy()

    # Since X is Nx3, use X @ M.T - b
    df.loc[:, cols] = X @ M.T - b

# =========================
# Load and Calibrate Datasets
# =========================
df1 = load_dataset("center_sternum.txt")

calibrate_data(df1, ["AccX",  "AccY",  "AccZ"],  ds1_acc_matrix,  ds1_acc_offset)
calibrate_data(df1, ["GyroX", "GyroY", "GyroZ"], ds1_gyro_matrix, ds1_gyro_offset)
calibrate_data(df1, ["MagnX", "MagnY", "MagnZ"], ds1_magn_matrix, ds1_magn_offset)

df2 = load_dataset("1_Stave_supine_static.txt")

calibrate_data(df2, ["AccX",  "AccY",  "AccZ"],  ds2_acc_matrix,  ds2_acc_offset)
calibrate_data(df2, ["GyroX", "GyroY", "GyroZ"], ds2_gyro_matrix, ds2_gyro_offset)
calibrate_data(df2, ["MagnX", "MagnY", "MagnZ"], ds2_magn_matrix, ds2_magn_offset)

# =========================
# Select dataset to show
# =========================
df = df2

def preprocess_time_metadata(df):
    # Validate LogMode
    if not (df['Log Mode'] == 5).all():
        raise ValueError("Unsupported Log Mode")

    # Validate LogFreq
    log_freq = df['Log Freq'].iloc[0]
    if not (df['Log Freq'] == log_freq).all():
        raise ValueError("Log frequency is not constant")

    # Drop columns no longer needed
    df.drop(columns=['Log Mode', 'Log Freq', 'Timestamp'], inplace=True)

    # Create the time column
    df['Time'] = df.index / log_freq

    return log_freq

# =========================
# Apply time preprocessing
# =========================
preprocess_time_metadata(df)


def apply_pca(df, columns):
    X = df[columns].to_numpy()
    N = X.shape[0]

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Perform SVD
    U, sigma, Vt = svd(X_centered, full_matrices=False)
    
    # Compute variability explained by each principal component
    eigenvalues = (sigma**2) / (N - 1)
    variability = eigenvalues / eigenvalues.sum()
    
    # Principal axes
    principal_axes = Vt.T
    
    # Project data onto principal components
    projected_data = X_centered @ principal_axes

    # Remove original columns. do not
    ####df.drop(columns=columns, inplace=True)

    # Add projected components to DataFrame
    prefix = columns[0][:-1]
    pc_columns = [f'{prefix}PC{i+1}' for i in range(X.shape[1])]
    for i, pc_col in enumerate(pc_columns):
        df[pc_col] = projected_data[:, i]
    
    return variability, principal_axes

# =========================
# Apply PCA to each sensor group
# =========================

variability_acc, pc_acc_axes = apply_pca(df, ['AccX', 'AccY', 'AccZ'])
variability_gyro, pc_gyro_axes = apply_pca(df, ['GyroX', 'GyroY', 'GyroZ'])
variability_magn, pc_magn_axes = apply_pca(df, ['MagnX', 'MagnY', 'MagnZ'])


def quaternion_product(p, q):
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q

    pq = [
        p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
        p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
        p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
        p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0
    ]

    return pq

def quaternion_conjugate(p):
    qconj = [p[0], -p[1], -p[2], -p[3]]
    return qconj

def quat2_rotation_matrix(q):
    R = np.eye(3)
    R[0, 0] = 1 - 2 * (q[2]**2 + q[3]**2)
    R[1, 1] = 1 - 2 * (q[1]**2 + q[3]**2)
    R[2, 2] = 1 - 2 * (q[1]**2 + q[2]**2)

    R[0, 1] = 2 * (q[1] * q[2] - q[3] * q[0])
    R[0, 2] = 2 * (q[1] * q[3] + q[2] * q[0])
    
    R[1, 0] = 2 * (q[1] * q[2] + q[3] * q[0])
    R[2, 0] = 2 * (q[1] * q[3] - q[2] * q[0])
    
    R[1, 2] = 2 * (q[2] * q[3] - q[1] * q[0])
    R[2, 1] = 2 * (q[2] * q[3] + q[1] * q[0])
    
    return R

fig = plt.figure(figsize=(8,6))
ax3d = fig.add_subplot(111, projection='3d')

# Fix the axes scaling 
ax3d.set_box_aspect([1,1,1])
ax3d.view_init(40,20)

# Axis limits
ax3d.set_xlim([-1,1])
ax3d.set_ylim([-1,1])
ax3d.set_zlim([-1,1])
ax3d.set_xticks([])
ax3d.set_yticks([])
ax3d.set_zticks([])



# Define arrow directions and lengths
arrow_directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Initial quivers
quiv_quat = ax3d.quiver([0,0,0],[0,0,0],[0,0,0],
                        [1,0,0],[0,1,0],[0,0,1],
                        color=['r','g','b'], arrow_length_ratio=0.1)

quiv_gyro = ax3d.quiver(0,0,0,0,0,0, color='c',arrow_length_ratio=0.1)
quiv_acc  = ax3d.quiver(0,0,0,0,0,0, color='m',arrow_length_ratio=0.1)
quiv_magn  = ax3d.quiver(0,0,0,0,0,0, color='g',arrow_length_ratio=0.1)

pc_gyro_quiv = [ax3d.quiver(0,0,0,*pc_gyro_axes[i],color='c',arrow_length_ratio=0.1) for i in range(3)]
pc_acc_quiv = [ax3d.quiver(0,0,0,*pc_acc_axes[i],color='m',arrow_length_ratio=0.1) for i in range(3)]
pc_mag_quiv = [ax3d.quiver(0,0,0,*pc_magn_axes[i],color='g',arrow_length_ratio=0.1) for i in range(3)]

# Normalize factors
mod_gyro_norm = np.mean(np.linalg.norm(df[['GyroX','GyroY','GyroZ']].values, axis=1))
mod_acc_norm  = np.mean(np.linalg.norm(df[['AccX','AccY','AccZ']].values, axis=1))
mod_magn_norm  = np.mean(np.linalg.norm(df[['MagnX','MagnY','MagnZ']].values, axis=1))

fig.canvas.draw()

q_prev = [1, 0, 0, 0]

# --- Slider ---
ax_slider = plt.axes([0.2,0.05,0.6,0.03])
time_slider = Slider(ax_slider,"Time",0,len(df)-1,valinit=0,valstep=1)

# --- Toggle buttons ---
ax_check = plt.axes([0.02,0.4,0.15,0.2])
check = CheckButtons(ax_check,
                     ['Quaternions', 'Acc', 'Gyro', 'Magn', 'Acc PC axes','Gyro PC axes','Magn PC axes'],
                     [True,True,True,True,True,True,True])

def update(val):
    global quiv_quat, quiv_gyro, quiv_acc, quiv_magn, q_prev, arrow_directions
    
    row = df.iloc[time_slider.val]

    q_curr = np.array([row['qw'],row['qi'],row['qj'],row['qk']])

    qdelta = quaternion_product(q_curr, quaternion_conjugate(q_prev))
    Rquat = quat2_rotation_matrix(qdelta)

    q_prev = q_curr

    arrow_dirs = (Rquat @ arrow_directions.T).T

    gyro = np.array([row['GyroX'],row['GyroY'],row['GyroZ']])
    gyro_normed = gyro / mod_gyro_norm

    acc = np.array([row['AccX'],row['AccY'],row['AccZ']])
    acc_normed = acc / mod_acc_norm

    magn = np.array([row['MagnX'],row['MagnY'],row['MagnZ']])
    magn_normed = magn / mod_magn_norm

    if quiv_quat.get_visible():
        quiv_quat.remove()
        quiv_quat = ax3d.quiver([0,0,0],[0,0,0],[0,0,0],
                            arrow_dirs[:,0],arrow_dirs[:,1],arrow_dirs[:,2],
                            color=['b','r','y'],arrow_length_ratio=0.1)
        
    if quiv_gyro.get_visible():
        quiv_gyro.remove()
        quiv_gyro = ax3d.quiver(0,0,0,*gyro_normed,color='c',arrow_length_ratio=0.1)

    if quiv_acc.get_visible():
        quiv_acc.remove()
        quiv_acc  = ax3d.quiver(0,0,0,*acc_normed, color='m',arrow_length_ratio=0.1)

    if quiv_magn.get_visible():
        quiv_magn.remove()
        quiv_magn  = ax3d.quiver(0,0,0,*magn_normed, color='g',arrow_length_ratio=0.1)
    

    fig.canvas.draw_idle()

time_slider.on_changed(update)

def toggle(label):
    if label == 'Quaternions':
        quiv_quat.set_visible(not quiv_quat.get_visible())

    if label == 'Acc':
        quiv_acc.set_visible(not quiv_acc.get_visible())

    if label == 'Gyro':
        quiv_gyro.set_visible(not quiv_gyro.get_visible())

    if label == 'Magn':
        quiv_magn.set_visible(not quiv_magn.get_visible())

    if label == 'Acc PC axes':
        for q in pc_acc_quiv:
            q.set_visible(not q.get_visible())

    if label == 'Gyro PC axes':
        for q in pc_gyro_quiv:
            q.set_visible(not q.get_visible())

    if label == 'Magn PC axes':
        for q in pc_mag_quiv:
            q.set_visible(not q.get_visible())

    fig.canvas.draw_idle()

check.on_clicked(toggle)

plt.show()
