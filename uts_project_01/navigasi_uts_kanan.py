import math
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

# =========================================================================
# DEFINISI WAYPOINT (Koordinat tidak diubah agar lintasan tetap akurat)
# =========================================================================
WAYPOINT_TRACK_KIRI = [
    (0.60, 2.38),   # 1: Depan pintu masuk
    (2.42, 2.38),   # 2: Sudut naik
    (2.42, 3.04),   # 3: Sudut ke kanan
    (3.66, 3.04),   # 4: Sudut turun
    (3.66, 0.79),   # 5: Lorong panjang bawah
    (7.91, 0.79),   # 6: Sudut naik sisi kanan
    (7.91, 5.86),   # 7: Awal busur
    (7.68, 6.62),   # 8: Busur 30 derajat
    (7.11, 7.16),   # 9: Busur 60 derajat
    (6.36, 7.35),   # 10: Busur 90 derajat (puncak)
    (5.61, 7.15),   # 11: Busur 120 derajat
    (5.06, 6.61),   # 12: Busur 150 derajat
    (4.84, 5.86),   # 13: Akhir busur
    (4.84, 4.27),   # 14: Sudut ke kiri
    (3.94, 4.27),   # 15: Sudut naik
    (3.94, 6.78),   # 16: Masuk ruangan
    (3.00, 6.78),   # 17: Di dalam ruangan
    (1.50, 7.54),   # 18: Menuju celah atas
    (0.28, 7.54),   # 19: Sudut turun ke lorong terakhir
    (0.28, 4.80),   # 20: FINISH
]

# Track kanan adalah cerminan track kiri
WAYPOINT_TRACK_KANAN = [(x, round(-0.1188 - y, 2)) for (x, y) in WAYPOINT_TRACK_KIRI]


def normalisasi_sudut(sudut):
    """Menjaga nilai sudut tetap berada di rentang -pi hingga pi."""
    return math.atan2(math.sin(sudut), math.cos(sudut))


def batasi_nilai(nilai, batas_bawah, batas_atas):
    """Fungsi clamp untuk membatasi nilai agar tidak melebihi batas maksimum/minimum."""
    return max(batas_bawah, min(batas_atas, nilai))


class NodeNavigasiUTSKanan(Node):
    """Node ROS 2 untuk navigasi robot mengikuti waypoint menggunakan odometri roda."""

    def __init__(self):
        super().__init__('node_navigasi_uts_kanan')

        # -------------------------------------------------------------------
        # 1. DEKLARASI PARAMETER
        # -------------------------------------------------------------------
        self.declare_parameter('mode', 'track')
        self.declare_parameter('pilihan_track', 'kanan')  # Default: KANAN
        self.declare_parameter('radius_roda', 0.03)
        self.declare_parameter('jarak_antar_roda', 0.17)
        self.declare_parameter('kecepatan_linear_maks', 0.6)
        self.declare_parameter('kecepatan_angular_maks', 1.5)
        self.declare_parameter('akselerasi_linear', 1.0)
        self.declare_parameter('akselerasi_angular', 2.0)
        self.declare_parameter('gain_kecepatan', 2.5)
        self.declare_parameter('gain_sudut', 2.0)
        self.declare_parameter('batas_putar_di_tempat', 0.5)
        self.declare_parameter('toleransi_waypoint', 0.10)
        self.declare_parameter('toleransi_finish', 0.08)

        # Mengambil nilai parameter
        self.mode = str(self.get_parameter('mode').value).lower()
        pilihan = str(self.get_parameter('pilihan_track').value).lower()
        self.r = float(self.get_parameter('radius_roda').value)
        self.s = float(self.get_parameter('jarak_antar_roda').value)
        self.v_max = float(self.get_parameter('kecepatan_linear_maks').value)
        self.w_max = float(self.get_parameter('kecepatan_angular_maks').value)
        self.acc_lin = float(self.get_parameter('akselerasi_linear').value)
        self.acc_ang = float(self.get_parameter('akselerasi_angular').value)
        self.k_v = float(self.get_parameter('gain_kecepatan').value)
        self.k_theta = float(self.get_parameter('gain_sudut').value)
        self.turn_threshold = float(self.get_parameter('batas_putar_di_tempat').value)
        self.tol_wp = float(self.get_parameter('toleransi_waypoint').value)
        self.tol_finish = float(self.get_parameter('toleransi_finish').value)

        # Menentukan lintasan berdasarkan parameter
        if self.mode == 'track':
            self.daftar_waypoint = WAYPOINT_TRACK_KANAN if pilihan == 'kanan' else WAYPOINT_TRACK_KIRI
            self.nama_track = pilihan
        else:
            self.daftar_waypoint = [(5.0, 0.0)]
            self.nama_track = 'default'

        # -------------------------------------------------------------------
        # 2. STATE VARIABLE
        # -------------------------------------------------------------------
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.theta = 0.0
        
        self.akumulasi_vel_kiri = 0.0
        self.akumulasi_vel_kanan = 0.0
        self.jumlah_sampel = 0
        self.sudah_terima_data_roda = False
        self.waktu_odometri_terakhir = None

        self.idx_wp = 0
        self.sudah_mulai = False
        self.sudah_finish = False
        self.waktu_mulai = None
        self.waktu_log_terakhir = None
        
        self.perintah_v_terakhir = 0.0
        self.perintah_w_terakhir = 0.0
        self.periode_kontrol = 0.05

        # -------------------------------------------------------------------
        # 3. INISIALISASI PUBLISHER & SUBSCRIBER
        # -------------------------------------------------------------------
        self.subscriber_roda = self.create_subscription(
            Float64MultiArray, '/wheel/speed', self.callback_odometri, 10)

        self.publisher_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer_kontrol = self.create_timer(self.periode_kontrol, self.loop_kontrol)

        self.get_logger().info(f"Node Siap! Mode: {self.mode.upper()}, Track: {self.nama_track.upper()}")
        self.get_logger().info("Menunggu data pertama dari topik '/wheel/speed' untuk memulai...")

    # -----------------------------------------------------------------------
    # 4. CALLBACK ODOMETRI
    # -----------------------------------------------------------------------
    def callback_odometri(self, msg):
        if len(msg.data) < 2:
            return

        self.akumulasi_vel_kiri += float(msg.data[0])
        self.akumulasi_vel_kanan += float(msg.data[1])
        self.jumlah_sampel += 1
        self.sudah_terima_data_roda = True

        waktu_sekarang = self.get_clock().now()
        if self.waktu_odometri_terakhir is None:
            self.waktu_odometri_terakhir = waktu_sekarang
            self.reset_sampel()
            return

        dt = (waktu_sekarang - self.waktu_odometri_terakhir).nanoseconds * 1e-9
        if dt <= 0.0 or dt > 0.5:
            self.waktu_odometri_terakhir = waktu_sekarang
            self.reset_sampel()
            return

        vel_kiri = self.akumulasi_vel_kiri / self.jumlah_sampel
        vel_kanan = self.akumulasi_vel_kanan / self.jumlah_sampel
        self.reset_sampel()
        self.waktu_odometri_terakhir = waktu_sekarang

        v_linear_lokal = (self.r / 2.0) * (vel_kiri + vel_kanan)
        v_angular_lokal = (self.r / self.s) * (-vel_kiri + vel_kanan)

        theta_tengah = self.theta + (v_angular_lokal * dt / 2.0)
        delta_x = v_linear_lokal * math.cos(theta_tengah) * dt
        delta_y = v_linear_lokal * math.sin(theta_tengah) * dt

        self.pos_x += delta_x
        self.pos_y += delta_y
        self.theta = normalisasi_sudut(self.theta + (v_angular_lokal * dt))

    def reset_sampel(self):
        self.akumulasi_vel_kiri = 0.0
        self.akumulasi_vel_kanan = 0.0
        self.jumlah_sampel = 0

    # -----------------------------------------------------------------------
    # 5. LOOP KONTROL
    # -----------------------------------------------------------------------
    def loop_kontrol(self):
        if self.sudah_finish:
            self.kirim_perintah(0.0, 0.0, halus=False)
            return

        if not self.sudah_mulai:
            if not self.sudah_terima_data_roda:
                return
            if self.get_clock().now().nanoseconds == 0:
                return 
            
            self.sudah_mulai = True
            self.waktu_mulai = self.get_clock().now()
            self.get_logger().info(">>> START! Robot mulai bergerak mengejar waypoint.")

        target_x, target_y = self.daftar_waypoint[self.idx_wp]
        
        dx = target_x - self.pos_x
        dy = target_y - self.pos_y
        jarak_ke_target = math.hypot(dx, dy)
        
        sudut_target = math.atan2(dy, dx)
        error_sudut = normalisasi_sudut(sudut_target - self.theta)

        is_finish = (self.idx_wp == len(self.daftar_waypoint) - 1)
        batas_toleransi = self.tol_finish if is_finish else self.tol_wp

        if jarak_ke_target <= batas_toleransi:
            if is_finish:
                self.selesaikan_misi()
                return
            
            self.idx_wp += 1
            self.get_logger().info(f"Waypoint {self.idx_wp} tercapai. Lanjut ke berikutnya.")
            return

        if abs(error_sudut) > self.turn_threshold:
            v_target = 0.0
            w_target = batasi_nilai(self.k_theta * error_sudut, -self.w_max, self.w_max)
        else:
            v_target = batasi_nilai(self.k_v * jarak_ke_target, 0.0, self.v_max) * math.cos(error_sudut)
            w_target = batasi_nilai(self.k_theta * error_sudut, -self.w_max, self.w_max)

        self.kirim_perintah(v_target, w_target, halus=True)
        self.catat_status(jarak_ke_target)

    # -----------------------------------------------------------------------
    # 6. FUNGSI BANTUAN
    # -----------------------------------------------------------------------
    def kirim_perintah(self, v, w, halus=True):
        if halus:
            dv = self.acc_lin * self.periode_kontrol
            dw = self.acc_ang * self.periode_kontrol
            v = batasi_nilai(v, self.perintah_v_terakhir - dv, self.perintah_v_terakhir + dv)
            w = batasi_nilai(w, self.perintah_w_terakhir - dw, self.perintah_w_terakhir + dw)
        
        self.perintah_v_terakhir = v
        self.perintah_w_terakhir = w

        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.publisher_cmd.publish(msg)

    def catat_status(self, jarak):
        waktu_sekarang = self.get_clock().now()
        if self.waktu_log_terakhir is not None:
            if (waktu_sekarang - self.waktu_log_terakhir).nanoseconds < 1e9:
                return
        self.waktu_log_terakhir = waktu_sekarang
        
        waktu_berjalan = (waktu_sekarang - self.waktu_mulai).nanoseconds * 1e-9
        self.get_logger().info(
            f"Time: {waktu_berjalan:5.1f}s | WP: {self.idx_wp+1}/{len(self.daftar_waypoint)} | "
            f"Jarak: {jarak:.2f}m | Pos: X={self.pos_x:.2f}, Y={self.pos_y:.2f}"
        )

    def selesaikan_misi(self):
        self.sudah_finish = True
        self.kirim_perintah(0.0, 0.0, halus=False)
        total_waktu = (self.get_clock().now() - self.waktu_mulai).nanoseconds * 1e-9
        
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"FINISH! Robot berhasil menyelesaikan Track {self.nama_track.upper()}")
        self.get_logger().info(f"WAKTU TEMPUH TOTAL: {total_waktu:.2f} DETIK")
        self.get_logger().info(f"Posisi Akhir Estimasi: X={self.pos_x:.2f}, Y={self.pos_y:.2f}")
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = NodeNavigasiUTSKanan()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.kirim_perintah(0.0, 0.0, halus=False)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

