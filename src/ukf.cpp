#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace {

constexpr double kM_PI = 3.1415926535;
constexpr double kChiSquareLimitLidar = 5.991;
constexpr double kChiSquareLimitRadar = 7.815;

double NormalizeAngle(double angle) {
  while (angle > kM_PI) angle -= 2.*kM_PI;
  while (angle <-kM_PI) angle += 2.*kM_PI;
  return angle;
}

void normalize_radar(VectorXd* x) {
  (*x)(1) = NormalizeAngle((*x)(1));
}

void normalize_lidar(VectorXd* x) {
  // Intentionally left empty.
}
}  // namespace

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_.fill(0);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);
  P_(2,2) = 10;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;

  Q_ = MatrixXd(2, 2);
  Q_.fill(0);
  Q_(0, 0) = std_a_ * std_a_;
  Q_(1, 1) = std_yawdd_ * std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State dimension
  n_x_ = x_.size();
  n_aug_ = n_x_ + 2;
  n_sig_ = 2 * n_aug_ + 1;

  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(n_sig_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < weights_.size(); i++) {
      weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0,std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
      if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
          // Perform the conversion from radar measurements to the cartesian.
          float rho = meas_package.raw_measurements_[0];
          float phi = meas_package.raw_measurements_[1];
          float px = rho * cos(phi);
          float py = rho * sin(phi);
          x_ << px, py, 0, 0, 0;
      } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
          x_ << meas_package.raw_measurements_[0],
                meas_package.raw_measurements_[1],
                0, 0, 0;
      }
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
      return;
  }
  // Calculate the timestep between measurements in seconds
  double dt = (meas_package.timestamp_ - time_us_);
  dt /= 1000000.0; // convert micros to s
  time_us_ = meas_package.timestamp_;
  Prediction(dt);

  switch (meas_package.sensor_type_) {
      case MeasurementPackage::LASER:
          if (use_laser_) {
              UpdateLidar(meas_package);
          }
          break;
      case MeasurementPackage::RADAR:
          if (use_radar_) {
              UpdateRadar(meas_package);
          }
          break;
  }
}

MatrixXd UKF::BuildSigmaPointsAugmentedState() const {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(n_x_) = x_;

  //create augmented covariance matrix
  P_aug.block(0, 0, n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q_;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  MatrixXd shiftMatrix = sqrt(lambda_ + n_aug_) * A;

  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) = shiftMatrix.colwise() + x_aug;
  Xsig_aug.block(0, 1 + n_aug_, n_aug_, n_aug_) =
          (-1 * shiftMatrix).colwise() + x_aug;

  return Xsig_aug;
}

VectorXd UKF::PredictStateChange(const VectorXd& x_aug,
                                 double delta_t,
                                 double delta_t_2) const {
  //extract values for better readability
  double p_x = x_aug(0);
  double p_y = x_aug(1);
  double v = x_aug(2);
  double yaw = x_aug(3);
  double yawd = x_aug(4);
  double nu_a = x_aug(5);
  double nu_yawdd = x_aug(6);

  VectorXd noise(n_x_);
  noise(0) = 1.0/2 * delta_t_2 * cos(yaw) * nu_a;
  noise(1) = 1.0/2 * delta_t_2 * sin(yaw) * nu_a;
  noise(2) = delta_t * nu_a;
  noise(3) = 1.0/2 * delta_t_2 * nu_yawdd;
  noise(4) = delta_t * nu_yawdd;

  VectorXd change(n_x_);
  if (yawd == 0) {
    change(0) = v * cos(yaw) * delta_t;
    change(1) = v * sin(yaw) * delta_t;
  } else {
    change(0) = v/yawd * (sin(delta_t * yawd + yaw) - sin(yaw));
    change(1) = v/yawd * (- 1 * cos(delta_t * yawd + yaw) + cos(yaw));
  }
  change(2) = 0;
  change(3) = yawd * delta_t;
  change(4) = 0;

  return (x_aug.head(n_x_) + change + noise);
}
/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  MatrixXd Xsig_aug = BuildSigmaPointsAugmentedState();
  // Save some computation.
  double delta_t_2 = delta_t * delta_t;
  // Calculate the predicted sigma points.
  for (size_t col = 0; col < Xsig_aug.cols(); ++col) {
    Xsig_pred_.col(col) = PredictStateChange(Xsig_aug.col(col),
                                             delta_t,
                                             delta_t_2);
  }

  //predicted state mean
  x_.fill(0.0);
  x_ = Xsig_pred_ * weights_;

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < Xsig_pred_.cols(); i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

MatrixXd UKF::PredictLidarMeasurement() const {
  constexpr int n_z = 2;
  return Xsig_pred_.block(0, 0, 2, n_sig_);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  MatrixXd Zsig = PredictLidarMeasurement();
  double nis = UpdateState(meas_package, Zsig, R_lidar_, &normalize_lidar);
  ++measurements_nr_;
  if (nis > kChiSquareLimitLidar) {
      ++under_estimated_nr_;
  }
}

MatrixXd UKF::PredictRadarMeasurement() const {
  constexpr int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  //transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {  //2n+1 sigma points
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }
  return Zsig;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  MatrixXd Zsig = PredictRadarMeasurement();
  double nis = UpdateState(meas_package, Zsig, R_radar_, &normalize_radar);
  ++measurements_nr_;
  if (nis > kChiSquareLimitRadar) {
    ++under_estimated_nr_;
  }
}

double UKF::UpdateState(const MeasurementPackage& meas_package,
                        const MatrixXd& Zsig,
                        const MatrixXd& MeasurementNoise,
                        NormalizerFn normalize) {
  int n_z = Zsig.rows();
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < n_sig_; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //2n+1 sigma points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    normalize(&z_diff);
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + MeasurementNoise;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //2n+1 sigma points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    normalize(&z_diff);

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  //angle normalization
  normalize(&z_diff);

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS
  return z_diff.transpose() * S.inverse() * z_diff;
}

double UKF::getUnderEstimatedFrequency() const {
  return measurements_nr_ == 0
          ? 0
          : (1.0 * under_estimated_nr_) / measurements_nr_;
}
