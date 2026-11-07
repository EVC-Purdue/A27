#!/usr/bin/env bash
# https://github.com/EVC-Purdue/A27/issues/15
# Provision a host for one role: jetson, rubik, unoq or dev. Safe to rerun.
set -euo pipefail

usage() { echo "usage: $0 {jetson|rubik|unoq|dev}" >&2; exit 2; }
[ $# -eq 1 ] || usage
ROLE="$1"

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"

preflight() {
  say "Preflight ($ROLE)"
  . /etc/os-release
  echo "    os        $PRETTY_NAME ($VERSION_CODENAME, $(dpkg --print-architecture))"
  echo "    workspace $WS"
  # Jazzy binaries target Ubuntu 24.04 only
  if [ "$VERSION_CODENAME" != noble ]; then
    echo "    ROS 2 $ROS_DISTRO needs Ubuntu 24.04 (noble); this is $VERSION_CODENAME. Stop here." >&2
    exit 1
  fi
}

ros_apt() {
  say "ROS 2 apt repository"
  if grep -Rqs packages.ros.org/ros2 /etc/apt/sources.list /etc/apt/sources.list.d/; then
    echo "    already configured"
    return
  fi
  $SUDO apt-get update
  $SUDO apt-get install -y --no-install-recommends software-properties-common curl gnupg ca-certificates
  $SUDO add-apt-repository -y universe
  $SUDO curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $VERSION_CODENAME main" \
    | $SUDO tee /etc/apt/sources.list.d/ros2.list >/dev/null
}

ros_packages() {
  say "Packages"
  $SUDO apt-get update
  $SUDO apt-get install -y --no-install-recommends \
    "ros-${ROS_DISTRO}-ros-base" \
    "ros-${ROS_DISTRO}-rmw-cyclonedds-cpp" \
    "ros-${ROS_DISTRO}-rosbag2-storage-mcap" \
    python3-colcon-common-extensions python3-rosdep python3-pip python3-pytest \
    build-essential git
}

rosdep_install() {
  say "rosdep"
  [ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || $SUDO rosdep init
  rosdep update
  # shellcheck disable=SC1090
  set +u; . "/opt/ros/${ROS_DISTRO}/setup.bash"; set -u
  rosdep install --from-paths "$WS/src" --ignore-src -y --rosdistro "$ROS_DISTRO"
}

build() {
  say "Build"
  set +u; . "/opt/ros/${ROS_DISTRO}/setup.bash"; set -u
  ( cd "$WS" && colcon build --symlink-install )
  mkdir -p "$WS/data" "$WS/logs" "$WS/figures"
}

case "$ROLE" in
  dev)
    preflight
    ros_apt
    ros_packages
    rosdep_install
    build
    say "Done"
    echo "    source $WS/install/setup.bash"
    echo "    ros2 launch autonomous_kart bringup_sim.launch.py"
    ;;
  jetson|rubik|unoq)
    # TODO(#15): per-role udev rules, systemd units and journald config
    echo "role $ROLE is not provisioned yet (#15)" >&2
    exit 1
    ;;
  *)
    usage
    ;;
esac
